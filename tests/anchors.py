"""
NUMERICAL REGRESSION ANCHORS for the pfpiml solvers.

Every published number in the paper comes from these code paths, so any refactor must leave
them BITWISE unchanged. This script runs a tiny (seconds-long) version of each example, records
the reaction force F at every load step to full precision, and compares against a stored
baseline.

The runs are deliberately small (2 load steps, ~30 Adam iterations, coarse encoding) -- they are
NOT physics, they are fingerprints: they exercise the full pipeline (geometry map, encoding,
kinematics, energy split, sampler with and without a previous model, force extraction, plots,
checkpointing) and any change in the arithmetic shows up in the digits.

Usage
  capture a baseline (run this BEFORE refactoring):
      python tests/anchors.py --save tests/anchors_baseline.json
  check the current tree against it (run this AFTER):
      python tests/anchors.py --check tests/anchors_baseline.json
  a single case (into a scratch file -- NEVER over the baseline, which must stay complete):
      python tests/anchors.py --save one_case.json --only sens_pf2

Determinism: every case fixes --seed, and the solvers seed torch/numpy/Sobol from it. Results are
reproducible on the SAME machine and device; do not compare a CPU baseline against a GPU run.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# small-but-complete settings shared by every case: 2 load steps so that BOTH the cold start
# (prev_model=None) and the warm/adaptive path (sampler density built from prev_model, drive
# field, irreversibility against phi_prev) are exercised.
SMALL = ["--nsteps", "2", "--iters", "30", "--iters0", "40", "--N", "800",
         "--levels", "32,64", "--enc_start", "0,0", "--grid_n", "32",
         "--width", "32", "--depth", "2", "--seed", "0", "--fem", "NONE"]

CASES = {
    # ---- the 4 SEN-family examples, 2nd order (the production defaults) ----
    "sens_pf2":   ["--delta_max", "5e-4"],
    "sent_pf2":   ["--delta_max", "5e-4", "--loading", "tension"],
    "bifurc_pf2": ["--delta_max", "5e-4", "--split", "iso", "--fixed_notch",
                   "--rho_new", "2e5"],
    "coal_pf2":   ["--delta_max", "5e-4", "--loading", "tension",
                   "--cracks", "0.25,0.35,0.35,0.45;0.45,0.45,0.55,0.55;0.65,0.55,0.75,0.65"],
    # ---- 4th-order (Borden): the extra autodiff pass + the modified crack functional ----
    "sens_pf4":   ["--delta_max", "5e-4", "--pf_order", "4"],
    "sent_pf4":   ["--delta_max", "5e-4", "--pf_order", "4", "--loading", "tension"],
    # ---- energy-split variants (ablation paths) ----
    "spectral":   ["--delta_max", "5e-4", "--split", "spectral"],
    "amor":       ["--delta_max", "5e-4", "--split", "amor"],
    # ---- geometry map: the distorted-patch pullback null test ----
    "distort":    ["--delta_max", "5e-4", "--geo", "distort"],
    # ---- curved NURBS patch (ring) ----
    "ring":       ["--geo", "ring", "--ring_ri", "5.0", "--ring_ro", "20.0",
                   "--ring_notch", "3.0", "--E", "210e3", "--nu", "0.3", "--Gc", "2.7",
                   "--l", "0.2", "--U_ref", "0.033", "--gc_grip", "0", "--grip_l", "0",
                   "--k_bc", "1e6", "--delta_max", "6e-4"],
    # ---- plate with a hole: domain mask + nucleation (no pre-existing crack) ----
    "nucleate":   ["--mode", "nucleate", "--loading", "tension_x", "--cracks", "none",
                   "--hole_r", "0.1", "--gc_grip", "1.0", "--delta_max", "5e-4",
                   "--rho_new", "1.5e5"],
    "kirsch":     ["--mode", "kirsch", "--loading", "tension_x", "--cracks", "none",
                   "--hole_r", "0.1", "--delta_max", "5e-4"],
}

# the documented entry script for each case
SCRIPT = {
    "sens_pf2":   "examples/sen_shear.py",
    "sent_pf2":   "examples/sen_tension.py",
    "bifurc_pf2": "examples/branching.py",
    "coal_pf2":   "examples/coalescence.py",
    "sens_pf4":   "examples/sen_shear.py",
    "sent_pf4":   "examples/sen_tension.py",
    "spectral":   "examples/sen_shear.py",
    "amor":       "examples/sen_shear.py",
    "distort":    "examples/sen_shear.py",
    "ring":       "examples/ring.py",
    "nucleate":   "examples/nucleation.py",
    "kirsch":     "examples/kirsch.py",
}

# the pre-package entry points, kept working by the shims at the repository root (--legacy)
LEGACY = {c: "stageB_msenc.py" for c in CASES}
LEGACY["ring"] = "stageB_msenc_ring.py"
LEGACY["nucleate"] = "stageB_msenc_nucleation.py"
LEGACY["kirsch"] = "stageB_msenc_nucleation.py"

# most cases are fingerprinted by their force-displacement curve; the elastic Kirsch check does
# no load stepping, so it is fingerprinted by the hoop stress it measures around the hole
ARTIFACT = {c: ("FD.txt", 1) for c in CASES}
ARTIFACT["kirsch"] = ("kirsch_scf.txt", 1)

def _read_artifact(out_dir, case):
    """Read the case's fingerprint file into [[col0, col1, ...], ...]."""
    name, _ = ARTIFACT[case]
    path = os.path.join(out_dir, name)
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(t) for t in line.split()])
    return rows

def run_case(name, entry, workdir, python_exe):
    out = os.path.join(workdir, name)
    if os.path.exists(out):
        shutil.rmtree(out)
    cmd = [python_exe, entry] + CASES[name] + SMALL + ["--out", out]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    dt = time.time() - t0
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "")[-1500:]
        return {"ok": False, "error": tail, "seconds": dt}
    fd = _read_artifact(out, name)
    if not fd:
        return {"ok": False, "error": f"no {ARTIFACT[name][0]} produced", "seconds": dt}
    return {"ok": True, "FD": fd, "seconds": dt}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", type=str, default=None, help="write results to this JSON")
    ap.add_argument("--check", type=str, default=None, help="compare against this JSON baseline")
    ap.add_argument("--only", type=str, default=None, help="comma-separated case names")
    ap.add_argument("--workdir", type=str, default=os.path.join(ROOT, "runs", "_anchors"))
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--entry", type=str, default=None,
                    help="override the entry script for ALL cases")
    ap.add_argument("--legacy", action="store_true",
                    help="drive the runs through the pre-package entry points at the repository "
                         "root (stageB_msenc*.py) instead of examples/, to check the shims")
    a = ap.parse_args()

    names = list(CASES) if not a.only else [s.strip() for s in a.only.split(",")]
    if a.legacy and not os.path.exists(os.path.join(ROOT, LEGACY["sens_pf2"])):
        sys.exit("--legacy needs the pre-package entry points (stageB_msenc*.py) at the "
                 "repository root; this tree does not ship them, so run without --legacy.")
    os.makedirs(a.workdir, exist_ok=True)
    results = {}
    for n in names:
        entry = a.entry or (LEGACY[n] if a.legacy else SCRIPT[n])
        r = run_case(n, entry, a.workdir, a.python)
        results[n] = r
        if r["ok"]:
            col = ARTIFACT[n][1]
            vals = [row[col] for row in r["FD"]]
            shown = "  ".join(f"{v:.10f}" for v in vals[:4]) + (" ..." if len(vals) > 4 else "")
            print(f"  {n:12s} OK   {r['seconds']:5.1f}s   {len(vals)} values: {shown}")
        else:
            print(f"  {n:12s} FAIL {r['seconds']:5.1f}s")
            print("      " + r["error"].replace("\n", "\n      ")[:900])
        sys.stdout.flush()

    if a.save:
        if a.only:
            print("\nNOTE: --only with --save writes a PARTIAL baseline. Do not let it overwrite "
                  "the full one: a later --check would report every missing case as a failure.")
        with open(a.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nbaseline written -> {a.save}")

    if a.check:
        with open(a.check) as f:
            base = json.load(f)
        bad = []
        compared = 0
        for n in names:
            cur, ref = results.get(n), base.get(n)
            if ref is None:
                # NOT a skip: a case the baseline does not contain is a case this run did not
                # check, and reporting it as passing would turn the guard silently off
                bad.append(f"{n}: no entry in the baseline (re-capture the baseline with --save)")
                continue
            if not cur["ok"] or not ref["ok"]:
                bad.append(f"{n}: run failed")
                continue
            if len(cur["FD"]) != len(ref["FD"]):
                bad.append(f"{n}: {len(ref['FD'])} steps -> {len(cur['FD'])}")
                continue
            col = ARTIFACT[n][1]
            compared += 1
            for i, (rc, rr) in enumerate(zip(cur["FD"], ref["FD"])):
                if rc[col] != rr[col]:
                    bad.append(f"{n} row {i}: {rr[col]!r} -> {rc[col]!r} "
                               f"(rel {abs(rc[col]-rr[col])/max(abs(rr[col]),1e-30):.3e})")
                    break                                   # one line per case is enough
        print()
        if bad:
            print("ANCHORS CHANGED:")
            for b in bad:
                print("  " + b)
            sys.exit(1)
        # the count is of cases actually COMPARED against the baseline, not of cases run
        print(f"ALL {compared} ANCHORS BITWISE IDENTICAL")

if __name__ == "__main__":
    main()
