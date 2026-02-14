#!/usr/bin/env python3
# check_foam_case_integrity.py
#
# Scans an OpenFOAM decomposed case and reports:
#   1) missing time folders (globally missing vs per-processor missing)
#   2) incomplete time folders where required field files are missing (e.g., U, p)
#
# Usage examples:
#   python3 check_foam_case_integrity.py --run runs/run_20260211_233244_spin20_np16 \
#       --t-start 33311 --t-end 33511 --dt 0.5 --fields U p --out report.json
#
#   # If you prefer relative time window (0..200) from t_spin:
#   python3 check_foam_case_integrity.py --run runs/run_20260211_233244_spin20_np16 \
#       --t-spin 33311 --t0 0 --t1 200 --dt 0.5 --fields U p

import argparse, json, math, os, re, sys
from pathlib import Path
from collections import defaultdict

NUMERIC_RE = re.compile(r"^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$")

def is_numeric_dir(p: Path) -> bool:
    return p.is_dir() and NUMERIC_RE.match(p.name) is not None

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory (case), e.g. runs/run_..._np16")
    ap.add_argument("--fields", nargs="+", default=["U", "p"], help="required field files to check")
    ap.add_argument("--dt", type=float, required=True, help="expected time step, e.g. 0.5")
    ap.add_argument("--tol", type=float, default=1e-6, help="tolerance for snapping times to dt grid")
    ap.add_argument("--out", default="", help="optional path to write JSON report")
    # absolute window
    ap.add_argument("--t-start", type=float, default=None)
    ap.add_argument("--t-end", type=float, default=None)
    # relative window
    ap.add_argument("--t-spin", type=float, default=None)
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    return ap.parse_args()

def resolve_window(a):
    if a.t_start is not None and a.t_end is not None:
        return float(a.t_start), float(a.t_end)
    if a.t_spin is not None and a.t0 is not None and a.t1 is not None:
        t0 = float(a.t_spin) + float(a.t0)
        t1 = float(a.t_spin) + float(a.t1)
        return t0, t1
    raise SystemExit("Provide either (--t-start --t-end) or (--t-spin --t0 --t1).")

def snap_tick(t, t_start, dt, tol):
    k = int(round((t - t_start) / dt))
    t_hat = t_start + k * dt
    if abs(t - t_hat) <= tol:
        return k, t_hat
    return None, None

def main():
    a = parse_args()
    run_dir = Path(a.run).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    t_start, t_end = resolve_window(a)
    dt = float(a.dt)
    if dt <= 0:
        raise SystemExit("--dt must be > 0")

    # Expected ticks in [t_start, t_end] inclusive
    n_steps = int(round((t_end - t_start) / dt))
    if n_steps < 0:
        raise SystemExit("Bad window: t_end < t_start")
    expected_ticks = list(range(0, n_steps + 1))
    expected_times = {k: (t_start + k * dt) for k in expected_ticks}

    procs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("processor")],
                   key=lambda p: int(re.sub(r"\D", "", p.name) or "0"))
    if not procs:
        raise SystemExit(f"No processor* dirs found under: {run_dir}")

    # Map: proc_name -> tick -> time_dir_name
    proc_tick_dir = {p.name: {} for p in procs}
    # Also record existing time dirs that do not snap to the grid (debug)
    offgrid = {p.name: [] for p in procs}

    for p in procs:
        for td in p.iterdir():
            if not is_numeric_dir(td):
                continue
            try:
                t = float(td.name)
            except Exception:
                continue
            if t < t_start - 10*dt or t > t_end + 10*dt:
                continue
            k, t_hat = snap_tick(t, t_start, dt, a.tol)
            if k is None:
                offgrid[p.name].append(td.name)
                continue
            if 0 <= k <= n_steps:
                # If duplicates map to same tick, keep the closest-to-grid one
                prev = proc_tick_dir[p.name].get(k, None)
                if prev is None:
                    proc_tick_dir[p.name][k] = td.name
                else:
                    prev_t = float(prev)
                    if abs(t - (t_start + k*dt)) < abs(prev_t - (t_start + k*dt)):
                        proc_tick_dir[p.name][k] = td.name

    # Missing time folders
    missing_by_time = defaultdict(list)  # tick -> [proc_name...]
    for k in expected_ticks:
        for pn in proc_tick_dir:
            if k not in proc_tick_dir[pn]:
                missing_by_time[k].append(pn)

    missing_global = [k for k, miss in missing_by_time.items() if len(miss) == len(proc_tick_dir)]
    missing_partial = {k: miss for k, miss in missing_by_time.items()
                       if 0 < len(miss) < len(proc_tick_dir)}

    # Missing field files inside existing time folders
    missing_fields = []  # list of dicts
    for pn, tick_map in proc_tick_dir.items():
        pdir = run_dir / pn
        for k, tdir_name in tick_map.items():
            tdir = pdir / tdir_name
            for fld in a.fields:
                fpath = tdir / fld
                if not fpath.exists():
                    missing_fields.append({
                        "processor": pn,
                        "tick": k,
                        "time": expected_times[k],
                        "time_dir": tdir_name,
                        "field": fld,
                        "path": str(fpath),
                    })

    report = {
        "run_dir": str(run_dir),
        "n_procs": len(proc_tick_dir),
        "processors": list(proc_tick_dir.keys()),
        "t_start": t_start,
        "t_end": t_end,
        "dt": dt,
        "n_expected_times": len(expected_ticks),
        "missing_global_times": [{"tick": k, "time": expected_times[k]} for k in missing_global],
        "missing_partial_times": [
            {"tick": k, "time": expected_times[k], "missing_processors": miss}
            for k, miss in sorted(missing_partial.items())
        ],
        "missing_field_files": missing_fields,
        "offgrid_time_dirs_sample": {pn: offgrid[pn][:20] for pn in offgrid if offgrid[pn]},
    }

    # Console summary (compact)
    print(f"[case] {run_dir}")
    print(f"[procs] {len(proc_tick_dir)}  [window] {t_start}:{t_end}  [dt] {dt}  [expected] {len(expected_ticks)}")
    print(f"[missing global] {len(missing_global)} time(s)")
    if missing_global:
        print("  first few:", ", ".join(f"{expected_times[k]:.6f}" for k in missing_global[:10]))
    print(f"[missing partial] {len(missing_partial)} time(s)")
    if missing_partial:
        k0 = sorted(missing_partial.keys())[0]
        print(f"  example time={expected_times[k0]:.6f} missing {len(missing_partial[k0])}/{len(proc_tick_dir)} procs")
    print(f"[missing fields] {len(missing_fields)} file(s) across all procs/times")
    if missing_fields:
        ex = missing_fields[0]
        print(f"  example: {ex['processor']}/{ex['time_dir']}/{ex['field']} is missing")

    if a.out:
        outp = Path(a.out).resolve()
    else:
        outp = run_dir / "integrity_report.json"
    outp.write_text(json.dumps(report, indent=2))
    print(f"[report] wrote {outp}")

if __name__ == "__main__":
    main()