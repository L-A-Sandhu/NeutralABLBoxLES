#!/usr/bin/env python3
import argparse, os, re
from pathlib import Path

NUM = re.compile(r"^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--t-start", type=float, required=True)
    ap.add_argument("--t-end", type=float, required=True)
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--fields", nargs="+", default=["U","p"])
    ap.add_argument("--rename", action="store_true", help="rename bad times and all >= first_bad to *.BAD")
    args = ap.parse_args()

    run = Path(args.run).resolve()
    procs = sorted([p for p in run.iterdir() if p.is_dir() and p.name.startswith("processor")])
    if not procs: raise SystemExit(f"no processor* in {run}")

    # map each proc: numeric_time -> dirname
    proc_map = {}
    for p in procs:
        m = {}
        for d in p.iterdir():
            if d.is_dir() and NUM.match(d.name):
                m[float(d.name)] = d.name
        proc_map[p.name] = m

    nsteps = int(round((args.t_end - args.t_start)/args.dt))
    exp = [args.t_start + k*args.dt for k in range(nsteps+1)]

    def status(t):
        missing = 0
        badfield = 0
        for p in procs:
            pm = proc_map[p.name]
            if t not in pm:
                missing += 1
                continue
            td = p / pm[t]
            for f in args.fields:
                if not (td / f).is_file():
                    badfield += 1
                    break
        if missing == len(procs): return "MISSING_ALL"
        if missing > 0: return "MISSING_SOME"
        if badfield > 0: return "INCOMPLETE"
        return "OK"

    first_bad = None
    last_good = None
    bad_times = []
    for t in exp:
        s = status(t)
        if s == "OK":
            if first_bad is None: last_good = t
        else:
            bad_times.append((t,s))
            if first_bad is None: first_bad = t

    print(f"[run] {run}")
    print(f"[expected] {len(exp)}  [bad] {len(bad_times)}")
    print(f"[first_bad] {first_bad}")
    print(f"[last_good]  {last_good}")

    if args.rename and first_bad is not None:
        # quarantine ALL times >= first_bad (even if some exist), to keep a consistent suffix segment
        for p in procs:
            pm = proc_map[p.name]
            for t in [x for x in pm.keys() if x >= first_bad]:
                src = p / pm[t]
                dst = p / (pm[t] + ".BAD")
                if src.exists() and not dst.exists():
                    src.rename(dst)
        print("[rename] done")

if __name__ == "__main__":
    main()