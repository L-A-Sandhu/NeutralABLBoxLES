#!/usr/bin/env python3
# les_eta.py  (CLI + importable API)

from __future__ import annotations
import argparse, glob, json, time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

@dataclass
class ETAResult:
    run_dir: Path
    log_path: Path
    current_sim: float
    wall_per_sim: float
    t_spin: float
    end_time: float
    eta_spin_s: float
    eta_end_s: float

def _hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"

def _run_dirs(repo: Path) -> List[Path]:
    patt = str(repo / "runs" / "run_*_spin*_np*")
    return [Path(p) for p in glob.glob(patt)]

def _best_log_mtime(run_dir: Path) -> float:
    mt = 0.0
    for name in ("log.capture", "log.spinup"):
        p = run_dir / name
        if p.exists() and p.stat().st_size > 0:
            mt = max(mt, p.stat().st_mtime)
    # fallback: capture window json
    cw = run_dir / "capture_window.json"
    if cw.exists():
        mt = max(mt, cw.stat().st_mtime)
    # fallback: directory itself
    mt = max(mt, run_dir.stat().st_mtime)
    return mt

def _latest_run_dir(repo: Path) -> Path:
    cands = _run_dirs(repo)
    if not cands:
        raise FileNotFoundError(f"No run dirs under {repo/'runs'} matching run_*_spin*_np*")
    return max(cands, key=_best_log_mtime)

def _pick_log(run_dir: Path) -> Path:
    logs = []
    for name in ("log.capture", "log.spinup"):
        p = run_dir / name
        if p.exists() and p.stat().st_size > 0:
            logs.append(p)
    if not logs:
        raise FileNotFoundError(f"Missing/empty log in {run_dir} (log.spinup or log.capture)")
    return max(logs, key=lambda p: p.stat().st_mtime)

def _parse_time_clock(log_path: Path) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    t: Optional[float] = None
    with log_path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Time ="):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        t = float(parts[2])
                    except ValueError:
                        t = None
            if "ClockTime" in line:
                parts = line.replace("=", " = ").split()
                try:
                    i = parts.index("ClockTime")
                    ct = float(parts[i + 2])
                    if t is not None:
                        pairs.append((t, ct))
                except Exception:
                    continue
    return pairs

def estimate_eta(repo: str | Path = ".", run: str | Path | None = None, window: int = 200) -> ETAResult:
    repo = Path(repo).resolve()
    run_dir = Path(run).resolve() if run is not None else _latest_run_dir(repo)

    cap = run_dir / "capture_window.json"
    if not cap.exists():
        raise FileNotFoundError(f"Missing {cap}")
    j = json.loads(cap.read_text())
    t_spin = float(j["t_spin"])
    end_time = float(j["end_time"])

    log_path = _pick_log(run_dir)
    pairs = _parse_time_clock(log_path)
    if len(pairs) < 5:
        raise RuntimeError("Not enough ClockTime samples yet")

    w = min(max(2, int(window)), len(pairs))
    t0, c0 = pairs[-w]
    t1, c1 = pairs[-1]
    ds = t1 - t0
    dw = c1 - c0
    if ds <= 0 or dw <= 0:
        raise RuntimeError("Bad window (non-positive ds or dw)")

    wall_per_sim = dw / ds
    rem_spin = max(t_spin - t1, 0.0)
    rem_end  = max(end_time - t1, 0.0)

    return ETAResult(
        run_dir=run_dir,
        log_path=log_path,
        current_sim=t1,
        wall_per_sim=wall_per_sim,
        t_spin=t_spin,
        end_time=end_time,
        eta_spin_s=rem_spin * wall_per_sim,
        eta_end_s=rem_end  * wall_per_sim,
    )

def _print(res: ETAResult) -> None:
    print(f"run        = {res.run_dir}")
    print(f"log        = {res.log_path.name}")
    print(f"currentSim = {res.current_sim:g} sim-sec")
    print(f"rate(win)  = {res.wall_per_sim:g} wall-sec per sim-sec")
    print(f"ETA spinup = {_hms(res.eta_spin_s)}")
    print(f"ETA endTime= {_hms(res.eta_end_s)}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="repo root that contains ./runs/")
    ap.add_argument("--run", default=None, help="explicit run dir (overrides auto-pick)")
    ap.add_argument("--window", type=int, default=200, help="recent ClockTime samples")
    ap.add_argument("--follow", action="store_true", help="repeat periodically")
    ap.add_argument("--interval", type=float, default=15.0, help="seconds between updates")
    args = ap.parse_args()

    while True:
        try:
            res = estimate_eta(args.repo, args.run, args.window)
            _print(res)
        except Exception as e:
            print(f"[les-eta] {e}")
        if not args.follow:
            break
        time.sleep(max(1.0, args.interval))

if __name__ == "__main__":
    main()
