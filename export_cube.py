#!/usr/bin/env python3
# scripts/export_les_UP_onceplot.py
#
# Export LES fields from an OpenFOAM run folder:
#   - U.npy: (Nt, Nz, Ny, Nx, 3)
#   - P.npy: (Nt, Nz, Ny, Nx)
# and ONE 3D scatter plot using the TIME-MEAN speed field (all times aggregated).
#
# Coordinates:
#   --coords paper  => x'=x-400, y'=y-400, z'=z  (matches paper axes: x[-400,1400], y[-400,400], z[0,600])
#   --coords foam   => x in [0,Lx], y in [0,Ly]
#   --coords center => x in [-Lx/2,Lx/2], y in [-Ly/2,Ly/2]
#
# Requirements: numpy, matplotlib, and OpenFOAM env for postProcess (or generate C beforehand).

from __future__ import annotations
import argparse, json, re, shutil, subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def list_times(case_dir: Path):
    times = []
    name_of = {}
    for p in case_dir.iterdir():
        if p.is_dir():
            try:
                t = float(p.name)
                times.append(t); name_of[t] = p.name
            except Exception:
                pass
    times.sort()
    return times, name_of

def _is_binary_format(txt: str) -> bool:
    return bool(re.search(r"\bformat\s+binary\s*;", txt))

def read_internal_field(path: Path, ncomp: int, expected_N: int | None = None) -> np.ndarray:
    txt = path.read_text(errors="ignore")
    if _is_binary_format(txt):
        raise RuntimeError(
            f"{path}: format binary detected.\n"
            f"Fix: set writeFormat ascii in system/controlDict and rerun (or convert fields to ascii)."
        )

    # uniform
    mu = re.search(r"internalField\s+uniform\s+([^;]+);", txt)
    if mu:
        if expected_N is None:
            raise RuntimeError(f"{path}: uniform internalField but expected_N not provided.")
        val = mu.group(1).strip().replace("(", " ").replace(")", " ").replace(",", " ")
        nums = np.fromstring(val, sep=" ", dtype=np.float32)
        if ncomp == 1:
            if nums.size != 1:
                raise RuntimeError(f"{path}: expected 1 scalar, got {nums.size}")
            return np.full((expected_N,), float(nums[0]), dtype=np.float32)
        if nums.size != ncomp:
            raise RuntimeError(f"{path}: expected {ncomp} comps, got {nums.size}")
        return np.tile(nums.reshape(1, ncomp), (expected_N, 1)).astype(np.float32)

    # nonuniform List<...> N ( ... ) ;
    m = re.search(r"internalField\s+nonuniform\s+List<\w+>\s+(\d+)\s*\(", txt, flags=re.S)
    if not m:
        raise RuntimeError(f"{path}: expected ASCII internalField (uniform or nonuniform).")
    N = int(m.group(1))
    start = m.end()
    m2 = re.search(r"\)\s*;", txt[start:], flags=re.S)
    if not m2:
        raise RuntimeError(f"{path}: could not find end of internalField list (')' then ';').")
    end = start + m2.start()

    raw = txt[start:end].replace("(", " ").replace(")", " ").replace(",", " ")
    arr = np.fromstring(raw, sep=" ", dtype=np.float32)
    if arr.size == 0:
        arr = np.asarray(_num_re.findall(raw), dtype=np.float32)

    if ncomp == 1:
        if arr.size != N:
            raise RuntimeError(f"{path}: expected {N}, got {arr.size}")
        return arr
    if arr.size != N * ncomp:
        raise RuntimeError(f"{path}: expected {N*ncomp}, got {arr.size}")
    return arr.reshape(N, ncomp)

def build_perm(C, Lx, Ly, Lz, Nx, Ny, Nz):
    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
    i = np.rint(C[:, 0] / dx - 0.5).astype(np.int64)
    j = np.rint(C[:, 1] / dy - 0.5).astype(np.int64)
    k = np.rint(C[:, 2] / dz - 0.5).astype(np.int64)
    lin = (k * Ny + j) * Nx + i
    p = np.argsort(lin)
    return p, dx, dy, dz

def ensure_cell_centres(run_dir: Path, t_name: str):
    C_path = run_dir / t_name / "C"
    if C_path.exists():
        return
    if shutil.which("postProcess") is None:
        raise RuntimeError(
            "postProcess not found in PATH.\n"
            "Run this inside the OpenFOAM shell/env (the same one where blockMesh works),\n"
            "or generate C once manually:\n"
            f"  cd {run_dir}\n"
            f"  postProcess -func writeCellCentres -time {t_name}\n"
        )
    cmd = ["postProcess", "-func", "writeCellCentres", "-time", t_name]
    r = subprocess.run(cmd, cwd=str(run_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"postProcess failed:\n{r.stdout}\nMissing: {C_path}")

def select_times(run_dir: Path, times, name_of, t0_rel, t1_rel, dt, tol=1e-6):
    # If capture_window.json exists, interpret t0/t1 as RELATIVE to t_spin (like your --t0 0 --t1 10 usage)
    base = 0.0
    cw = run_dir / "capture_window.json"
    if cw.exists():
        j = json.loads(cw.read_text())
        base = float(j.get("t_spin", 0.0))

    if t0_rel is None and t1_rel is None:
        # default: use all times >= base (capture)
        return [t for t in times if t >= base], base

    if t0_rel is None: t0_rel = 0.0
    if t1_rel is None:
        t1_rel = max(0.0, (max(times) - base))

    abs0, abs1 = base + float(t0_rel), base + float(t1_rel)
    out = []
    for t in times:
        if t < abs0 - tol or t > abs1 + tol:
            continue
        n = round((t - abs0) / float(dt))
        tgrid = abs0 + n * float(dt)
        if abs(t - tgrid) <= 1e-3:  # robust for float folder names
            out.append(t)
    out.sort()
    return out, base

def coord_offsets(coords: str, Lx, Ly, dx, dy):
    if coords == "paper":   # x-=400, y-=400
        return (dx/2 - 400.0, dy/2 - 400.0)
    if coords == "center":  # centered domain
        return (-Lx/2 + dx/2, -Ly/2 + dy/2)
    return (dx/2, dy/2)     # foam native

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="path to runs/run_* folder")
    ap.add_argument("--Nx", type=int, default=180)
    ap.add_argument("--Ny", type=int, default=80)
    ap.add_argument("--Nz", type=int, default=80)
    ap.add_argument("--Lx", type=float, default=1800.0)
    ap.add_argument("--Ly", type=float, default=800.0)
    ap.add_argument("--Lz", type=float, default=600.0)
    ap.add_argument("--U_name", default="U")
    ap.add_argument("--p_name", default="p")
    ap.add_argument("--dt", type=float, default=0.5, help="sampling interval (s) for export (paper uses 0.5)")
    ap.add_argument("--t0", type=float, default=None, help="start time RELATIVE to capture start (default: capture start)")
    ap.add_argument("--t1", type=float, default=None, help="end time RELATIVE to capture start (default: capture end)")
    ap.add_argument("--coords", choices=["paper", "foam", "center"], default="paper")
    ap.add_argument("--plot_stride", type=int, default=1, help="1 = plot all grid points (can be heavy)")
    args = ap.parse_args()

    run_dir = Path(args.run).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")

    times, name_of = list_times(run_dir)
    if not times:
        raise SystemExit("No numeric time directories found in run folder.")

    times_use, base = select_times(run_dir, times, name_of, args.t0, args.t1, args.dt)
    if not times_use:
        raise SystemExit("No times selected. Check --t0/--t1/--dt and what time folders exist.")

    Nt = len(times_use)
    Ncells = args.Nx * args.Ny * args.Nz

    # Cell centres from first selected time
    t0_name = name_of[times_use[0]]
    ensure_cell_centres(run_dir, t0_name)
    C = read_internal_field(run_dir / t0_name / "C", 3, expected_N=Ncells).astype(np.float64)
    perm, dx, dy, dz = build_perm(C, args.Lx, args.Ly, args.Lz, args.Nx, args.Ny, args.Nz)

    # Outputs in run_dir
    U_out = run_dir / f"U_{args.t0 or 0:g}_{args.t1 or (times_use[-1]-base):g}_dt{args.dt:g}.npy"
    P_out = run_dir / f"P_{args.t0 or 0:g}_{args.t1 or (times_use[-1]-base):g}_dt{args.dt:g}.npy"
    M_out = run_dir / f"meta_UP_dt{args.dt:g}.json"
    plot_out = run_dir / f"speed_3d_timeMean_stride{args.plot_stride}.png"

    U_mm = np.lib.format.open_memmap(U_out, mode="w+", dtype=np.float32,
                                     shape=(Nt, args.Nz, args.Ny, args.Nx, 3))
    P_mm = np.lib.format.open_memmap(P_out, mode="w+", dtype=np.float32,
                                     shape=(Nt, args.Nz, args.Ny, args.Nx))

    # streaming mean of speed over time (all times aggregated)
    mean_speed = np.zeros((args.Nz, args.Ny, args.Nx), dtype=np.float64)

    for it, t in enumerate(times_use):
        t_name = name_of[t]
        U_path = run_dir / t_name / args.U_name
        p_path = run_dir / t_name / args.p_name
        if not U_path.exists() or not p_path.exists():
            raise SystemExit(f"Missing fields at {t_name}: {U_path.name} or {p_path.name}")

        U = read_internal_field(U_path, 3, expected_N=Ncells)
        P = read_internal_field(p_path, 1, expected_N=Ncells)

        Ugrid = U[perm].reshape(args.Nz, args.Ny, args.Nx, 3)
        Pgrid = P[perm].reshape(args.Nz, args.Ny, args.Nx)

        U_mm[it] = Ugrid
        P_mm[it] = Pgrid

        mean_speed += np.linalg.norm(Ugrid, axis=3)

        if (it % max(1, Nt//10)) == 0 or it == Nt - 1:
            print(f"[io] {it+1}/{Nt} wrote t={t_name}")

    mean_speed /= float(Nt)

    # ONE plot: time-mean speed over all times (all times aggregated)
    s = max(1, int(args.plot_stride))
    kk = np.arange(0, args.Nz, s)
    jj = np.arange(0, args.Ny, s)
    ii = np.arange(0, args.Nx, s)

    ox, oy = coord_offsets(args.coords, args.Lx, args.Ly, dx, dy)
    oz = dz/2

    x = ox + ii * dx
    y = oy + jj * dy
    z = oz + kk * dz
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    sp = mean_speed[kk][:, jj][:, :, ii].ravel()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), c=sp, s=1, alpha=0.9)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="time-mean |U| (m/s)")
    plt.tight_layout()
    plt.savefig(plot_out, dpi=200)
    plt.close(fig)

    M_out.write_text(json.dumps({
        "run_dir": str(run_dir),
        "base_time_capture_start_s": base,
        "times_abs_s": [float(t) for t in times_use],
        "times_rel_s": [float(t - base) for t in times_use],
        "domain_m": {"Lx": args.Lx, "Ly": args.Ly, "Lz": args.Lz},
        "grid": {"Nx": args.Nx, "Ny": args.Ny, "Nz": args.Nz, "dx": dx, "dy": dy, "dz": dz},
        "coords": args.coords,
        "files": {"U": U_out.name, "P": P_out.name, "plot_timeMean_speed": plot_out.name},
        "ordering": {"U": "U[t,k,j,i,comp]", "P": "P[t,k,j,i]"},
        "note": "Plot aggregates all timestamps via time-mean speed."
    }, indent=2))

    print(f"[done] wrote {U_out.name}, {P_out.name}, {M_out.name}")
    print(f"[plot] wrote {plot_out.name} (plot_stride={args.plot_stride})")

if __name__ == "__main__":
    main()
