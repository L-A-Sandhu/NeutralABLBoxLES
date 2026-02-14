#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_cube.py (max-speed, parallel + consolidate)

What it does
- Ensures reconstructed time folders exist (optionally runs reconstructPar -fields '(U p)').
- Selects the capture window (supports the same t0/t1 anchoring logic).
- Exports U and p into .npy using PROCESS-level parallelism (multiple workers).
- Each worker writes a chunk file, then the main process consolidates into:
    <out_prefix>_U.npy, <out_prefix>_p.npy, <out_prefix>.meta.json

Assumptions
- OpenFOAM fields are written in ASCII (binary will error).
- Reconstructed folders exist OR processor*/ exist (for reconstructPar).
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from numpy.lib.format import open_memmap
from concurrent.futures import ProcessPoolExecutor, as_completed

NUMERIC_DIR_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$")
_INTERNAL_HDR_RE_B = re.compile(
    br"internalField\s+nonuniform\s+List<[^>]+>\s*([0-9]+)\s*\(",
    flags=re.M,
)
_LIST_END_RE_B = re.compile(br"^\s*\)\s*;\s*$", flags=re.M)


def _run(cmd: List[str], cwd: Path) -> None:
    p = subprocess.run(
        cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")


def _list_numeric_dirs(parent: Path) -> List[float]:
    out: List[float] = []
    if not parent.exists():
        return out
    for p in parent.iterdir():
        if p.is_dir() and NUMERIC_DIR_RE.match(p.name):
            try:
                out.append(float(p.name))
            except ValueError:
                pass
    return sorted(out)


def _processor_dirs(run_dir: Path) -> List[Path]:
    return sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("processor")])


def _read_capture_window(run_dir: Path) -> Optional[Tuple[float, float]]:
    jpath = run_dir / "capture_window.json"
    if not jpath.exists():
        return None
    try:
        j = json.loads(jpath.read_text())
        return float(j["t_spin"]), float(j["end_time"])
    except Exception:
        return None


def _probe_internal_field_kind(path: Path) -> Tuple[str, str]:
    """
    Returns (format, kind):
      format in {"ascii","binary","unknown"}
      kind   in {"uniform","nonuniform","missing"}
    Fast probe reads only header chunk.
    """
    if not path.exists():
        return "unknown", "missing"
    try:
        head = path.read_bytes()[:256 * 1024]
        h = head.decode("latin1", errors="ignore")

        fmt = "unknown"
        if "format" in h:
            if "binary" in h:
                fmt = "binary"
            elif "ascii" in h:
                fmt = "ascii"

        kind = "missing"
        m = re.search(r"internalField\s+([^;]+);", h)
        if m:
            s = m.group(1)
            if "nonuniform" in s:
                kind = "nonuniform"
            elif "uniform" in s:
                kind = "uniform"
        else:
            if "internalField" in h and "nonuniform" in h:
                kind = "nonuniform"
            elif "internalField" in h and "uniform" in h:
                kind = "uniform"

        return fmt, kind
    except Exception:
        return "unknown", "missing"


def _ensure_reconstructed(run_dir: Path, tmin: float, tmax: float) -> None:
    # If any reconstructed nonuniform U/p exist, skip.
    for t in _list_numeric_dirs(run_dir):
        up = run_dir / f"{t:g}" / "U"
        pp = run_dir / f"{t:g}" / "p"
        _, uk = _probe_internal_field_kind(up)
        _, pk = _probe_internal_field_kind(pp)
        if uk == "nonuniform" and pk == "nonuniform":
            return

    if not _processor_dirs(run_dir):
        return

    tr = f"{tmin}:{tmax}"
    print(f"[export] Reconstructing with: reconstructPar -time '{tr}' -fields '(U p)' ...")
    _run(["reconstructPar", "-time", tr, "-fields", "(U p)"], cwd=run_dir)


def _read_internal_field_ascii_fast(path: Path, kind: str, dtype: np.dtype) -> np.ndarray:
    """
    Fast parse OpenFOAM ASCII internalField using one np.fromstring on the list block.
    kind: "scalar" or "vector"
    """
    fmt, k = _probe_internal_field_kind(path)
    if k != "nonuniform":
        raise RuntimeError(f"{path} internalField is not nonuniform (found {k}).")
    if fmt == "binary":
        raise RuntimeError(f"{path} is binary. Set writeFormat ascii and regenerate capture outputs.")

    b = path.read_bytes()
    m = _INTERNAL_HDR_RE_B.search(b)
    if not m:
        raise RuntimeError(f"Could not find nonuniform internalField header in {path}")
    n = int(m.group(1))
    start = m.end()
    tail = b[start:]
    m_end = _LIST_END_RE_B.search(tail)
    if not m_end:
        raise RuntimeError(f"Could not find end of internalField list in {path}")

    data = tail[:m_end.start()]
    data = data.replace(b"(", b" ").replace(b")", b" ")
    # np.fromstring can take bytes in many builds, but decoding is safest cross-version.
    s = data.decode("latin1", errors="ignore")
    arr = np.fromstring(s, sep=" ", dtype=dtype)

    if kind == "scalar":
        if arr.size != n:
            raise RuntimeError(f"{path}: parsed {arr.size} scalars but header says {n}")
        return arr
    else:
        if arr.size != 3 * n:
            raise RuntimeError(f"{path}: parsed {arr.size} floats but expected {3*n} for {n} vectors")
        return arr.reshape((n, 3))


def _available_nonuniform_times(run_dir: Path, U_name: str, p_name: str) -> List[float]:
    times = _list_numeric_dirs(run_dir)
    good: List[float] = []
    for t in times:
        up = run_dir / f"{t:g}" / U_name
        pp = run_dir / f"{t:g}" / p_name
        _, uk = _probe_internal_field_kind(up)
        _, pk = _probe_internal_field_kind(pp)
        if uk == "nonuniform" and pk == "nonuniform":
            good.append(t)
    return sorted(good)


def _select_times(avail: List[float], t0: float, t1: float, dt: float) -> List[float]:
    if not avail:
        return []

    anchor = avail[0]
    if abs(t0) < 1e-12:
        abs_t0 = anchor + t0
        abs_t1 = anchor + t1
        print(f"[export] Anchoring t0/t1 at first nonuniform time {anchor:.6f}: range=[{abs_t0:.6f},{abs_t1:.6f}]")
    else:
        abs_t0, abs_t1 = t0, t1

    if abs_t1 < abs_t0:
        abs_t0, abs_t1 = abs_t1, abs_t0

    start_candidates = [t for t in avail if t >= abs_t0 - 1e-9]
    if not start_candidates:
        return []
    t_start = start_candidates[0]

    # Target grid
    nsteps = int(math.floor((abs_t1 - t_start) / dt + 1e-9)) + 1
    targets = [t_start + k * dt for k in range(max(0, nsteps))]

    # Snap to nearest available within tolerance
    tol = max(1e-6, 0.05 * dt)
    chosen: List[float] = []
    ai = 0
    for tt in targets:
        while ai + 1 < len(avail) and avail[ai + 1] <= tt:
            ai += 1
        best = avail[ai]
        if ai + 1 < len(avail) and abs(avail[ai + 1] - tt) < abs(best - tt):
            best = avail[ai + 1]
        if abs(best - tt) <= tol:
            chosen.append(best)

    # Preserve order, drop duplicates
    out: List[float] = []
    seen = set()
    for t in chosen:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _infer_dims(ncells: int, Nx: int, Ny: int, Nz: int) -> Optional[Tuple[int, int, int]]:
    if Nx > 0 and Ny > 0 and Nz > 0 and Nx * Ny * Nz == ncells:
        return (Nx, Ny, Nz)
    return None


@dataclass
class ChunkResult:
    chunk_id: int
    start_index: int
    nt: int
    U_path: str
    p_path: str
    times_path: str


def _worker_export_chunk(
    run_dir_str: str,
    times: List[float],
    start_index: int,
    chunk_id: int,
    tmp_dir_str: str,
    U_name: str,
    p_name: str,
    dtype_str: str,
    reshape: bool,
    Nz: int,
    Ny: int,
    Nx: int,
    ncells: int,
) -> ChunkResult:
    run_dir = Path(run_dir_str)
    tmp_dir = Path(tmp_dir_str)
    dtype = np.float32 if dtype_str == "float32" else np.float64

    # per-chunk outputs
    U_path = tmp_dir / f"part_{chunk_id:04d}_U.npy"
    p_path = tmp_dir / f"part_{chunk_id:04d}_p.npy"
    times_path = tmp_dir / f"part_{chunk_id:04d}_times.json"

    nt = len(times)
    if reshape:
        U_shape = (nt, Nz, Ny, Nx, 3)
        p_shape = (nt, Nz, Ny, Nx)
    else:
        U_shape = (nt, ncells, 3)
        p_shape = (nt, ncells)

    U_mm = open_memmap(U_path, mode="w+", dtype=dtype, shape=U_shape)
    p_mm = open_memmap(p_path, mode="w+", dtype=dtype, shape=p_shape)

    for i, t in enumerate(times):
        U = _read_internal_field_ascii_fast(run_dir / f"{t:g}" / U_name, "vector", dtype)
        p = _read_internal_field_ascii_fast(run_dir / f"{t:g}" / p_name, "scalar", dtype)

        if U.shape[0] != ncells or p.shape[0] != ncells:
            raise RuntimeError(f"[chunk {chunk_id}] Cell-count mismatch at time {t:g}")

        if reshape:
            U_mm[i, ...] = U.reshape((Nz, Ny, Nx, 3), order="C")
            p_mm[i, ...] = p.reshape((Nz, Ny, Nx), order="C")
        else:
            U_mm[i, ...] = U
            p_mm[i, ...] = p

    del U_mm
    del p_mm

    times_path.write_text(json.dumps(times, indent=2))
    return ChunkResult(
        chunk_id=chunk_id,
        start_index=start_index,
        nt=nt,
        U_path=str(U_path),
        p_path=str(p_path),
        times_path=str(times_path),
    )


def _split_into_chunks(sel: List[float], procs: int, chunk_size: int) -> List[Tuple[int, List[float]]]:
    """
    Returns list of (start_index, times_chunk)
    If chunk_size <= 0, auto-split into ~procs chunks of similar size.
    """
    T = len(sel)
    if T == 0:
        return []
    if chunk_size <= 0:
        procs = max(1, min(procs, T))
        chunk_size = int(math.ceil(T / procs))
    chunks = []
    for s in range(0, T, chunk_size):
        chunks.append((s, sel[s:s + chunk_size]))
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--Nx", type=int, default=180)
    ap.add_argument("--Ny", type=int, default=80)
    ap.add_argument("--Nz", type=int, default=80)
    ap.add_argument("--U_name", default="U")
    ap.add_argument("--p_name", default="p")
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=200.0)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--out-prefix", dest="out_prefix", default="U_all")
    ap.add_argument("--no-reconstruct", action="store_true")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 1)),
                    help="Number of worker PROCESSES for export (higher = faster until disk saturates).")
    ap.add_argument("--chunk-size", type=int, default=0,
                    help="Times per chunk. 0 = auto.")
    ap.add_argument("--keep-tmp", action="store_true", help="Keep temporary part files after consolidation.")
    ap.add_argument("--tmp-dir", default="", help="Optional temp dir. Default: <run>/.export_tmp_<prefix>_<ts>")
    args = ap.parse_args()

    run_dir = Path(args.run).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    # 1) reconstruct if needed
    cw = _read_capture_window(run_dir)
    if not args.no_reconstruct:
        if cw is not None:
            _ensure_reconstructed(run_dir, cw[0], cw[1])
        else:
            procs_dirs = _processor_dirs(run_dir)
            if procs_dirs:
                pt = _list_numeric_dirs(procs_dirs[0])
                if pt:
                    _ensure_reconstructed(run_dir, pt[0], pt[-1])

    # 2) select times
    avail = _available_nonuniform_times(run_dir, args.U_name, args.p_name)
    if not avail:
        raise SystemExit("No reconstructed nonuniform time folders found (need ASCII U and p).")

    sel = _select_times(avail, args.t0, args.t1, args.dt)
    if not sel:
        raise SystemExit("No times selected. Check --t0/--t1/--dt and available time folders.")

    print(f"[export] Selected {len(sel)} times: {sel[0]:.6f} ... {sel[-1]:.6f}")

    # 3) infer shapes by reading first time
    dtype = np.float32 if args.dtype == "float32" else np.float64
    U0 = _read_internal_field_ascii_fast(run_dir / f"{sel[0]:g}" / args.U_name, "vector", dtype)
    p0 = _read_internal_field_ascii_fast(run_dir / f"{sel[0]:g}" / args.p_name, "scalar", dtype)
    ncells = int(U0.shape[0])
    if p0.shape[0] != ncells:
        raise RuntimeError(f"Cell-count mismatch at first time: U={ncells} p={p0.shape[0]}")

    dims = _infer_dims(ncells, args.Nx, args.Ny, args.Nz)
    reshape = dims is not None
    if reshape:
        Nx, Ny, Nz = dims
        U_shape = (len(sel), Nz, Ny, Nx, 3)
        p_shape = (len(sel), Nz, Ny, Nx)
    else:
        Nx, Ny, Nz = args.Nx, args.Ny, args.Nz
        U_shape = (len(sel), ncells, 3)
        p_shape = (len(sel), ncells)

    # 4) setup tmp dir for part outputs
    ts = time.strftime("%Y%m%d_%H%M%S")
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else (run_dir / f".export_tmp_{args.out_prefix}_{ts}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 5) parallel export into chunk files
    chunks = _split_into_chunks(sel, args.procs, args.chunk_size)
    print(f"[export] Parallel chunks: {len(chunks)}  |  workers={min(args.procs, len(chunks))}")
    t_wall0 = time.time()

    # write first time via chunking too (keeps logic consistent)
    results: List[ChunkResult] = []
    with ProcessPoolExecutor(max_workers=min(args.procs, len(chunks))) as ex:
        futs = []
        for chunk_id, (start_idx, times_chunk) in enumerate(chunks):
            futs.append(ex.submit(
                _worker_export_chunk,
                str(run_dir),
                times_chunk,
                start_idx,
                chunk_id,
                str(tmp_dir),
                args.U_name,
                args.p_name,
                args.dtype,
                reshape,
                Nz, Ny, Nx,
                ncells,
            ))
        for fut in as_completed(futs):
            results.append(fut.result())

    results.sort(key=lambda r: r.start_index)

    # 6) consolidate into final memmaps (single-writer, sequential, RAM-safe)
    out_u = run_dir / f"{args.out_prefix}_U.npy"
    out_p = run_dir / f"{args.out_prefix}_p.npy"
    out_meta = run_dir / f"{args.out_prefix}.meta.json"

    print(f"[export] Consolidating -> {out_u.name}, {out_p.name}")
    U_mm = open_memmap(out_u, mode="w+", dtype=dtype, shape=U_shape)
    p_mm = open_memmap(out_p, mode="w+", dtype=dtype, shape=p_shape)

    # Copy parts in order
    written = 0
    for r in results:
        U_part = np.load(r.U_path, mmap_mode="r")
        p_part = np.load(r.p_path, mmap_mode="r")
        s = r.start_index
        e = s + r.nt
        U_mm[s:e, ...] = U_part
        p_mm[s:e, ...] = p_part
        written = e
        if written % 50 == 0 or written == len(sel):
            dt_wall = time.time() - t_wall0
            rate = written / max(dt_wall, 1e-9)
            print(f"[export] {written}/{len(sel)} consolidated ({rate:.2f} steps/s)")

    del U_mm
    del p_mm

    # 7) metadata
    meta: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "times": sel,
        "dt_target": args.dt,
        "t0_arg": args.t0,
        "t1_arg": args.t1,
        "fields": {"U": args.U_name, "p": args.p_name},
        "dtype": args.dtype,
        "ncells": ncells,
        "dims": {"Nx": Nx, "Ny": Ny, "Nz": Nz} if reshape else None,
        "shapes": {"U": list(U_shape), "p": list(p_shape)},
        "capture_window": {"t_spin": cw[0], "end_time": cw[1]} if cw is not None else None,
        "parallel": {
            "procs": int(args.procs),
            "chunk_size": int(args.chunk_size),
            "num_chunks": len(chunks),
            "tmp_dir": str(tmp_dir),
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    t_wall = time.time() - t_wall0
    print(f"[export] DONE  wall_time={t_wall:.2f}s  avg={(len(sel)/max(t_wall,1e-9)):.2f} steps/s")

    # 8) cleanup
    if not args.keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[export] tmp cleaned: {tmp_dir}")
    else:
        print(f"[export] tmp kept: {tmp_dir}")


if __name__ == "__main__":
    main()