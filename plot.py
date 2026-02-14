# #!/usr/bin/env python3
# # make_zstack_gif.py
# #
# # Creates a 6x10 grid (60 subplots) GIF where each subplot is one z-plane.
# # Default field: |U|. Full X-Y extent (no crop). Saves into --run directory.

# from __future__ import annotations
# import argparse, json
# from pathlib import Path
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, PillowWriter

# def load_meta(meta_path: Path):
#     if meta_path.exists():
#         return json.loads(meta_path.read_text())
#     return {}

# def z_centers(Nz: int, Lz: float) -> np.ndarray:
#     dz = Lz / Nz
#     return (np.arange(Nz) + 0.5) * dz

# def compute_field(U_t_zi: np.ndarray, field: str) -> np.ndarray:
#     # U_t_zi: (Ny, Nx, 3)
#     if field == "Umag":
#         return np.sqrt(np.sum(U_t_zi * U_t_zi, axis=-1))
#     comp = {"Ux": 0, "Uy": 1, "Uz": 2}[field]
#     return U_t_zi[..., comp]

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--run", required=True, help="folder containing U_all_U.npy and (optionally) U_all.meta.json")
#     ap.add_argument("--U", default="U_all_U.npy", help="velocity .npy file name")
#     ap.add_argument("--meta", default="U_all.meta.json", help="metadata json file name")
#     ap.add_argument("--out", default=None, help="output GIF name (saved inside --run). default: zstack_60.gif")

#     ap.add_argument("--field", choices=["Umag","Ux","Uy","Uz"], default="Umag",
#                     help="what to visualize per plane")
#     ap.add_argument("--normalize", choices=["none","mean_xy"], default="none",
#                     help="none: raw field; mean_xy: divide each plane by its (x,y) mean at that time and z")

#     ap.add_argument("--nslices", type=int, default=60, help="number of z-planes to plot (default 60)")
#     ap.add_argument("--rows", type=int, default=6, help="subplot rows (default 6)")
#     ap.add_argument("--cols", type=int, default=10, help="subplot cols (default 10)")

#     ap.add_argument("--t0", type=int, default=0, help="start time index (0-based)")
#     ap.add_argument("--t1", type=int, default=-1, help="end time index inclusive (default: last)")
#     ap.add_argument("--stride", type=int, default=1, help="use every stride-th frame")

#     ap.add_argument("--fps", type=float, default=None, help="GIF fps (default: infer from meta dt and stride)")
#     ap.add_argument("--clim_pct", type=float, nargs=2, default=[1.0, 99.0],
#                     help="color limits from percentiles computed on the first rendered frame (default 1 99)")
#     ap.add_argument("--center_axes", action="store_true",
#                     help="plot x,y as centered coordinates [-Lx/2,Lx/2], [-Ly/2,Ly/2]")

#     args = ap.parse_args()
#     run = Path(args.run).resolve()
#     U_path = run / args.U
#     meta = load_meta(run / args.meta)

#     if not U_path.exists():
#         raise FileNotFoundError(f"Missing {U_path}")

#     U = np.load(U_path, mmap_mode="r")  # (T, Nz, Ny, Nx, 3)
#     T, Nz, Ny, Nx, _ = U.shape

#     # Domain lengths (fall back to your case defaults if meta missing)
#     Lx = float(meta.get("Lx", 1800.0))
#     Ly = float(meta.get("Ly", 800.0))
#     Lz = float(meta.get("Lz", 600.0))
#     dt = float(meta.get("dt", 0.5))

#     # Choose 60 z-indices approximately uniformly across [0, Nz-1]
#     ns = min(args.nslices, Nz)
#     z_idx = np.linspace(0, Nz - 1, ns).round().astype(int)
#     # ensure unique and keep count by re-linspacing if duplicates appear
#     z_idx = np.unique(z_idx)
#     if len(z_idx) != ns:
#         z_idx = np.linspace(0, Nz - 1, ns, endpoint=True).astype(int)
#         z_idx = np.unique(z_idx)
#     # If still short (only possible when Nz < ns), just use all planes
#     if len(z_idx) < ns:
#         z_idx = np.arange(Nz)

#     zc = z_centers(Nz, Lz)[z_idx]

#     # Time indices
#     t0 = max(0, args.t0)
#     t1 = (T - 1) if args.t1 < 0 else min(T - 1, args.t1)
#     frames = list(range(t0, t1 + 1, max(1, args.stride)))
#     if len(frames) == 0:
#         raise RuntimeError("No frames selected. Check --t0/--t1/--stride against U.shape[0].")

#     # Extent (full domain, no crop)
#     if args.center_axes:
#         extent = [-Lx/2, Lx/2, -Ly/2, Ly/2]
#         xlab, ylab = "x (m, centered)", "y (m, centered)"
#     else:
#         extent = [0.0, Lx, 0.0, Ly]
#         xlab, ylab = "x (m)", "y (m)"

#     # Figure grid
#     R, C = args.rows, args.cols
#     if R * C < len(z_idx):
#         raise ValueError(f"rows*cols={R*C} but need at least {len(z_idx)} subplots for nslices={len(z_idx)}")

#     fig, axs = plt.subplots(R, C, figsize=(2.2*C, 2.0*R), constrained_layout=True)
#     axs = np.array(axs).reshape(R, C)

#     # Prepare first frame to set consistent color limits
#     t_first = frames[0]
#     first_stack = []
#     for zi in z_idx:
#         fld = compute_field(U[t_first, zi, :, :, :], args.field).astype(np.float32)
#         if args.normalize == "mean_xy":
#             m = float(np.mean(fld))
#             if m != 0.0:
#                 fld = fld / m
#         first_stack.append(fld)
#     first_concat = np.concatenate([a.ravel() for a in first_stack])
#     p_lo, p_hi = np.percentile(first_concat, args.clim_pct)
#     vmin, vmax = float(p_lo), float(p_hi)

#     # Create image artists
#     ims = []
#     for k in range(R*C):
#         r, c = divmod(k, C)
#         ax = axs[r, c]
#         ax.set_xticks([])
#         ax.set_yticks([])
#         if k < len(z_idx):
#             im = ax.imshow(first_stack[k], origin="lower", extent=extent,
#                            vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
#             ax.set_title(f"z = {zc[k]:.1f} m", fontsize=9)
#             ims.append(im)
#         else:
#             ax.axis("off")

#     # One shared colorbar
#     cbar = fig.colorbar(ims[0], ax=axs, shrink=0.75, pad=0.01)
#     cbar.set_label(f"{args.field}" + ("/⟨·⟩xy" if args.normalize == "mean_xy" else ""), rotation=90)

#     # Labels only once (lightweight)
#     axs[-1, 0].set_xlabel(xlab)
#     axs[-1, 0].set_ylabel(ylab)

#     # Animation update
#     def update(frame_ti: int):
#         t = frame_ti
#         for j, zi in enumerate(z_idx):
#             fld = compute_field(U[t, zi, :, :, :], args.field).astype(np.float32)
#             if args.normalize == "mean_xy":
#                 m = float(np.mean(fld))
#                 if m != 0.0:
#                     fld = fld / m
#             ims[j].set_data(fld)
#         fig.suptitle(f"t index = {t}    t = {t*dt:.2f} s", fontsize=12)
#         return ims

#     fps = args.fps if args.fps is not None else (1.0 / (dt * max(1, args.stride)))
#     out_name = args.out or "zstack_60.gif"
#     out_path = run / out_name

#     ani = FuncAnimation(fig, update, frames=frames, blit=False, interval=1000.0/fps)
#     ani.save(out_path, writer=PillowWriter(fps=fps))
#     print(f"[saved] {out_path}")
#     print(f"[info] U shape={U.shape}  field={args.field}  frames={len(frames)}  z_slices={len(z_idx)}")

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
# make_zstack_gif.py
#
# Creates a 2x3 GIF where each subplot is one z-plane chosen at ~100 m spacing.
# Default: z centers at (z_step/2) + k*z_step, i.e. 50,150,250,350,450,550 m for Lz=600.
# Field: |U| by default. Full X-Y extent (no crop). Saves into --run directory.

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def load_meta(meta_path: Path):
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def z_centers(Nz: int, Lz: float) -> np.ndarray:
    dz = Lz / Nz
    return (np.arange(Nz) + 0.5) * dz


def nearest_z_indices(zc_all: np.ndarray, targets: np.ndarray) -> np.ndarray:
    # Map target heights (m) to nearest z-center indices, unique, sorted.
    idx = [int(np.argmin(np.abs(zc_all - zt))) for zt in targets]
    return np.array(sorted(set(idx)), dtype=int)


def compute_field(U_t_zi: np.ndarray, field: str) -> np.ndarray:
    # U_t_zi: (Ny, Nx, 3)
    if field == "Umag":
        return np.sqrt(np.sum(U_t_zi * U_t_zi, axis=-1))
    comp = {"Ux": 0, "Uy": 1, "Uz": 2}[field]
    return U_t_zi[..., comp]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="folder containing U_all_U.npy and (optionally) U_all.meta.json")
    ap.add_argument("--U", default="U_all_U.npy", help="velocity .npy file name")
    ap.add_argument("--meta", default="U_all.meta.json", help="metadata json file name")
    ap.add_argument("--out", default=None, help="output GIF name (saved inside --run). default: zstack_6.gif")

    ap.add_argument("--field", choices=["Umag", "Ux", "Uy", "Uz"], default="Umag",
                    help="what to visualize per plane")
    ap.add_argument("--normalize", choices=["none", "mean_xy"], default="none",
                    help="none: raw field; mean_xy: divide each plane by its (x,y) mean at that time and z")

    # NEW: pick planes by physical height spacing
    ap.add_argument("--z-step", type=float, default=100.0, help="z spacing in meters (default 100)")
    ap.add_argument("--z-start", type=float, default=None,
                    help="first target z in meters. default: z_step/2 (cell-center-friendly)")
    ap.add_argument("--max-planes", type=int, default=6, help="cap number of planes (default 6)")

    # Layout for 6 plots
    ap.add_argument("--rows", type=int, default=2, help="subplot rows (default 2)")
    ap.add_argument("--cols", type=int, default=3, help="subplot cols (default 3)")

    ap.add_argument("--t0", type=int, default=0, help="start time index (0-based)")
    ap.add_argument("--t1", type=int, default=-1, help="end time index inclusive (default: last)")
    ap.add_argument("--stride", type=int, default=1, help="use every stride-th frame")

    ap.add_argument("--fps", type=float, default=None, help="GIF fps (default: infer from meta dt and stride)")
    ap.add_argument("--clim_pct", type=float, nargs=2, default=[1.0, 99.0],
                    help="color limits from percentiles computed on the first rendered frame (default 1 99)")
    ap.add_argument("--center_axes", action="store_true",
                    help="plot x,y as centered coordinates [-Lx/2,Lx/2], [-Ly/2,Ly/2]")

    args = ap.parse_args()
    run = Path(args.run).resolve()
    U_path = run / args.U
    meta = load_meta(run / args.meta)

    if not U_path.exists():
        raise FileNotFoundError(f"Missing {U_path}")

    U = np.load(U_path, mmap_mode="r")
    if U.ndim != 5:
        raise RuntimeError(f"Expected U to have shape (T,Nz,Ny,Nx,3). Got {U.shape}")

    T, Nz, Ny, Nx, _ = U.shape

    # Domain lengths
    Lx = float(meta.get("Lx", 1800.0))
    Ly = float(meta.get("Ly", 800.0))
    Lz = float(meta.get("Lz", 600.0))
    dt = float(meta.get("dt", meta.get("dt_target", 0.5)))

    # Choose target heights
    z0 = (args.z_step / 2.0) if (args.z_start is None) else float(args.z_start)
    targets = np.arange(z0, Lz + 1e-9, args.z_step)
    zc_all = z_centers(Nz, Lz)
    z_idx = nearest_z_indices(zc_all, targets)

    # Cap to max-planes (take evenly spaced among chosen if too many)
    if len(z_idx) > args.max_planes:
        pick = np.linspace(0, len(z_idx) - 1, args.max_planes).round().astype(int)
        z_idx = z_idx[pick]

    zc = zc_all[z_idx]
    if len(z_idx) == 0:
        raise RuntimeError("No z-planes selected. Check --z-step/--z-start and Lz in meta.")

    # Time indices
    t0 = max(0, args.t0)
    t1 = (T - 1) if args.t1 < 0 else min(T - 1, args.t1)
    frames = list(range(t0, t1 + 1, max(1, args.stride)))
    if not frames:
        raise RuntimeError("No frames selected. Check --t0/--t1/--stride against U.shape[0].")

    # Plot extent
    if args.center_axes:
        extent = [-Lx / 2, Lx / 2, -Ly / 2, Ly / 2]
        xlab, ylab = "x (m, centered)", "y (m, centered)"
    else:
        extent = [0.0, Lx, 0.0, Ly]
        xlab, ylab = "x (m)", "y (m)"

    R, C = args.rows, args.cols
    if R * C < len(z_idx):
        raise ValueError(f"rows*cols={R*C} but need >= {len(z_idx)} subplots")

    fig, axs = plt.subplots(R, C, figsize=(4.8 * C, 4.0 * R), constrained_layout=True)
    axs = np.array(axs).reshape(R, C)

    # First frame for color limits
    t_first = frames[0]
    first_stack = []
    for zi in z_idx:
        fld = compute_field(U[t_first, zi, :, :, :], args.field).astype(np.float32)
        if args.normalize == "mean_xy":
            m = float(np.mean(fld))
            if m != 0.0:
                fld = fld / m
        first_stack.append(fld)

    first_concat = np.concatenate([a.ravel() for a in first_stack])
    vmin, vmax = map(float, np.percentile(first_concat, args.clim_pct))

    ims = []
    for k in range(R * C):
        r, c = divmod(k, C)
        ax = axs[r, c]
        ax.set_xticks([])
        ax.set_yticks([])
        if k < len(z_idx):
            im = ax.imshow(first_stack[k], origin="lower", extent=extent,
                           vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
            ax.set_title(f"z ≈ {zc[k]:.0f} m (idx {z_idx[k]})", fontsize=10)
            ims.append(im)
        else:
            ax.axis("off")

    cbar = fig.colorbar(ims[0], ax=axs, shrink=0.85, pad=0.01)
    cbar.set_label(f"{args.field}" + ("/⟨·⟩xy" if args.normalize == "mean_xy" else ""), rotation=90)

    axs[-1, 0].set_xlabel(xlab)
    axs[-1, 0].set_ylabel(ylab)

    def update(ti: int):
        t = ti
        for j, zi in enumerate(z_idx):
            fld = compute_field(U[t, zi, :, :, :], args.field).astype(np.float32)
            if args.normalize == "mean_xy":
                m = float(np.mean(fld))
                if m != 0.0:
                    fld = fld / m
            ims[j].set_data(fld)
        fig.suptitle(f"t index = {t}    t = {t * dt:.2f} s", fontsize=13)
        return ims

    fps = args.fps if args.fps is not None else (1.0 / (dt * max(1, args.stride)))
    out_path = run / (args.out or "zstack_6.gif")

    ani = FuncAnimation(fig, update, frames=frames, blit=False, interval=1000.0 / fps)
    ani.save(out_path, writer=PillowWriter(fps=fps))
    print(f"[saved] {out_path}")
    print(f"[info] U shape={U.shape} field={args.field} frames={len(frames)} z_planes={len(z_idx)}")
    print(f"[info] z_targets_step={args.z_step}m z_selected(m)={np.round(zc,3).tolist()}")


if __name__ == "__main__":
    main()