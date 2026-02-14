# PeriodicNeutralABLBoxLES

This repository provides a reproducible OpenFOAM case template for a neutral Atmospheric Boundary Layer (ABL) Large-Eddy Simulation (LES) in a horizontally periodic box. The setup models a mechanically driven boundary layer over a rough wall with a shear-free top boundary and cyclic lateral boundaries. It maintains a prescribed mean streamwise velocity in a thin forcing slab via `meanVelocityForce`. Together these choices yield a controlled reference flow that supports algorithm development in wind-field reconstruction, data assimilation, and spatiotemporal learning.

The repository stays lightweight by tracking only dictionaries, initial fields, and helper scripts. You regenerate the mesh (`blockMesh`) and forcing zone (`topoSet`) at run time. You do not commit generated mesh connectivity (`constant/polyMesh`) or solver outputs.

## What the case produces

During the capture stage, the solver writes full 3D velocity snapshots `U(x,y,z,t)` on an exact and user-controlled cadence (default `0.5 s`). The control dictionaries use `writeControl adjustableRunTime`, which aligns writes to the requested physical time grid even when the solver adapts time step size under a CFL constraint.

## Baseline configuration

| Component | Value |
|---|---|
| Domain | `Lx × Ly × Lz = 1800 × 800 × 600 m` |
| Mesh | `Nx × Ny × Nz = 180 × 80 × 80` (single structured `blockMesh` block) |
| Lateral boundaries | `cyclic` in `x` and `y` |
| Top boundary | `slip` (shear free) |
| Bottom boundary | rough wall via `atmNutUWallFunction` with `z0 = 0.01 m` |
| SGS model | `dynamicLagrangian` (LES) |
| Mean-flow maintenance | `meanVelocityForce` on `cellZone forcingZone` (built by `topoSet`) |
| Default target mean speed | `Ubar = (8 0 0) m/s` |

## Repository layout

```
0/                         initial fields (U, p, nut, flm, fmm, ...)
constant/                  physical properties and LES model
system/                    numerics, controlDicts, topoSetDict, fvOptions
run_les.sh                 one-command spin-up + capture runner
export_cube.py             exports reconstructed snapshots to NumPy
```

Generated artifacts such as `runs/`, time folders, `processor*/`, `postProcessing/`, and `constant/polyMesh/` should remain untracked.

## Requirements

You need an OpenFOAM.com release that supports `meanVelocityForce` (tested with v2412). For parallel execution, you need an MPI runtime consistent with your OpenFOAM build. The exporter uses Python 3 with NumPy and Matplotlib.

## Run with the wrapper

The recommended workflow runs inside an isolated timestamped directory under `runs/`. The runner creates the run directory, copies the template, regenerates the mesh and forcing zone, performs spin-up, then performs capture.

```bash
# from the case root
chmod +x run_les.sh

# example: 8 ranks, 20 turnover-times of spin-up, then 200 s capture at 0.5 s cadence
NP=8 SPIN_MULT=20 CAPTURE_SEC=200 CAPTURE_DT=0.5 ./run_les.sh
```

The runner estimates a turnover time using a log-law friction-velocity estimate
`u_tau ≈ κ U_inf / log((z_ref + z0)/z0)` and sets `t* = H/u_tau`. It then sets `t_spin = SPIN_MULT × t*` and captures on `[t_spin, t_spin + CAPTURE_SEC]`. It writes `capture_window.json` in the run directory so you can recover these values later.

If you run in parallel and want single-domain time folders for post-processing, enable reconstruction:

```bash
NP=8 RECONSTRUCT=1 ./run_les.sh
```

### Runtime knobs (environment variables)

The wrapper exposes the most common run controls as environment variables.

| Variable | Default | Effect |
|---|---:|---|
| `SOLVER` | `pimpleFoam` | OpenFOAM solver executable. |
| `NP` | `1` | MPI ranks (when `NP>1`, the runner uses `mpirun` and writes a matching `decomposeParDict`). |
| `SPIN_MULT` | `20` | Spin-up duration multiplier in turnover times (`t_spin = SPIN_MULT × t*`). |
| `CAPTURE_SEC` | `200` | Capture duration in seconds. |
| `CAPTURE_DT` | `0.5` | Write cadence in seconds during capture (`writeInterval`). |
| `DT` | `0.05` | Base time step (`deltaT`) used for both stages (subject to CFL). |
| `RECONSTRUCT` | `0` | If `1`, runs `reconstructPar` over the capture window. |
| `UINF` | `8.0` | Reference mean speed used only for turnover-time estimation (`u_tau` and `t*`). |
| `Z0` | `0.01` | Roughness length used only for turnover-time estimation (keep consistent with `0/nut`). |
| `ZREF` | `90.0` | Reference height for turnover-time estimation. |
| `H` | `600.0` | Boundary-layer height used for turnover-time scaling (`t* = H/u_tau`). |

If you change forcing in `system/fvOptions` (for example `Ubar`), also update `UINF` so the turnover-time estimate remains aligned with the forcing target.

## Manual workflow

If you prefer to run in the case directory (for debugging), regenerate the mesh and forcing zone and then run the solver with the desired control dictionary:

```bash
blockMesh
topoSet
cp system/controlDict.spinup system/controlDict
pimpleFoam | tee log.spinup
cp system/controlDict.capture system/controlDict
pimpleFoam | tee log.capture
```

For parallel runs, add `decomposePar -force`, run `pimpleFoam -parallel`, and then `reconstructPar` for the capture interval.

## Export velocity cubes

`export_cube.py` expects reconstructed time folders (either from a serial run or after `reconstructPar`). It uses `capture_window.json` to define the capture origin and then selects frames on a uniform grid.

```bash
python3 export_cube.py   --run runs/run_YYYYMMDD_HHMMSS_spin20_np8   --field U   --dt 0.5   --nframes 401   --out U_cube
```

This command writes:

- `U_cube.npy` with shape `(T, Nz, Ny, Nx, 3)`
- `U_cube.meta.json` with grid and time metadata

The exporter assumes `writeFormat ascii` for OpenFOAM fields. The provided `controlDict` files already set this.

## What to edit for common changes

| Change you want | File to edit | What to change |
|---|---|---|
| Domain size or resolution | `system/blockMeshDict` | Edit `vertices` and the `blocks` cell counts (currently `hex ... (180 80 80)`). |
| Forcing slab location | `system/topoSetDict` | Edit the `boxToCell` z-range (currently `z ∈ [80, 100] m`); rerun `topoSet`. |
| Target mean velocity | `system/fvOptions` | Edit `Ubar (8 0 0)`; keep `0/U` internalField consistent if you want a warm start. |
| Roughness length | `0/nut` | Edit `z0 0.01`; keep `Z0` (runner) consistent for turnover-time diagnostics. |
| LES model | `constant/turbulenceProperties` | Change `LESModel` and coefficients. |
| Capture cadence | `system/controlDict.capture` or `CAPTURE_DT` | Prefer `CAPTURE_DT` when using the wrapper; it overrides `writeInterval` in the run directory. |
| Capture duration | `CAPTURE_SEC` | Wrapper sets `endTime = t_spin + CAPTURE_SEC`. |
| Solver stability | `system/controlDict.*` | Adjust `deltaT`, `maxCo`, and `maxDeltaT` as needed. |

Whenever you change `blockMeshDict` or `topoSetDict`, rerun `blockMesh` and `topoSet` (or rerun the wrapper) so generated geometry and zones match the dictionaries you committed.

## Scientific assumptions and limits

This case captures shear-driven neutral-wall turbulence in a periodic domain. It often provides a stable baseline for method evaluation because it avoids inflow transients and omits buoyancy and Coriolis effects. However, many atmospheric sites show stratification, rotation, and heterogeneous surfaces. For those regimes you usually move to buoyant solvers and the corresponding ABL boundary-condition stack, and you add forcing consistent with the desired geostrophic balance.

## References

Chen, Y., Wang, D., Feng, D., Tian, G., Gupta, V., Cao, R., Wan, M., & Chen, S. Three-dimensional spatiotemporal wind field reconstruction based on LiDAR and multi-scale PINN. *Applied Energy*, 377, 124577 (2025). doi:10.1016/j.apenergy.2024.124577

### BibTeX

```bibtex
@article{Chen2025MSPINNWindField,
  title   = {Three-dimensional spatiotemporal wind field reconstruction based on LiDAR and multi-scale PINN},
  author  = {Chen, Yuanqing and Wang, Ding and Feng, Dachuan and Tian, Geng and Gupta, Vikrant and Cao, Renjing and Wan, Minping and Chen, Shiyi},
  journal = {Applied Energy},
  volume  = {377},
  pages   = {124577},
  year    = {2025},
  doi     = {10.1016/j.apenergy.2024.124577}
}
```
