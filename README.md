# NeutralABLBoxLES

This repository provides an OpenFOAM case template for a neutral atmospheric boundary layer (ABL) large-eddy simulation (LES) in a horizontally periodic box. Neutral means the configuration omits buoyancy and thermal stratification, so shear over a rough wall drives turbulence and a body-force term maintains the target mean wind.

The default setup uses a 1800 × 800 × 600 m domain with a 180 × 80 × 80 `blockMesh`, cyclic lateral boundaries in x and y, a rough-wall bottom (`atmNutUWallFunction`), a shear-free top (`slip`), and the `dynamicLagrangian` subgrid-scale (SGS) model. A forcing slab (`cellZone`) spanning z = 80–100 m applies `meanVelocityForce` to maintain `Ubar = (8 0 0) m/s`.

## Requirements

- OpenFOAM.com (tested with v2412)
- MPI runtime for parallel runs
- Python 3 + NumPy for `export_cube.py` (optional)

## Run steps

After cloning, rebuild the generated mesh and the forcing zone. These are not tracked in git.

```bash
# from the case root
blockMesh
topoSet
```

Run the case with the wrapper script:

```bash
chmod +x run_les.sh

# example: 8 ranks, 20× turnover-time spin-up, 200 s capture, 0.5 s snapshots
NP=8 SPIN_MULT=20 CAPTURE_DURATION=200 WR_INTERVAL=0.5 ./run_les.sh
```

The script creates a timestamped directory under `runs/`, executes a spin-up, then switches to a capture stage that writes velocity snapshots on an exact time grid (via `writeControl adjustableRunTime`). If `RECONSTRUCT=1` (default), it reconstructs parallel fields into single-domain time folders.

### Wrapper controls

Set these as environment variables on the command line.

- `NP` number of MPI ranks (use `NP=1` for serial).
- `SPIN_MULT` spin-up duration multiplier in turnover-time units `t* = H / u_τ` (the script estimates `u_τ` from a log-law at `zref`).
- `CAPTURE_DURATION` capture length in seconds.
- `WR_INTERVAL` write interval in seconds (default 0.5).
- `SOLVER` solver name (default `pimpleFoam`).
- `RECONSTRUCT` `0/1` to disable/enable `reconstructPar` (default 1).

## Export velocity cubes

Run export after you have reconstructed time folders (the wrapper does this when `RECONSTRUCT=1`).

```bash
python3 export_cube.py --run runs/<run_dir> --t0 0 --t1 200 --dt 0.5
```

Outputs are written into the run directory:

- `U_...npy` with shape `(Nt, Nz, Ny, Nx, 3)`
- `P_...npy` with shape `(Nt, Nz, Ny, Nx)`
- `meta_UP_dt...json` with grid and timing metadata

If you change mesh resolution, update `Nx`, `Ny`, and `Nz` in `export_cube.py` so reshaping matches your grid.

## What to edit for common changes

Domain size and resolution
Edit `system/blockMeshDict` (vertices and `blocks`). Rerun `blockMesh`.

Forcing slab position and thickness
Edit `system/topoSetDict` (the `boxToCell` z-range and extents). Rerun `topoSet`. Keep the zone name consistent with `system/fvOptions`.

Target mean wind and forcing response
Edit `system/fvOptions` under `meanVelocityForceCoeffs`.
- `Ubar` sets the target mean velocity vector.
- `relaxation` controls how aggressively the forcing drives the mean.

Surface roughness
Edit `0/nut` on the `bottom` patch.
- `z0` sets roughness length (default 0.01 m).

LES closure
Edit `constant/turbulenceProperties`.
- `LESModel` selects the SGS model.
- `delta` selects the filter-width definition.

Time-step control and output cadence
Edit `system/controlDict.spinup` and `system/controlDict.capture`.
- Keep `writeControl adjustableRunTime` in capture if you want exact snapshot timing.
- If you tighten `WR_INTERVAL`, check `maxCo` and `maxDeltaT` for stability.

Numerics and solver stability
Edit `system/fvSolution` (PIMPLE, tolerances) and `system/fvSchemes` (discretization).

Parallel decomposition
Edit `system/decomposeParDict` to change the decomposition strategy.

## Related publication

A similar neutral ABL LES configuration (periodic lateral boundaries, rough-wall treatment, and mean-flow forcing) is used as reference flow data in:

Y. Chen, D. Wang, D. Feng, G. Tian, V. Gupta, R. Cao, M. Wan, S. Chen, “Three-dimensional spatiotemporal wind field reconstruction based on LiDAR and multi-scale PINN,” *Applied Energy*, 377 (2025) 124577. https://doi.org/10.1016/j.apenergy.2024.124577
