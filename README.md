# NeutralABLBoxLES

Neutral atmospheric boundary layer (ABL) large-eddy simulation (LES) in a horizontally periodic box, provided as a clean OpenFOAM case template. The setup omits buoyancy and thermal stratification, so turbulence is shear driven over a rough wall and the mean streamwise wind is maintained by a body-force term in a forcing slab.

The workflow runs a spin-up to reach a statistically steady state, then writes 3D velocity snapshots on an exact cadence (default 0.5 s) during a capture window. This output pattern fits downstream post-processing, data-assimilation, and machine-learning pipelines that expect uniform time sampling.

## Requirements

You need OpenFOAM.com (tested with v2412). For parallel runs you need MPI. The optional exporter uses Python 3 with NumPy.

## Quick start

After cloning, rebuild generated artifacts that are not tracked in git:

```bash
blockMesh
topoSet
```

Run the case with the wrapper script:

```bash
chmod +x run_les.sh
NP=8 SPIN_MULT=20 CAPTURE_DURATION=200 WR_INTERVAL=0.5 ./run_les.sh
```

### Run controls

The wrapper reads these environment variables:

- `NP` sets the MPI rank count. Set `NP=1` for serial runs.
- `SPIN_MULT` scales the spin-up length in turnover-time units.
- `CAPTURE_DURATION` sets the capture window length in seconds.
- `WR_INTERVAL` sets the capture write cadence in seconds.
- `RECONSTRUCT` toggles `reconstructPar` (use `RECONSTRUCT=0` to keep only `processor*/` folders).
- `SOLVER` selects the solver (default `pimpleFoam`).

## Outputs

Each run directory under `runs/` contains the logs and the capture metadata:

- `log.spinup`, `log.capture`
- `capture_window.json`
- reconstructed time folders (if `RECONSTRUCT=1`) or `processor*/` folders (if `RECONSTRUCT=0`)

## Export velocity cubes

If you ran in parallel, use reconstructed time folders for export (the wrapper does this when `RECONSTRUCT=1`). Then export a uniform time grid:

```bash
python3 export_cube.py --run runs/<run_dir> --t0 0 --t1 200 --dt 0.5 --out-prefix U_all
```

The exporter writes `.npy` arrays plus a JSON metadata file describing grid spacing, domain extents, and the sampled time stamps.

## Configuration map

The case is intentionally modular. The table below points to the files you typically edit for common changes.

| Change you want | File to edit | What to edit |
|---|---|---|
| Change domain size or resolution | `system/blockMeshDict` | Domain extents (vertices) and cell counts (`blocks`). Rerun `blockMesh`. |
| Move or resize the forcing slab | `system/topoSetDict` | The `boxToCell` z-range and extents that build `forcingZone`. Rerun `topoSet`. |
| Change target mean wind speed | `system/fvOptions` | `meanVelocityForceCoeffs.Ubar` (vector). |
| Tune forcing response | `system/fvOptions` | `meanVelocityForceCoeffs.relaxation` (stability vs responsiveness). |
| Change surface roughness | `0/nut` | `bottom` patch `atmNutUWallFunction.z0`. |
| Change SGS model | `constant/turbulenceProperties` | `LESModel` and `delta` options. |
| Change time step and CFL control | `system/controlDict.*` | `maxCo`, `maxDeltaT`, and `deltaT` behavior. |
| Change output cadence | `system/controlDict.capture` | `writeControl` and `writeInterval`. Keep `adjustableRunTime` if you want an exact cadence. |
| Change solver settings | `system/fvSolution` | PIMPLE loops, tolerances, relaxation. |
| Change discretization | `system/fvSchemes` | Time and spatial schemes (stability and dissipation). |
| Change parallel decomposition | `system/decomposeParDict` | Decomposition method and layout. |

## Repository notes

This repository tracks the case template only. It does not track generated mesh connectivity (`constant/polyMesh/`) or solver outputs (time folders, `processor*/`, `postProcessing/`, `runs/`). Rebuild the mesh and forcing zone with `blockMesh` and `topoSet` after cloning.

## References

Chen, Y., Wang, D., Feng, D., Tian, G., Gupta, V., Cao, R., Wan, M., Chen, S. Three-dimensional spatiotemporal wind field reconstruction based on LiDAR and multi-scale PINN. *Applied Energy* 377 (2025) 124577. DOI: 10.1016/j.apenergy.2024.124577.

### BibTeX

```bibtex
@article{Chen2025ThreeDimensionalWindReconstruction,
  title   = {Three-dimensional spatiotemporal wind field reconstruction based on {LiDAR} and multi-scale {PINN}},
  author  = {Chen, Yuanqing and Wang, Ding and Feng, Dachuan and Tian, Geng and Gupta, Vikrant and Cao, Renjing and Wan, Minping and Chen, Shiyi},
  journal = {Applied Energy},
  volume  = {377},
  pages   = {124577},
  year    = {2025},
  doi     = {10.1016/j.apenergy.2024.124577}
}
```
