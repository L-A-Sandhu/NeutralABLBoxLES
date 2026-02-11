#!/usr/bin/env bash
set -euo pipefail

# Run from any PWD: resolve case root from script location
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ---------------- User knobs (env vars) ----------------
SOLVER="${SOLVER:-pimpleFoam}"
NP="${NP:-1}"
SPIN_MULT="${SPIN_MULT:-20}"         # multiples of t*
CAPTURE_SEC="${CAPTURE_SEC:-200}"    # seconds
DT="${DT:-0.5}"                      # solver deltaT
CAPTURE_DT="${CAPTURE_DT:-0.5}"      # write interval during capture
RECONSTRUCT="${RECONSTRUCT:-0}"      # 1 => reconstructPar -latestTime after capture

# ---------------- ABL constants (paper-like defaults) ----------------
UINF="${UINF:-8}"        # m/s
H="${H:-600}"            # m
Z0="${Z0:-0.01}"         # m
ZREF="${ZREF:-90}"       # m
KAPPA="${KAPPA:-0.41}"   # -

# Basic tool sanity
command -v "$SOLVER" >/dev/null
command -v blockMesh >/dev/null
command -v decomposePar >/dev/null
command -v mpirun >/dev/null
command -v python3 >/dev/null

# ---- compute u_tau and t* WITHOUT relying on exported env ----
UTAU="$(python3 - <<PY
import math
U=float("$UINF"); z0=float("$Z0"); zref=float("$ZREF"); k=float("$KAPPA")
print(k*U/math.log(zref/z0))
PY
)"
TSTAR="$(python3 - <<PY
H=float("$H"); ut=float("$UTAU")
print(H/ut)
PY
)"
TSPIN="$(python3 - <<PY
import math
print(int(round(float("$TSTAR")*float("$SPIN_MULT"))))
PY
)"
ENDTIME="$((TSPIN + CAPTURE_SEC))"

ts="$(date +%Y%m%d_%H%M%S)"
RUNDIR="$ROOT/runs/run_${ts}_spin${SPIN_MULT}_np${NP}"

echo "[info] SOLVER=$SOLVER NP=$NP"
echo "[info] u_tau_est=$UTAU  t*=$TSTAR  SPIN_MULT=$SPIN_MULT => TSPIN=$TSPIN s"
echo "[info] CAPTURE_SEC=$CAPTURE_SEC => endTime=$ENDTIME s"
echo "[info] template: $ROOT"
echo "[info] run dir : $RUNDIR"

mkdir -p "$RUNDIR"

# Copy template case into run dir (do not bring old runs/processor*/postProcessing/logs)
rsync -a \
  --exclude 'runs' \
  --exclude 'postProcessing' \
  --exclude 'processor*' \
  --exclude 'log.*' \
  "$ROOT"/ "$RUNDIR"/

# Write capture window metadata (handy for export scripts)
cat > "$RUNDIR/capture_window.json" <<EOF
{
  "np": $NP,
  "spin_mult": "$SPIN_MULT",
  "u_tau": $UTAU,
  "t_star": $TSTAR,
  "t_spin": $TSPIN,
  "capture_sec": $CAPTURE_SEC,
  "end_time": $ENDTIME,
  "deltaT": $DT,
  "capture_dt": $CAPTURE_DT
}
EOF

# Helper: set dictionary entries quietly
setDict () {
  local dict="$1" entry="$2" value="$3"
  if command -v foamDictionary >/dev/null 2>&1; then
    foamDictionary "$dict" -entry "$entry" -set "$value" >/dev/null 2>&1
  else
    sed -i -E "s|^([[:space:]]*$entry[[:space:]]+).*(;[[:space:]]*)$|\1$value\2|g" "$dict"
  fi
}

# Always generate decomposeParDict INSIDE run dir using NP
cat > "$RUNDIR/system/decomposeParDict" <<EOF
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}
numberOfSubdomains $NP;
method          scotch;
distributed     no;
roots           ();
EOF

# ---------------- Stage 0: mesh (serial) ----------------
(
  cd "$RUNDIR"
  blockMesh > log.blockMesh 2>&1
  if [ -f system/topoSetDict ]; then
    topoSet > log.topoSet 2>&1 || true
  fi
)

# ---------------- Stage 1: spin-up ----------------
echo "[info] spin-up: running to t=$TSPIN s"
(
  cd "$RUNDIR"
  cp -f system/controlDict.spinup system/controlDict

  setDict system/controlDict "application"   "$SOLVER"
  setDict system/controlDict "deltaT"        "$DT"
  setDict system/controlDict "startFrom"     "startTime"
  setDict system/controlDict "startTime"     "0"
  setDict system/controlDict "stopAt"        "endTime"
  setDict system/controlDict "endTime"       "$TSPIN"
  setDict system/controlDict "writeControl"  "runTime"
  setDict system/controlDict "writeInterval" "$TSPIN"   # write once at end of spin-up
  setDict system/controlDict "purgeWrite"    "0"

  if [ "$NP" -gt 1 ]; then
    rm -rf processor*
    decomposePar -force > log.decompose 2>&1
    mpirun -np "$NP" "$SOLVER" -parallel > log.spinup 2>&1
  else
    "$SOLVER" > log.spinup 2>&1
  fi
)

# ---------------- Stage 2: capture ----------------
echo "[info] capture: running to t=$ENDTIME s with writeInterval=$CAPTURE_DT"
(
  cd "$RUNDIR"
  cp -f system/controlDict.capture system/controlDict

  setDict system/controlDict "startFrom"     "latestTime"
  setDict system/controlDict "stopAt"        "endTime"
  setDict system/controlDict "endTime"       "$ENDTIME"
  setDict system/controlDict "writeControl"  "adjustableRunTime"
  setDict system/controlDict "writeInterval" "$CAPTURE_DT"
  setDict system/controlDict "purgeWrite"    "0"

  if [ "$NP" -gt 1 ]; then
    mpirun -np "$NP" "$SOLVER" -parallel > log.capture 2>&1
    if [ "$RECONSTRUCT" = "1" ]; then
      reconstructPar -time "${TSPIN}:${ENDTIME}" > log.reconstruct 2>&1
    fi
  else
    "$SOLVER" > log.capture 2>&1
  fi
)

echo "[done] Run finished: $RUNDIR"
echo "[note] NP>1 writes into processor*/time/. If RECONSTRUCT=1, you also get reconstructed latestTime."
