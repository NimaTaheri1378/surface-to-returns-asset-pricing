#!/usr/bin/env bash
set -o pipefail

mkdir -p logs
export PYTHONPATH="${PYTHONPATH:-src}"
PY="${SURFACE_RETURNS_PYTHON:-python}"

echo "flagship_sdf_full slurm_job=${SLURM_JOB_ID:-none} host=$(hostname) started=$(date -Is)"
"${PY}" -c 'import torch; print("cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'

"${PY}" scripts/train_conditional_autoencoder_sdf.py \
  --epochs 24 \
  --validation-months 24 \
  --early-stop-patience 6 \
  --hidden-dim 192 \
  --branch-dim 96 \
  --latent-dim 8 \
  --dropout 0.10 \
  --ig-max-assets 8000 \
  --ig-steps 24 \
  --min-assets-per-month 80 \
  2>&1 | tee logs/conditional_autoencoder_sdf_flagship_full.log
sdf_status=${PIPESTATUS[0]}
echo "flagship_sdf_train_finished=$(date -Is) status=${sdf_status}"
if [ "${sdf_status}" -ne 0 ]; then
  exit "${sdf_status}"
fi

"${PY}" scripts/run_model_interpretation.py --top-n 30 \
  2>&1 | tee logs/model_interpretation_after_flagship_sdf.log
interp_status=${PIPESTATUS[0]}
echo "flagship_sdf_interpretation_finished=$(date -Is) status=${interp_status}"
if [ "${interp_status}" -ne 0 ]; then
  exit "${interp_status}"
fi

"${PY}" scripts/build_evidence_dashboard.py \
  2>&1 | tee logs/evidence_dashboard_after_flagship_sdf.log
dash_status=${PIPESTATUS[0]}
echo "flagship_sdf_dashboard_finished=$(date -Is) status=${dash_status}"
if [ "${dash_status}" -ne 0 ]; then
  exit "${dash_status}"
fi

"${PY}" scripts/public_safety_scan.py \
  2>&1 | tee logs/public_safety_scan_after_flagship_sdf.log
scan_status=${PIPESTATUS[0]}
echo "flagship_sdf_full finished=$(date -Is) status=${scan_status}"
exit "${scan_status}"
