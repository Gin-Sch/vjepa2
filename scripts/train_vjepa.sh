#!/bin/bash
#SBATCH --job-name=TrainVJEPA2
#SBATCH --output=logs/train_vjepa2_%j.out
#SBATCH --error=logs/train_vjepa2_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=99:00:00
#SBATCH --gres=gpu:h200:1

# Pretrain V-JEPA 2 from a user YAML+CSV, plot training metrics, extract frozen
# encoder embeddings, and save a label-colored t-SNE. No attentive probe.
#
# Usage:
#   PRETRAIN_CONFIG=configs/train/vitl16/pretrain_vitl16_h200.yaml \
#   INDEX_CSV=/path/to/videos.csv \
#   sbatch scripts/train_vjepa.sh
#
# INDEX_CSV defaults to the first entry under data.datasets in PRETRAIN_CONFIG.
# OUT_DIR defaults to folder from PRETRAIN_CONFIG with a timestamp
# suffix (_YYYY-MM-DD_HHMMSS) so each launch writes to a fresh directory.
# Set TIMESTAMP_OUT_DIR=0 to disable (e.g. resume into an existing folder).
# A run-local YAML with folder rewritten is passed to training/export.

set -euo pipefail

module load python3/anaconda3-2024.06-py3.12
module load nvidia-hpc-sdk
module load nvhpc-hpcx-cuda12/24.7

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TRAINING_ENV=server

VJEPA2_DIR="${VJEPA2_DIR:-/rhomes/gschum/vjepa2}"
PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-${VJEPA2_DIR}/configs/train/vitl16/pretrain_vitl16_h200.yaml}"
TIMESTAMP_OUT_DIR="${TIMESTAMP_OUT_DIR:-1}"

conda activate vjepa_pretrain

cd "${VJEPA2_DIR}"
export PYTHONPATH="${VJEPA2_DIR}:${PYTHONPATH:-}"
mkdir -p "${VJEPA2_DIR}/logs"

# Resolve folder, write_tag, and default INDEX_CSV from the pretrain YAML.
eval "$(
python - "${PRETRAIN_CONFIG}" <<'PY'
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1])
with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

folder = cfg.get("folder") or (cfg.get("logging") or {}).get("folder")
write_tag = (cfg.get("logging") or {}).get("write_tag") or cfg.get("write_tag") or "jepa"
datasets = cfg.get("data", {}).get("datasets") or []
index_csv = datasets[0] if datasets else ""

def sh_quote(value: str) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"

print(f"RESOLVED_OUT_DIR={sh_quote(folder)}")
print(f"RESOLVED_WRITE_TAG={sh_quote(write_tag)}")
print(f"RESOLVED_INDEX_CSV={sh_quote(index_csv)}")
PY
)"

BASE_OUT_DIR="${OUT_DIR:-${RESOLVED_OUT_DIR}}"
BASE_OUT_DIR="${BASE_OUT_DIR%/}"
WRITE_TAG="${WRITE_TAG:-${RESOLVED_WRITE_TAG}}"
INDEX_CSV="${INDEX_CSV:-${RESOLVED_INDEX_CSV}}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y-%m-%d_%H%M%S)}"

if [[ "${TIMESTAMP_OUT_DIR}" == "1" ]]; then
  OUT_DIR="${BASE_OUT_DIR}_${RUN_TIMESTAMP}"
else
  OUT_DIR="${BASE_OUT_DIR}"
fi

if [[ -z "${OUT_DIR}" ]]; then
  echo "OUT_DIR is empty; set folder in ${PRETRAIN_CONFIG} or pass OUT_DIR" >&2
  exit 1
fi

if [[ -z "${INDEX_CSV}" ]]; then
  echo "INDEX_CSV is empty; set INDEX_CSV or data.datasets in ${PRETRAIN_CONFIG}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

# Materialize a run-local config so train + export share the timestamped folder.
RUN_CONFIG="${OUT_DIR}/params-pretrain-launch.yaml"
python - "${PRETRAIN_CONFIG}" "${RUN_CONFIG}" "${OUT_DIR}" "${WRITE_TAG}" "${RUN_TIMESTAMP}" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, out_dir, write_tag, timestamp = sys.argv[1:6]
with Path(src).open("r", encoding="utf-8") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# V-JEPA2 uses top-level folder; keep logging.folder in sync if present.
cfg["folder"] = out_dir
logging_cfg = cfg.setdefault("logging", {})
logging_cfg["folder"] = out_dir

mlflow = logging_cfg.setdefault("mlflow", {})
if not mlflow.get("run_name"):
    mlflow["run_name"] = f"{write_tag}_{timestamp}"

Path(dst).parent.mkdir(parents=True, exist_ok=True)
with Path(dst).open("w", encoding="utf-8") as f:
    yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)
print(dst)
PY

LOG_CSV="${LOG_CSV:-${OUT_DIR}/${WRITE_TAG}_r0.csv}"
RAW_NPZ="${RAW_NPZ:-${OUT_DIR}/embeddings_raw.npz}"
METRICS_PNG="${METRICS_PNG:-${OUT_DIR}/train_metrics.png}"
TSNE_PNG="${TSNE_PNG:-${OUT_DIR}/embeddings_tsne.png}"

echo "=== [1/4] Pretrain ==="
echo "config=${PRETRAIN_CONFIG}"
echo "run_config=${RUN_CONFIG}"
echo "out_dir=${OUT_DIR}"
echo "index_csv=${INDEX_CSV}"

python -m app.main \
  --fname "${RUN_CONFIG}" \
  --devices cuda:0

echo "=== [2/4] Plot training metrics ==="
python scripts/plot_vjepa_train_metrics.py \
  --log-csv "${LOG_CSV}" \
  --output "${METRICS_PNG}" \
  --title "V-JEPA2 pretrain (${WRITE_TAG})"

echo "=== [3/4] Extract frozen embeddings ==="
python scripts/export_embeddings_vjepa.py extract \
  --pretrain-config "${RUN_CONFIG}" \
  --index-csv "${INDEX_CSV}" \
  --raw-output "${RAW_NPZ}" \
  --num-workers 4

echo "=== [4/4] t-SNE of embeddings ==="
python scripts/plot_vjepa_embeddings_tsne.py \
  --raw-npz "${RAW_NPZ}" \
  --index-csv "${INDEX_CSV}" \
  --output "${TSNE_PNG}" \
  --title "V-JEPA2 encoder t-SNE (${WRITE_TAG})"

echo "Done."
echo "  checkpoint: ${OUT_DIR}/${WRITE_TAG}-latest.pt"
echo "  run config: ${RUN_CONFIG}"
echo "  metrics:    ${METRICS_PNG}"
echo "  embeddings: ${RAW_NPZ}"
echo "  t-SNE:      ${TSNE_PNG}"
