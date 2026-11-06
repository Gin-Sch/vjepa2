#!/bin/bash
#SBATCH --job-name=TrainVJEPA2Mouse
#SBATCH --output=/rhomes/gschum/vjepa2/logs/train_vjepa2_mouse_%j.out
#SBATCH --error=/rhomes/gschum/vjepa2/logs/train_vjepa2_mouse_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=99:00:00
#SBATCH --gres=gpu:h200:1

# V-JEPA 2 on mouse beam-walk videos, then a MouseBeamWalk embedding table.
#
# VARIANT selects the pretraining set (mirrors the two V-JEPA1 tables):
#   alldata  (default) all 144 mice / 1544 videos   -> vjepa2_camera6_alldata
#   heldout  116 mice / 1264 videos (28 mice out)   -> vjepa2_camera6
#
#   [1/4] pretrain on the variant's CSV
#   [2/4] plot training metrics
#   [3/4] frozen-encoder features for ALL 1544 camera-6 videos
#         (MouseBeamWalk manifests/vjepa_index_camera6.csv, the same index the
#         V-JEPA1 tables use, so every leaderboard row covers identical videos)
#   [4/4] join manifest metadata -> ${MBW_DIR}/outputs/embeddings/<table>
#
# Usage:
#   sbatch scripts/train_vjepa2_mouse.sh                    # all data
#   VARIANT=heldout sbatch scripts/train_vjepa2_mouse.sh    # mouse-disjoint
#
# Export only, from a finished run (skips stages 1-2):
#   RUN_DIR=/rhomes/gschum/vjepa2/outputs/vjepa2_pretrain_mouse_alldata_<stamp> \
#   SKIP_TRAIN=1 sbatch scripts/train_vjepa2_mouse.sh
#
# Stage 4 runs in myenv_mini from the MouseBeamWalk checkout: it needs that
# repo's `src` package (this checkout ships its own `src`) plus pandas/pyarrow.

set -euo pipefail

module load python3/anaconda3-2024.06-py3.12
module load nvidia-hpc-sdk
module load nvhpc-hpcx-cuda12/24.7

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TRAINING_ENV=server

VJEPA2_DIR="${VJEPA2_DIR:-/rhomes/gschum/vjepa2}"
MBW_DIR="${MBW_DIR:-/rhomes/gschum/MouseBeamWalk}"
VARIANT="${VARIANT:-alldata}"
case "${VARIANT}" in
  alldata)
    DEFAULT_TABLE="vjepa2_camera6_alldata"
    DEFAULT_MODEL_NAME="V-JEPA2 ViT-L/16 (ours, all-data)" ;;
  heldout)
    DEFAULT_TABLE="vjepa2_camera6"
    DEFAULT_MODEL_NAME="V-JEPA2 ViT-L/16 (ours)" ;;
  *)
    echo "VARIANT must be alldata or heldout, got '${VARIANT}'" >&2
    exit 1 ;;
esac
PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-${VJEPA2_DIR}/configs/train/vitl16/pretrain_vitl16_mouse_${VARIANT}.yaml}"
EXPORT_INDEX_CSV="${EXPORT_INDEX_CSV:-${MBW_DIR}/manifests/vjepa_index_camera6.csv}"
MANIFEST="${MANIFEST:-${MBW_DIR}/manifests/manifest.jsonl}"
OUT_PREFIX="${OUT_PREFIX:-${MBW_DIR}/outputs/embeddings/${DEFAULT_TABLE}}"
MODEL_NAME="${MODEL_NAME:-${DEFAULT_MODEL_NAME}}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

export MBW_DIR
mkdir -p "${VJEPA2_DIR}/logs" "$(dirname "${OUT_PREFIX}")"

conda activate vjepa2
cd "${VJEPA2_DIR}"
export PYTHONPATH="${VJEPA2_DIR}:${PYTHONPATH:-}"

if [[ "${SKIP_TRAIN}" == "1" ]]; then
  if [[ -z "${RUN_DIR:-}" ]]; then
    echo "SKIP_TRAIN=1 needs RUN_DIR=<finished run folder>" >&2
    exit 1
  fi
  OUT_DIR="${RUN_DIR%/}"
  RUN_CONFIG="${OUT_DIR}/params-pretrain-launch.yaml"
else
  # Fresh timestamped folder next to the config's `folder`, with a run-local
  # YAML so training and export read the same checkpoint location.
  RUN_TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
  BASE_OUT_DIR="$(python -c "import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print(c.get('folder') or c['logging']['folder'])" "${PRETRAIN_CONFIG}")"
  OUT_DIR="${BASE_OUT_DIR%/}_${RUN_TIMESTAMP}"
  RUN_CONFIG="${OUT_DIR}/params-pretrain-launch.yaml"
  mkdir -p "${OUT_DIR}"
  python - "${PRETRAIN_CONFIG}" "${RUN_CONFIG}" "${OUT_DIR}" "${RUN_TIMESTAMP}" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, out_dir, timestamp = sys.argv[1:5]
cfg = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
cfg["folder"] = out_dir
logging_cfg = cfg.setdefault("logging", {})
logging_cfg["folder"] = out_dir
mlflow = logging_cfg.setdefault("mlflow", {})
if not mlflow.get("run_name"):
    mlflow["run_name"] = f"{logging_cfg.get('write_tag', 'vjepa2')}_{timestamp}"
Path(dst).write_text(yaml.dump(cfg, sort_keys=False, default_flow_style=False), encoding="utf-8")
PY
fi

if [[ ! -f "${RUN_CONFIG}" ]]; then
  echo "Run config not found: ${RUN_CONFIG}" >&2
  exit 1
fi
WRITE_TAG="$(python -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['logging']['write_tag'])" "${RUN_CONFIG}")"
RAW_NPZ="${OUT_DIR}/embeddings_raw_camera6.npz"

echo "variant=${VARIANT}"
echo "run_config=${RUN_CONFIG}"
echo "out_dir=${OUT_DIR}"
echo "export_index=${EXPORT_INDEX_CSV}"
echo "table=${OUT_PREFIX}"

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  echo "=== [1/4] Pretrain ==="
  python -m app.main --fname "${RUN_CONFIG}" --devices cuda:0

  echo "=== [2/4] Plot training metrics ==="
  python scripts/plot_vjepa_train_metrics.py \
    --log-csv "${OUT_DIR}/${WRITE_TAG}_r0.csv" \
    --output "${OUT_DIR}/train_metrics.png" \
    --title "V-JEPA2 mouse pretrain (${WRITE_TAG})"
fi

echo "=== [3/4] Extract frozen embeddings (all camera-6 videos) ==="
python scripts/export_embeddings_vjepa.py extract \
  --pretrain-config "${RUN_CONFIG}" \
  --index-csv "${EXPORT_INDEX_CSV}" \
  --raw-output "${RAW_NPZ}" \
  --num-workers 8

echo "=== [4/4] Join manifest metadata -> embedding table ==="
conda activate myenv_mini
cd "${MBW_DIR}"
python "${VJEPA2_DIR}/scripts/export_embeddings_vjepa.py" finalize \
  --raw-input "${RAW_NPZ}" \
  --index-csv "${EXPORT_INDEX_CSV}" \
  --manifest-path "${MANIFEST}" \
  --output-prefix "${OUT_PREFIX}" \
  --model-name "${MODEL_NAME}"

echo "Done."
echo "  checkpoint: ${OUT_DIR}/${WRITE_TAG}-latest.pt"
echo "  raw:        ${RAW_NPZ}"
echo "  table:      ${OUT_PREFIX}.{npz,parquet}"
echo "Next (in ${MBW_DIR}, env myenv_mini):"
echo "  python scripts/p2_test_set.py --embeddings simsiam=outputs/embeddings/simsiam_camera6 vjepa=outputs/embeddings/vjepa_camera6_alldata vjepa2_${VARIANT}=${OUT_PREFIX} --output-dir outputs/p2_testset"
