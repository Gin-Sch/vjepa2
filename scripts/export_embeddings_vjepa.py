#!/usr/bin/env python3
"""
Export frozen per-video embeddings from a V-JEPA 2 pretrain checkpoint.

Modes:
1. ``write-index`` — build an index CSV via MouseBeamWalk helpers (``MBW_DIR``).
2. ``extract`` — run the frozen encoder and dump ``X`` plus labels (this checkout).
3. ``finalize`` — join manifest metadata onto the raw matrix (``MBW_DIR``).

Example (used by ``scripts/train_vjepa.sh``)::

  cd $VJEPA2_DIR
  export PYTHONPATH=$VJEPA2_DIR:$PYTHONPATH
  python scripts/export_embeddings_vjepa.py extract \\
    --pretrain-config configs/train/vitl16/pretrain_vitl16_h200.yaml \\
    --index-csv /path/to/videos.csv \\
    --raw-output /path/to/embeddings_raw.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

VJEPA2_ROOT = Path(__file__).resolve().parents[1]
MBW_ROOT = Path(
    os.environ.get("MBW_DIR", "/rhomes/gschum/MouseBeamWalk")
).expanduser().resolve()


def _add_mbw_to_path() -> None:
    """Put MouseBeamWalk on ``sys.path`` for write-index / finalize."""
    if not MBW_ROOT.is_dir():
        raise FileNotFoundError(
            f"MouseBeamWalk not found at {MBW_ROOT}. "
            "Set MBW_DIR to the MouseBeamWalk checkout."
        )
    if str(MBW_ROOT) not in sys.path:
        sys.path.insert(0, str(MBW_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    index_parser = subparsers.add_parser(
        "write-index", help="Write the row-index-labelled V-JEPA index CSV."
    )
    index_parser.add_argument(
        "--manifest-path", type=Path, default=Path("manifests/manifest.jsonl")
    )
    index_parser.add_argument(
        "--index-csv", type=Path, default=Path("manifests/vjepa_index_all.csv")
    )
    index_parser.add_argument("--camera", type=int, default=6)
    index_parser.add_argument("--beam-width", default=None)
    index_parser.add_argument("--conditions", nargs="*", default=None)
    index_parser.add_argument("--skip-missing", action="store_true")

    extract_parser = subparsers.add_parser(
        "extract", help="Run the frozen V-JEPA 2 encoder."
    )
    extract_parser.add_argument(
        "--pretrain-config",
        type=Path,
        required=True,
        help="Pretrain YAML (app: vjepa). Checkpoint is {folder}/{write_tag}-latest.pt.",
    )
    extract_parser.add_argument("--index-csv", type=Path, required=True)
    extract_parser.add_argument("--raw-output", type=Path, required=True)
    extract_parser.add_argument("--device", default="cuda:0")
    extract_parser.add_argument("--batch-size", type=int, default=None)
    extract_parser.add_argument("--num-segments", type=int, default=None)
    extract_parser.add_argument("--num-views-per-segment", type=int, default=None)
    extract_parser.add_argument("--num-workers", type=int, default=4)
    extract_parser.add_argument("--max-videos", type=int, default=None)

    finalize_parser = subparsers.add_parser(
        "finalize", help="Join manifest metadata and write the embedding table."
    )
    finalize_parser.add_argument("--raw-input", type=Path, required=True)
    finalize_parser.add_argument("--index-csv", type=Path, required=True)
    finalize_parser.add_argument(
        "--manifest-path", type=Path, default=Path("manifests/manifest.jsonl")
    )
    finalize_parser.add_argument("--output-prefix", type=Path, required=True)
    finalize_parser.add_argument("--model-name", default="V-JEPA2 ViT-L/16")

    return parser


def build_filters(args: argparse.Namespace) -> dict:
    filters: dict = {}
    if args.camera is not None and args.camera >= 0:
        filters["camera"] = [int(args.camera)]
    if args.beam_width is not None:
        filters["beam_width"] = [args.beam_width]
    if args.conditions:
        filters["condition"] = list(args.conditions)
    return filters


def run_write_index(args: argparse.Namespace) -> int:
    _add_mbw_to_path()
    from src.utils.vjepa_csv import write_vjepa_index_csv

    csv_path, video_paths = write_vjepa_index_csv(
        args.manifest_path,
        args.index_csv,
        filters=build_filters(args) or None,
        skip_missing=args.skip_missing,
    )
    print(
        f"Wrote {csv_path} with {len(video_paths)} videos "
        f"(labels 0..{len(video_paths) - 1})"
    )
    return 0


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def resolve_extract_settings(args: argparse.Namespace) -> dict:
    """Build encoder / loader settings from a V-JEPA 2 pretrain YAML."""
    cfg = _load_yaml(args.pretrain_config)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    log_cfg = cfg.get("logging") or {}

    folder = Path(cfg.get("folder") or log_cfg.get("folder"))
    write_tag = log_cfg.get("write_tag") or cfg.get("write_tag")
    if write_tag:
        checkpoint_path = folder / f"{write_tag}-latest.pt"
    else:
        checkpoint_path = folder / "latest.pt"

    dataset_fpcs = data_cfg.get("dataset_fpcs")
    if dataset_fpcs:
        frames_per_clip = int(dataset_fpcs[0])
    else:
        frames_per_clip = int(data_cfg.get("num_frames", 16))

    frame_step = data_cfg.get("frame_step", data_cfg.get("sampling_rate", 4))

    settings = {
        "model_name": model_cfg.get("model_name", "vit_large"),
        "checkpoint_path": checkpoint_path,
        "checkpoint_key": "target_encoder",
        "patch_size": int(data_cfg.get("patch_size", 16)),
        "tubelet_size": int(data_cfg.get("tubelet_size", 2)),
        "frames_per_clip": frames_per_clip,
        "frame_step": int(frame_step),
        "resolution": int(data_cfg.get("crop_size", 256)),
        "batch_size": 4,
        "num_segments": 2,
        "num_views_per_segment": 3,
        "uniform_power": bool(model_cfg.get("uniform_power", True)),
        "use_rope": bool(model_cfg.get("use_rope", True)),
        "use_silu": bool(model_cfg.get("use_silu", False)),
        "wide_silu": bool(model_cfg.get("wide_silu", True)),
        "use_activation_checkpointing": bool(
            model_cfg.get("use_activation_checkpointing", False)
        ),
        "dataset_type": data_cfg.get("dataset_type", "VideoDataset"),
    }

    if args.batch_size is not None:
        settings["batch_size"] = int(args.batch_size)
    if args.num_segments is not None:
        settings["num_segments"] = int(args.num_segments)
    if args.num_views_per_segment is not None:
        settings["num_views_per_segment"] = int(args.num_views_per_segment)

    return settings


def run_extract(args: argparse.Namespace) -> int:
    import torch

    from evals.video_classification_frozen.models import init_module
    from evals.video_classification_frozen.eval import make_dataloader
    from src.utils.distributed import init_distributed

    settings = resolve_extract_settings(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    frames_per_clip = int(settings["frames_per_clip"])
    num_segments = int(settings["num_segments"])
    num_views = int(settings["num_views_per_segment"])
    frame_step = int(settings["frame_step"])
    resolution = int(settings["resolution"])
    batch_size = int(settings["batch_size"])
    checkpoint_path = Path(settings["checkpoint_path"])

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_path}")

    init_distributed(rank_and_world_size=(0, 1))

    print(f"Loading frozen encoder from {checkpoint_path}", flush=True)
    encoder = init_module(
        module_name="evals.video_classification_frozen.modelcustom.vit_encoder_multiclip",
        device=device,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=str(checkpoint_path),
        model_kwargs={
            "encoder": {
                "checkpoint_key": settings["checkpoint_key"],
                "model_name": settings["model_name"],
                "patch_size": settings["patch_size"],
                "tubelet_size": settings["tubelet_size"],
                "uniform_power": settings["uniform_power"],
                "use_rope": settings["use_rope"],
                "use_silu": settings["use_silu"],
                "wide_silu": settings["wide_silu"],
                "use_activation_checkpointing": False,
            }
        },
        wrapper_kwargs={
            "max_frames": max(128, frames_per_clip * num_segments),
            "use_pos_embed": False,
        },
    )
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    print(f"Building loader from {args.index_csv}", flush=True)
    loader, _ = make_dataloader(
        root_path=[str(args.index_csv)],
        batch_size=batch_size,
        world_size=1,
        rank=0,
        dataset_type=settings.get("dataset_type", "VideoDataset"),
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_segments=num_segments,
        num_views_per_segment=num_views,
        allow_segment_overlap=True,
        training=False,
        num_workers=args.num_workers,
    )

    feature_blocks: List[np.ndarray] = []
    row_index_blocks: List[np.ndarray] = []
    n_views_averaged = 0
    n_seen = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(loader, start=1):
            clips = [
                [view.to(device, non_blocking=True) for view in segment]
                for segment in data[0]
            ]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            batch_labels = data[1].to(torch.int64)

            outputs = encoder(clips, clip_indices)
            # list over spatial views, each [B, N, D]
            tokens = torch.stack(outputs, dim=0).mean(dim=0)
            n_views_averaged = len(outputs)
            features = tokens.mean(dim=1)  # [B, D]

            feature_blocks.append(features.float().cpu().numpy())
            row_index_blocks.append(batch_labels.cpu().numpy())
            n_seen += int(batch_labels.shape[0])
            if batch_idx % 20 == 0:
                print(f"  encoded {n_seen} videos", flush=True)
            if args.max_videos is not None and n_seen >= args.max_videos:
                break

    if not feature_blocks:
        raise RuntimeError("No videos were encoded; check the index CSV and video paths.")

    X = np.concatenate(feature_blocks, axis=0).astype(np.float32)
    labels = np.concatenate(row_index_blocks, axis=0).astype(np.int64)
    if args.max_videos is not None and X.shape[0] > args.max_videos:
        X = X[: args.max_videos]
        labels = labels[: args.max_videos]

    duplicates = labels.size - np.unique(labels).size
    if duplicates:
        print(
            f"Note: {duplicates} duplicate label(s) in the loader output; "
            "treating CSV column 2 as class labels for embedding metadata.",
            flush=True,
        )

    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.raw_output,
        X=X,
        labels=labels,
        row_index=labels,
        checkpoint=np.asarray(str(checkpoint_path)),
        model_name=np.asarray(str(settings["model_name"])),
        checkpoint_key=np.asarray(str(settings["checkpoint_key"])),
        frames_per_clip=np.asarray(frames_per_clip),
        frame_step=np.asarray(frame_step),
        num_segments=np.asarray(num_segments),
        num_views_per_segment=np.asarray(num_views),
        resolution=np.asarray(resolution),
        n_views_averaged=np.asarray(n_views_averaged),
    )
    print(
        f"Wrote {args.raw_output}: X={X.shape} over {labels.size} videos "
        f"({n_views_averaged} view vectors averaged per video)"
    )
    return 0


def run_finalize(args: argparse.Namespace) -> int:
    _add_mbw_to_path()

    from src.analysis.embedding_io import save_embedding_table
    from src.analysis.manifest_join import join_manifest_by_video_path, load_manifest_frame
    from src.utils.vjepa_csv import read_vjepa_index_csv

    with np.load(args.raw_input, allow_pickle=True) as payload:
        X = np.asarray(payload["X"], dtype=np.float32)
        row_index = np.asarray(payload["row_index"], dtype=np.int64)
        provenance = {
            key: str(payload[key].item() if payload[key].ndim == 0 else payload[key])
            for key in payload.files
            if key not in {"X", "row_index", "labels"}
        }

    video_paths = read_vjepa_index_csv(args.index_csv)
    if row_index.size and row_index.max() >= len(video_paths):
        raise ValueError(
            f"Raw npz references row index {row_index.max()} but the index CSV "
            f"has only {len(video_paths)} rows; the CSV and the extraction disagree."
        )
    encoded_paths = [video_paths[int(i)] for i in row_index]

    manifest = load_manifest_frame(args.manifest_path)
    metadata = join_manifest_by_video_path(manifest, encoded_paths)
    metadata["n_clips"] = int(provenance.get("n_views_averaged", 1) or 1)

    missing = len(video_paths) - len(encoded_paths)
    if missing:
        print(
            f"Note: {missing} of {len(video_paths)} indexed videos produced no "
            "features (skipped by the loader); they are absent from the table.",
            flush=True,
        )

    npz_path, parquet_path = save_embedding_table(
        args.output_prefix,
        X,
        metadata,
        model=args.model_name,
        provenance={
            "source": "vjepa2_self_pretrained",
            "index_csv": str(args.index_csv),
            "manifest_path": str(args.manifest_path),
            "raw_input": str(args.raw_input),
            "n_indexed_videos": str(len(video_paths)),
            **provenance,
        },
    )
    print(
        f"Wrote {npz_path}\nWrote {parquet_path}\n"
        f"{X.shape[0]} videos, dim={X.shape[1]}, "
        f"{metadata['mouse_uid'].nunique()} mice, "
        f"{metadata['condition'].nunique()} conditions"
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "write-index":
        return run_write_index(args)
    if args.mode == "extract":
        return run_extract(args)
    if args.mode == "finalize":
        return run_finalize(args)
    parser.error(f"Unknown mode: {args.mode}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
