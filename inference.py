from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from sfa_gsdiff_model import SFAGSDiffGNN
from spot_metrics import evaluate_one_sample


LOGGER = logging.getLogger("mosaic_inference")


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def safe_torch_load(path: Path, *, map_location: str | torch.device = "cpu") -> Any:
    """Load checkpoints and PyG objects across recent PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def as_plain_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"Expected mapping-like checkpoint args, got {type(value)!r}.")


def unwrap_singleton(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def sanitise_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._")
    return cleaned or "sample"


def resolve_device(device_text: str) -> torch.device:
    if device_text.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA was requested but is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(device_text)


# ---------------------------------------------------------------------------
# Gene names and graph discovery
# ---------------------------------------------------------------------------

def load_gene_list(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Gene-list file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        values = np.load(path, allow_pickle=True).tolist()
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            values = json.load(f)
    else:
        with path.open("r", encoding="utf-8") as f:
            values = [line.strip() for line in f if line.strip()]

    if isinstance(values, dict):
        for key in ("genes", "gene_names", "selected_genes"):
            if key in values:
                values = values[key]
                break
        else:
            raise ValueError(
                f"JSON gene list must contain one of: genes, gene_names, selected_genes."
            )

    genes = [str(x) for x in values]
    if not genes:
        raise ValueError(f"Gene-list file is empty: {path}")
    return genes


def discover_graph_files(
    graph_dir: Optional[Path],
    graph_files: Optional[Sequence[Path]],
    file_list: Optional[Path],
) -> List[Path]:
    paths: List[Path] = []

    if graph_files:
        paths.extend(Path(p) for p in graph_files)

    if file_list is not None:
        if not file_list.exists():
            raise FileNotFoundError(f"Graph file list not found: {file_list}")
        with file_list.open("r", encoding="utf-8") as f:
            listed = [Path(line.strip()) for line in f if line.strip()]
        for item in listed:
            paths.append(item if item.is_absolute() or graph_dir is None else graph_dir / item)

    if graph_dir is not None and not graph_files and file_list is None:
        if not graph_dir.exists():
            raise FileNotFoundError(f"Graph directory not found: {graph_dir}")
        paths.extend(sorted(graph_dir.glob("*.pt")))

    unique_paths: List[Path] = []
    seen = set()
    for path in paths:
        path = path.expanduser().resolve()
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)

    if not unique_paths:
        raise ValueError(
            "No graph files were found. Provide --graph_dir, --graph_files, or --file_list."
        )

    missing = [str(path) for path in unique_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing graph files:\n" + "\n".join(missing))

    return unique_paths


REQUIRED_GRAPH_ATTRS = (
    "x",
    "pos",
    "edge_index_low",
    "edge_weight_low",
    "edge_index_high",
    "edge_weight_high",
)


def validate_graph(data: Any, path: Path) -> None:
    missing = [name for name in REQUIRED_GRAPH_ATTRS if not hasattr(data, name)]
    if missing:
        raise ValueError(
            f"{path} is missing required graph attributes: {', '.join(missing)}"
        )

    if data.x.ndim != 2:
        raise ValueError(f"{path}: data.x must have shape [N, D], got {tuple(data.x.shape)}.")
    if data.pos.ndim != 2 or data.pos.size(0) != data.x.size(0):
        raise ValueError(
            f"{path}: data.pos must have shape [N, 2] with the same N as data.x."
        )

    for edge_name, weight_name in (
        ("edge_index_low", "edge_weight_low"),
        ("edge_index_high", "edge_weight_high"),
    ):
        edge_index = getattr(data, edge_name)
        edge_weight = getattr(data, weight_name)
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(f"{path}: {edge_name} must have shape [2, E].")
        if edge_weight.numel() != edge_index.size(1):
            raise ValueError(
                f"{path}: {weight_name} length does not match {edge_name}."
            )


# ---------------------------------------------------------------------------
# Feature standardisation
# ---------------------------------------------------------------------------

def normalise_scaler_dict(obj: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(obj, Mapping) and "feature_scaler" in obj:
        obj = obj["feature_scaler"]
    if not isinstance(obj, Mapping):
        raise TypeError("Feature scaler must be a dictionary-like object.")

    mean = obj.get("x_mean", obj.get("mean"))
    std = obj.get("x_std", obj.get("std"))
    if mean is None or std is None:
        raise KeyError("Feature scaler must contain x_mean/x_std or mean/std.")

    mean_t = torch.as_tensor(mean, dtype=torch.float32).flatten()
    std_t = torch.as_tensor(std, dtype=torch.float32).flatten().clamp_min(1e-6)
    if mean_t.shape != std_t.shape:
        raise ValueError("Feature-scaler mean and standard deviation have different shapes.")
    return mean_t, std_t


def fit_feature_scaler(graph_paths: Sequence[Path]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit mean/std over all spots in the supplied training graphs."""
    sum_x: Optional[torch.Tensor] = None
    sum_x2: Optional[torch.Tensor] = None
    count = 0

    for path in graph_paths:
        data = safe_torch_load(path)
        validate_graph(data, path)
        x = data.x.detach().cpu().float()
        finite = torch.isfinite(x)
        x_clean = torch.where(finite, x, torch.zeros_like(x))

        feature_count = finite.sum(dim=0)
        feature_sum = x_clean.sum(dim=0)
        feature_sum2 = x_clean.square().sum(dim=0)

        if sum_x is None:
            sum_x = feature_sum
            sum_x2 = feature_sum2
            counts = feature_count
        else:
            if x.size(1) != sum_x.numel():
                raise ValueError(
                    f"Feature dimension mismatch while fitting scaler at {path}."
                )
            sum_x += feature_sum
            sum_x2 += feature_sum2
            counts += feature_count
        count += x.size(0)

    if sum_x is None or sum_x2 is None:
        raise ValueError("No graphs were supplied for fitting the feature scaler.")

    counts = counts.clamp_min(1)
    mean = sum_x / counts
    var = (sum_x2 / counts) - mean.square()
    std = var.clamp_min(0).sqrt().clamp_min(1e-6)

    LOGGER.info(
        "Fitted feature scaler from %d graphs and %d total spots.",
        len(graph_paths),
        count,
    )
    return mean, std


def apply_feature_scaler(
    data: Any,
    mean: torch.Tensor,
    std: torch.Tensor,
    path: Path,
) -> Any:
    if data.x.size(1) != mean.numel():
        raise ValueError(
            f"{path}: graph feature dimension {data.x.size(1)} does not match "
            f"scaler dimension {mean.numel()}."
        )

    x = data.x.float()
    mean = mean.to(device=x.device, dtype=x.dtype)
    std = std.to(device=x.device, dtype=x.dtype)
    finite = torch.isfinite(x)
    x = torch.where(finite, x, mean)
    data.x = (x - mean) / std
    return data


# ---------------------------------------------------------------------------
# Checkpoint and model reconstruction
# ---------------------------------------------------------------------------

def strip_state_dict_prefixes(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    result = dict(state_dict)
    for prefix in ("module.", "model."):
        if result and all(key.startswith(prefix) for key in result):
            result = {key[len(prefix):]: value for key, value in result.items()}
    return result


def infer_num_genes(
    state_dict: Mapping[str, torch.Tensor],
    graph: Any,
    gene_names: Optional[Sequence[str]],
) -> int:
    candidates: List[int] = []

    if gene_names is not None:
        candidates.append(len(gene_names))

    graph_gene_names = unwrap_singleton(getattr(graph, "gene_names", None))
    if graph_gene_names is not None:
        try:
            candidates.append(len(graph_gene_names))
        except TypeError:
            pass

    if hasattr(graph, "y") and graph.y is not None and graph.y.ndim == 2:
        candidates.append(int(graph.y.size(1)))

    for key in (
        "direct_head.net.3.bias",
        "direct_head.net.3.weight",
        "diffusion_head.gene_bias",
        "scale_attention.gene_embedding.weight",
    ):
        tensor = state_dict.get(key)
        if tensor is not None:
            candidates.append(int(tensor.size(0)))

    if not candidates:
        raise ValueError(
            "Could not infer the number of genes. Supply --gene_list or use a graph "
            "containing gene_names/y."
        )

    if len(set(candidates)) != 1:
        raise ValueError(f"Inconsistent gene dimensions were detected: {candidates}")
    return candidates[0]


def checkpoint_arg(args: Mapping[str, Any], key: str, default: Any) -> Any:
    value = args.get(key, default)
    return default if value is None else value


def build_model(
    checkpoint_args: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
    graph: Any,
    gene_names: Optional[Sequence[str]],
    device: torch.device,
) -> SFAGSDiffGNN:
    in_dim = int(graph.x.size(1))
    num_genes = infer_num_genes(state_dict, graph, gene_names)

    kwargs = dict(
        in_dim=in_dim,
        num_genes=num_genes,
        hidden_dim=int(checkpoint_arg(checkpoint_args, "hidden_dim", 256)),
        num_layers=int(checkpoint_arg(checkpoint_args, "num_layers", 1)),
        dropout=float(checkpoint_arg(checkpoint_args, "dropout", 0.5)),
        num_diffusion_steps=int(
            checkpoint_arg(checkpoint_args, "num_diffusion_steps", 7)
        ),
        diffusion_residual=float(
            checkpoint_arg(checkpoint_args, "diffusion_residual", 0.0)
        ),
        gene_embed_dim=int(checkpoint_arg(checkpoint_args, "gene_embed_dim", 64)),
        scale_attn_hidden_dim=int(
            checkpoint_arg(checkpoint_args, "scale_attn_hidden_dim", 128)
        ),
        scale_temperature=float(
            checkpoint_arg(checkpoint_args, "scale_temperature", 1.0)
        ),
        pred_dim=checkpoint_arg(checkpoint_args, "pred_dim", None),
        alpha_mode=str(checkpoint_arg(checkpoint_args, "alpha_mode", "softplus")),
        alpha_init=float(checkpoint_arg(checkpoint_args, "alpha_init", 0.05)),
        residual_alpha=float(
            checkpoint_arg(checkpoint_args, "residual_alpha", 0.1)
        ),
        use_direct_branch=bool(
            checkpoint_arg(checkpoint_args, "use_direct_branch", True)
        ),
        diffusion_mode=str(
            checkpoint_arg(checkpoint_args, "diffusion_mode", "gene_specific")
        ),
        use_position=bool(checkpoint_arg(checkpoint_args, "use_position", True)),
        pos_rbf_dim=int(checkpoint_arg(checkpoint_args, "pos_rbf_dim", 8)),
        sfa_aggregation=str(
            checkpoint_arg(checkpoint_args, "sfa_aggregation", "attention")
        ),
        gate_type=str(checkpoint_arg(checkpoint_args, "gate_type", "scalar")),
        sfa_mode=str(checkpoint_arg(checkpoint_args, "sfa_mode", "full")),
        conv_type=str(checkpoint_arg(checkpoint_args, "conv_type", "gatv2")),
        heads=int(checkpoint_arg(checkpoint_args, "heads", 4)),
    )

    model = SFAGSDiffGNN(**kwargs).to(device)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint does not match the reconstructed model.\n"
            f"Missing keys: {incompatible.missing_keys}\n"
            f"Unexpected keys: {incompatible.unexpected_keys}"
        )

    model.eval()
    LOGGER.info(
        "Loaded MOSAIC: in_dim=%d, genes=%d, layers=%d, diffusion_steps=%d.",
        in_dim,
        num_genes,
        kwargs["num_layers"],
        kwargs["num_diffusion_steps"],
    )
    return model


def resolve_gene_names(
    explicit_gene_names: Optional[List[str]],
    graph: Any,
    num_genes: int,
) -> List[str]:
    if explicit_gene_names is not None:
        genes = explicit_gene_names
    else:
        graph_genes = unwrap_singleton(getattr(graph, "gene_names", None))
        genes = [str(x) for x in graph_genes] if graph_genes is not None else []

    if not genes:
        genes = [f"gene_{i}" for i in range(num_genes)]

    if len(genes) != num_genes:
        raise ValueError(
            f"Gene-name count ({len(genes)}) does not match model output "
            f"dimension ({num_genes})."
        )
    return genes


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def sample_name_from_graph(data: Any, path: Path) -> str:
    value = unwrap_singleton(getattr(data, "sample_name", None))
    if value is None or str(value).strip() == "":
        return path.stem
    return str(value)


def spot_ids_from_graph(data: Any) -> List[str]:
    value = unwrap_singleton(getattr(data, "spot_id", None))
    if value is None:
        value = unwrap_singleton(getattr(data, "spot_ids", None))
    if value is None:
        return [str(i) for i in range(data.x.size(0))]
    ids = [str(x) for x in value]
    if len(ids) != data.x.size(0):
        raise ValueError(
            f"Spot-ID count ({len(ids)}) does not match number of spots "
            f"({data.x.size(0)})."
        )
    return ids


def mean_metric_dicts(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


@torch.inference_mode()
def run_inference(
    model: SFAGSDiffGNN,
    graph_paths: Sequence[Path],
    device: torch.device,
    gene_names: List[str],
    output_dir: Path,
    scaler: Optional[Tuple[torch.Tensor, torch.Tensor]],
    save_csv: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []

    for path in graph_paths:
        data = safe_torch_load(path)
        validate_graph(data, path)

        if scaler is not None:
            data = apply_feature_scaler(data, scaler[0], scaler[1], path)

        data = data.to(device)
        outputs = model(data, return_aux=True)

        sample_name = sample_name_from_graph(data, path)
        spot_ids = spot_ids_from_graph(data)
        y_pred = outputs["y_pred"].detach().cpu()
        y_base = outputs["y_base"].detach().cpu()
        y_diff = outputs["y_diff"].detach().cpu()
        pos = data.pos.detach().cpu()

        row: Dict[str, Any] = {
            "sample_name": sample_name,
            "spot_id": spot_ids,
            "gene_names": gene_names,
            "pos": pos,
            "y_pred": y_pred,
            "y_base": y_base,
            "y_diff": y_diff,
        }

        if hasattr(data, "y") and data.y is not None:
            y_true = data.y.detach().cpu()
            if y_true.shape == y_pred.shape:
                row["y_true"] = y_true
                metrics = evaluate_one_sample(y_pred, y_true)
                metric_rows.append({"sample_name": sample_name, **metrics})
                LOGGER.info(
                    "%s | PCC=%.4f MSE=%.4f MAE=%.4f",
                    sample_name,
                    metrics["pcc"],
                    metrics["mse"],
                    metrics["mae"],
                )
            else:
                LOGGER.warning(
                    "%s: y shape %s differs from prediction shape %s; metrics skipped.",
                    sample_name,
                    tuple(y_true.shape),
                    tuple(y_pred.shape),
                )

        prediction_rows.append(row)

        if save_csv:
            csv_path = output_dir / f"{sanitise_filename(sample_name)}_predictions.csv"
            frame = pd.DataFrame(y_pred.numpy(), index=spot_ids, columns=gene_names)
            frame.index.name = "spot_id"
            frame.to_csv(csv_path)
            LOGGER.info("Saved %s", csv_path)

    prediction_path = output_dir / "predictions.pt"
    torch.save(prediction_rows, prediction_path)
    LOGGER.info("Saved %s", prediction_path)

    pi = model.get_scale_attention(device)
    expected_scale = model.get_expected_scale(pi)
    scale_path = output_dir / "gene_scale_attention.pt"
    torch.save(
        {
            "gene_names": gene_names,
            "pi": None if pi is None else pi.detach().cpu(),
            "expected_scale": (
                None if expected_scale is None else expected_scale.detach().cpu()
            ),
            "num_diffusion_steps": model.num_diffusion_steps,
        },
        scale_path,
    )
    LOGGER.info("Saved %s", scale_path)

    if metric_rows:
        summary = mean_metric_dicts(
            [
                {key: float(row[key]) for key in ("mse", "mae", "pcc")}
                for row in metric_rows
            ]
        )
        metrics_payload = {"per_sample": metric_rows, "mean": summary}
        metrics_path = output_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)
        LOGGER.info(
            "Mean metrics | PCC=%.4f MSE=%.4f MAE=%.4f",
            summary["pcc"],
            summary["mse"],
            summary["mae"],
        )
        LOGGER.info("Saved %s", metrics_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MOSAIC inference on precomputed spot graphs."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph_dir", type=Path, default=None)
    parser.add_argument("--graph_files", type=Path, nargs="+", default=None)
    parser.add_argument(
        "--file_list",
        type=Path,
        default=None,
        help="Text file containing graph paths or names, one per line.",
    )
    parser.add_argument("--gene_list", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save_csv", action="store_true")

    scaler_group = parser.add_mutually_exclusive_group()
    scaler_group.add_argument(
        "--feature_scaler",
        type=Path,
        default=None,
        help="Saved dictionary containing x_mean and x_std.",
    )
    scaler_group.add_argument(
        "--fit_scaler_graph_dir",
        type=Path,
        default=None,
        help=(
            "Recompute feature statistics from the exact training graphs used "
            "for this checkpoint."
        ),
    )
    parser.add_argument(
        "--fit_scaler_file_list",
        type=Path,
        default=None,
        help="Optional file list restricting --fit_scaler_graph_dir.",
    )
    parser.add_argument(
        "--allow_unscaled",
        action="store_true",
        help=(
            "Permit inference without standardisation even when the checkpoint "
            "was trained with --standardize_x. This is not recommended."
        ),
    )
    return parser


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = resolve_device(args.device)
    graph_paths = discover_graph_files(args.graph_dir, args.graph_files, args.file_list)
    first_graph = safe_torch_load(graph_paths[0])
    validate_graph(first_graph, graph_paths[0])

    checkpoint = safe_torch_load(args.checkpoint)
    if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        checkpoint_args = as_plain_dict(checkpoint.get("args"))
    elif isinstance(checkpoint, Mapping) and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        state_dict = checkpoint
        checkpoint_args = {}
    else:
        raise ValueError(
            "Checkpoint must be a raw state_dict or contain model_state_dict."
        )

    state_dict = strip_state_dict_prefixes(state_dict)
    explicit_gene_names = load_gene_list(args.gene_list)

    model = build_model(
        checkpoint_args=checkpoint_args,
        state_dict=state_dict,
        graph=first_graph,
        gene_names=explicit_gene_names,
        device=device,
    )
    gene_names = resolve_gene_names(
        explicit_gene_names,
        first_graph,
        model.num_genes,
    )

    scaler: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    if args.feature_scaler is not None:
        scaler_obj = safe_torch_load(args.feature_scaler)
        scaler = normalise_scaler_dict(scaler_obj)
        LOGGER.info("Loaded feature scaler from %s", args.feature_scaler)

    elif isinstance(checkpoint, Mapping) and "feature_scaler" in checkpoint:
        scaler = normalise_scaler_dict(checkpoint["feature_scaler"])
        LOGGER.info("Loaded feature scaler embedded in checkpoint.")

    elif args.fit_scaler_graph_dir is not None:
        scaler_paths = discover_graph_files(
            args.fit_scaler_graph_dir,
            graph_files=None,
            file_list=args.fit_scaler_file_list,
        )
        scaler = fit_feature_scaler(scaler_paths)

        scaler_path = args.output_dir / "feature_scaler_recomputed.pt"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"x_mean": scaler[0], "x_std": scaler[1]}, scaler_path)
        LOGGER.info("Saved recomputed feature scaler to %s", scaler_path)

    checkpoint_used_standardisation = bool(
        checkpoint_arg(checkpoint_args, "standardize_x", False)
    )
    if scaler is None and checkpoint_used_standardisation and not args.allow_unscaled:
        raise RuntimeError(
            "This checkpoint was trained with feature standardisation, but no "
            "training-fold scaler was supplied. Pass --feature_scaler, use "
            "--fit_scaler_graph_dir with the exact fold training graphs, or "
            "explicitly pass --allow_unscaled."
        )
    if scaler is None:
        LOGGER.warning("Inference will use the graph features without standardisation.")

    run_inference(
        model=model,
        graph_paths=graph_paths,
        device=device,
        gene_names=gene_names,
        output_dir=args.output_dir,
        scaler=scaler,
        save_csv=args.save_csv,
    )


if __name__ == "__main__":
    main()
