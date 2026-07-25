from typing import Dict

import torch


def mse_score(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Mean squared error over all spots and genes."""
    return float(torch.nn.functional.mse_loss(pred.float(), true.float()).item())


def mae_score(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Mean absolute error over all spots and genes."""
    return float(torch.nn.functional.l1_loss(pred.float(), true.float()).item())


def gene_wise_pcc(
    pred: torch.Tensor,
    true: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    Compute mean gene-wise Pearson correlation across spots.

    Args:
        pred: [N_spots, G_genes]
        true: [N_spots, G_genes]

    Returns:
        Mean PCC over genes.
    """
    pred = pred.float()
    true = true.float()

    if pred.size(0) < 3:
        return 0.0

    p = pred - pred.mean(dim=0, keepdim=True)
    t = true - true.mean(dim=0, keepdim=True)

    numerator = (p * t).sum(dim=0)
    denom = torch.sqrt((p ** 2).sum(dim=0) * (t ** 2).sum(dim=0))

    corr = numerator / denom.clamp_min(eps)
    corr = torch.where(denom > eps, corr, torch.zeros_like(corr))
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    return float(corr.mean().item())


def evaluate_one_sample(
    pred: torch.Tensor,
    true: torch.Tensor,
) -> Dict[str, float]:
    """
    Evaluate one sample / slide.

    PCC is computed per gene across spots within this sample.
    """
    return {
        "mse": mse_score(pred, true),
        "mae": mae_score(pred, true),
        "pcc": gene_wise_pcc(pred, true),
    }


def average_metric_dicts(metric_dicts: list[Dict[str, float]]) -> Dict[str, float]:
    """Average a list of metric dictionaries."""
    if len(metric_dicts) == 0:
        return {
            "mse": 0.0,
            "mae": 0.0,
            "pcc": 0.0,
        }

    keys = metric_dicts[0].keys()
    return {
        k: float(sum(m[k] for m in metric_dicts) / len(metric_dicts))
        for k in keys
    }


def evaluate_predictions(
    preds_by_sample: Dict[str, torch.Tensor],
    trues_by_sample: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    sample_metrics = []

    for sample_name in sorted(trues_by_sample.keys()):
        if sample_name not in preds_by_sample:
            raise KeyError(f"Missing prediction for sample: {sample_name}")

        pred = preds_by_sample[sample_name]
        true = trues_by_sample[sample_name]

        sample_metrics.append(evaluate_one_sample(pred, true))

    return average_metric_dicts(sample_metrics)