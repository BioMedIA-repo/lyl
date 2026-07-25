import argparse
import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scprep as scp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce
from tqdm import tqdm

# ============================================================================
# Logging
# ============================================================================

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

setup_logging()


# ============================================================================
# Deep Feature I/O
# ============================================================================

def load_deep_features(pt_path: str) -> Dict:
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Deep feature file not found: {pt_path}")

    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    required = ["spot_ids", "features", "coords"]
    for key in required:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in {pt_path}")

    # Convert to expected types if needed
    if isinstance(data["spot_ids"], np.ndarray):
        data["spot_ids"] = data["spot_ids"].tolist()
    spot_ids = [str(s) for s in data["spot_ids"]]

    features = data["features"]
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float32)

    coords = data["coords"]
    if not isinstance(coords, torch.Tensor):
        coords = torch.tensor(coords, dtype=torch.float32)

    return {
        "sample_name": data.get("sample_name", ""),
        "spot_ids": spot_ids,
        "features": features,
        "coords": coords,
        "encoder": data.get("encoder", "unknown"),
        "feature_dim": data.get("feature_dim", features.size(1)),
        "patch_size": data.get("patch_size", None),
        "resize_size": data.get("resize_size", None),
    }


# ============================================================================
# Count Matrix I/O
# ============================================================================

def read_count_matrix(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()

    if ext in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
        # Parquet may preserve spot_id as the index, or store it as a column.
        if isinstance(df.index, pd.RangeIndex):
            spot_id_cols = [c for c in ["spot_id", "barcode", "barcodes"] if c in df.columns]
            if spot_id_cols:
                df = df.set_index(spot_id_cols[0])
            else:
                first_col = df.columns[0]
                if not pd.api.types.is_numeric_dtype(df[first_col]):
                    df = df.set_index(first_col)
    else:
        sep_map = {".csv": ",", ".tsv": "\t", ".txt": "\t"}
        sep = sep_map.get(ext, "\t")
        df = pd.read_csv(path, sep=sep, index_col=0)

    # If the matrix is transposed (more genes than spots in rows), transpose it
    if df.shape[0] > df.shape[1]:
        logging.info(f"  Transposing count matrix (rows={df.shape[0]} > cols={df.shape[1]})")
        df = df.T

    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df

def normalize_df_expression(df: pd.DataFrame) -> pd.DataFrame:
    # Keep the same row/column order.
    x = df.astype(float)
    x_norm = scp.normalize.library_size_normalize(x)
    x_log = scp.transform.log(x_norm)

    # scprep normally preserves DataFrame metadata, but make this robust.
    if isinstance(x_log, pd.DataFrame):
        return x_log.reindex(index=df.index, columns=df.columns)
    return pd.DataFrame(x_log, index=df.index, columns=df.columns)

# ============================================================================
# Gene List Selection
# ============================================================================

def load_gene_list(
    gene_list_path: Optional[str] = None,
) -> List[str]:
    if gene_list_path is None:
        raise ValueError("--gene_list is required. Provide a .npy or .json gene list file.")

    if not os.path.exists(gene_list_path):
        raise FileNotFoundError(f"Gene list file not found: {gene_list_path}")

    ext = os.path.splitext(gene_list_path)[1].lower()
    if ext == ".npy":
        gene_list = list(np.load(gene_list_path, allow_pickle=True))
    elif ext == ".json":
        with open(gene_list_path, "r") as f:
            gene_list = json.load(f)
    else:
        raise ValueError(f"Unsupported gene list format: {ext}. Use .npy or .json.")

    logging.info(f"Loaded gene list from {gene_list_path}: {len(gene_list)} genes")
    return gene_list


# ============================================================================
# Y Normalization and Spatial Smoothing
# ============================================================================

def _parse_grid_spot_id(spot_id: str) -> Tuple[int, int]:
    """Parse HER2ST-style spot id such as '12x34' into integer grid coordinates."""
    match = re.fullmatch(r"\s*(-?\d+)x(-?\d+)\s*", str(spot_id))
    if match is None:
        raise ValueError(
            f"Cannot parse spot_id={spot_id!r} as HER2ST grid coordinate 'xxy'. "
            "3×3 smoothing requires spot ids like '12x34'."
        )
    return int(match.group(1)), int(match.group(2))


def smooth_exp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spatial smoothing: replace each spot by the mean expression
    of its 3×3 grid neighborhood, including itself. Missing neighbors at
    tissue borders are ignored.

    Args:
        df: Expression DataFrame after library-size normalization + log transform.
            Index must contain HER2ST-style grid spot ids such as '12x34'.

    Returns:
        Smoothed DataFrame with the same index and columns.
    """
    coords = [_parse_grid_spot_id(sid) for sid in df.index]
    coord_to_row = {}
    for row_idx, coord in enumerate(coords):
        if coord in coord_to_row:
            raise ValueError(f"Duplicate grid coordinate {coord} in expression index.")
        coord_to_row[coord] = row_idx

    values = df.to_numpy(dtype=np.float32, copy=False)
    smoothed = np.empty_like(values, dtype=np.float32)

    for row_idx, (gx, gy) in enumerate(coords):
        neighbor_rows = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor_row = coord_to_row.get((gx + dx, gy + dy))
                if neighbor_row is not None:
                    neighbor_rows.append(neighbor_row)
        smoothed[row_idx] = values[neighbor_rows].mean(axis=0)

    return pd.DataFrame(smoothed, index=df.index, columns=df.columns)


def normalize_expression(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full target preprocessing:
        1) library-size normalize selected-gene counts with scprep defaults
        2) log1p transform
        3) 3×3 grid-neighborhood smoothing
    """
    return smooth_exp(normalize_df_expression(df))


# ============================================================================
# MCFG Graph Building
# ============================================================================

def standardize_features(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    return (x - mean) / std


def reduce_morph_features(x: torch.Tensor, out_dim: int = 64) -> torch.Tensor:
    x = standardize_features(x)
    if x.numel() == 0:
        return x
    dim = min(int(out_dim), x.size(0), x.size(1))
    if dim <= 0:
        return x.new_zeros((x.size(0), 0))
    try:
        _, _, v = torch.pca_lowrank(x, q=dim, center=False)
        return x @ v[:, :dim]
    except RuntimeError:
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        return x @ vh[:dim].T


def estimate_bandwidth(v: torch.Tensor, eps: float = 1e-6) -> float:
    if v.numel() == 0:
        return 1.0
    vals = v.detach().reshape(-1).float()
    vals = vals[torch.isfinite(vals) & (vals > eps)]
    if vals.numel() == 0:
        return 1.0
    return float(vals.median().clamp_min(eps).item())


def coalesce_edge_weight_max(edge_index, edge_weight, num_nodes):
    if edge_index.numel() == 0:
        return edge_index, edge_weight
    try:
        return coalesce(edge_index, edge_weight, num_nodes=num_nodes, reduce="max")
    except TypeError:
        key = edge_index[0] * num_nodes + edge_index[1]
        order = torch.argsort(key)
        edge_index = edge_index[:, order]
        edge_weight = edge_weight[order]
        key = key[order]
        kept_edges = []
        kept_weights = []
        start = 0
        while start < key.numel():
            end = start + 1
            while end < key.numel() and key[end] == key[start]:
                end += 1
            kept_edges.append(edge_index[:, start])
            kept_weights.append(edge_weight[start:end].max())
            start = end
        return torch.stack(kept_edges, dim=1), torch.stack(kept_weights)


def _add_self_loops_with_weight(edge_index, edge_weight, num_nodes):
    loops = torch.arange(num_nodes, dtype=torch.long)
    loop_index = torch.stack([loops, loops], dim=0)
    loop_weight = torch.ones(num_nodes, dtype=edge_weight.dtype)
    return coalesce_edge_weight_max(
        torch.cat([edge_index, loop_index], dim=1),
        torch.cat([edge_weight, loop_weight], dim=0),
        num_nodes,
    )


def build_spatial_graph(
    coords: torch.Tensor,
    k,
    weight_mode="uniform",
    sigma_d=0.0,
    make_undirected=True,
    add_self_loops=True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eps = 1e-6
    num_nodes = coords.size(0)
    if weight_mode not in {"uniform", "gaussian"}:
        raise ValueError("weight_mode must be 'uniform' or 'gaussian'.")

    if num_nodes <= 1:
        edge_index = torch.zeros((2, num_nodes), dtype=torch.long)
        edge_weight = torch.ones(num_nodes, dtype=torch.float32)
        return edge_index, torch.zeros((num_nodes, 1)), edge_weight

    k_actual = min(int(k), num_nodes - 1)
    dist = torch.cdist(coords, coords)  # [S, S]
    dist_no_self = dist.clone()
    dist_no_self.fill_diagonal_(float("inf"))
    d_knn, nbr = torch.topk(dist_no_self, k=k_actual, dim=1, largest=False)

    dst = torch.arange(num_nodes).unsqueeze(1).expand(-1, k_actual).reshape(-1)
    src = nbr.reshape(-1)
    edge_index = torch.stack([src, dst], dim=0)
    d = d_knn.reshape(-1)

    if weight_mode == "uniform":
        edge_weight = torch.ones_like(d, dtype=torch.float32)
    else:
        sigma = float(sigma_d) if sigma_d and sigma_d > 0 else estimate_bandwidth(d, eps)
        edge_weight = torch.exp(-(d.float().pow(2)) / (2.0 * sigma * sigma + eps))

    if make_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
        edge_index, edge_weight = coalesce_edge_weight_max(edge_index, edge_weight, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = _add_self_loops_with_weight(edge_index, edge_weight, num_nodes)

    edge_attr = torch.linalg.norm(coords[edge_index[0]] - coords[edge_index[1]], dim=-1, keepdim=True)
    return edge_index.long(), edge_attr.float(), edge_weight.float()


def build_mcfg_graphs(
    coords,
    features,
    k_spatial=12,
    k_low=6,
    k_high=4,
    morph_dim=64,
    sigma_d=0.0,
    sigma_m=0.0,
    make_undirected=True,
):
    eps = 1e-6
    num_nodes = coords.size(0)
    if num_nodes <= 1:
        low = torch.zeros((2, num_nodes), dtype=torch.long)
        low_w = torch.ones(num_nodes, dtype=torch.float32)
        high = torch.zeros((2, 0), dtype=torch.long)
        high_w = torch.zeros(0, dtype=torch.float32)
        return low, low_w, high, high_w, {"sigma_d": 1.0, "sigma_m": 1.0}

    z = reduce_morph_features(features.float(), out_dim=morph_dim)
    k_cand = min(int(k_spatial), num_nodes - 1)
    dist = torch.cdist(coords.float(), coords.float())
    dist_no_self = dist.clone()
    dist_no_self.fill_diagonal_(float("inf"))
    d_cand, nbr = torch.topk(dist_no_self, k=k_cand, dim=1, largest=False)
    dst_all = torch.arange(num_nodes).unsqueeze(1).expand(-1, k_cand)
    src_all = nbr
    morph_dist = torch.linalg.norm(z[src_all] - z[dst_all], dim=-1)

    sd = float(sigma_d) if sigma_d and sigma_d > 0 else estimate_bandwidth(d_cand, eps)
    sm = float(sigma_m) if sigma_m and sigma_m > 0 else estimate_bandwidth(morph_dist, eps)
    s_spa = torch.exp(-(d_cand.float().pow(2)) / (2.0 * sd * sd + eps))
    s_mor = torch.exp(-(morph_dist.float().pow(2)) / (2.0 * sm * sm + eps))
    w_low_all = s_spa * s_mor
    w_high_all = s_spa * (1.0 - s_mor)

    def keep_top(weights, k_keep, add_loop):
        k_actual = min(int(k_keep), k_cand)
        vals, idx = torch.topk(weights, k=k_actual, dim=1, largest=True)
        dst = torch.arange(num_nodes).unsqueeze(1).expand(-1, k_actual).reshape(-1)
        src = torch.gather(src_all, 1, idx).reshape(-1)
        edge_index = torch.stack([src, dst], dim=0).long()
        edge_weight = vals.reshape(-1).float()
        if make_undirected:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
            edge_index, edge_weight = coalesce_edge_weight_max(edge_index, edge_weight, num_nodes)
        if add_loop:
            edge_index, edge_weight = _add_self_loops_with_weight(edge_index, edge_weight, num_nodes)
        return edge_index, edge_weight

    edge_index_low, edge_weight_low = keep_top(w_low_all, k_low, add_loop=True)
    edge_index_high, edge_weight_high = keep_top(w_high_all, k_high, add_loop=False)
    graph_info = {"sigma_d": sd, "sigma_m": sm, "morph_dim": int(z.size(1))}
    return edge_index_low, edge_weight_low, edge_index_high, edge_weight_high, graph_info


def build_graph_by_ablation_mode(
    coords,
    features,
    mode,
    k_spatial=12,
    k_low=6,
    k_high=4,
    morph_dim=64,
    sigma_d=0.0,
    sigma_m=0.0,
    make_undirected=True,
):
    if mode == "spatial_knn":
        ei, _, ew = build_spatial_graph(coords, k_spatial, "uniform", sigma_d, make_undirected, True)
        return ei, ew, ei.clone(), ew.clone(), {"mode": mode}
    if mode == "spatial_gaussian":
        ei, _, ew = build_spatial_graph(coords, k_spatial, "gaussian", sigma_d, make_undirected, True)
        return ei, ew, ei.clone(), ew.clone(), {"mode": mode}

    low_i, low_w, high_i, high_w, info = build_mcfg_graphs(
        coords, features, k_spatial, k_low, k_high, morph_dim, sigma_d, sigma_m, make_undirected
    )
    if mode == "mcfg_low_only":
        return low_i, low_w, low_i.clone(), low_w.clone(), {**info, "mode": mode}
    if mode == "mcfg_high_only":
        low_i, _, low_w = build_spatial_graph(coords, k_spatial, "gaussian", sigma_d, make_undirected, True)
        return low_i, low_w, high_i, high_w, {**info, "mode": mode}
    if mode == "mcfg_full":
        return low_i, low_w, high_i, high_w, {**info, "mode": mode}
    raise ValueError(f"Unsupported graph ablation mode: {mode}")


# ============================================================================
# Build graph for one sample
# ============================================================================

def build_sample_graph(
    sample_name: str,
    feature_path: str,
    df_cnt: pd.DataFrame,
    gene_list: List[str],
    graph_ablation: str,
    k_spatial: int,
    k_low: int,
    k_high: int,
    morph_dim: int,
    sigma_d: float,
    sigma_m: float,
    make_undirected: bool,
    normalize_y: bool,
) -> Data:
    # --- 1. Load deep features ---
    feat_data = load_deep_features(feature_path)
    feat_spots = set(feat_data["spot_ids"])
    logging.info(f"  Feature spots: {len(feat_spots)}")

    # --- 2. Expression ---
    expr_spots_raw = set(df_cnt.index)
    logging.info(f"  Expression spots: {len(expr_spots_raw)}")

    # --- 3. Align spots ---
    common_spots = sorted(feat_spots & expr_spots_raw)
    logging.info(f"  Common spots: {len(common_spots)}")

    if len(common_spots) == 0:
        raise RuntimeError(
            f"No common spots between features ({len(feat_spots)}) "
            f"and expression ({len(expr_spots_raw)}). "
            f"Feature spot examples: {list(feat_spots)[:5]}, "
            f"Expression spot examples: {list(expr_spots_raw)[:5]}"
        )

    # Build index maps
    feat_idx_map = {sid: i for i, sid in enumerate(feat_data["spot_ids"])}
    expr_idx_map = {sid: i for i, sid in enumerate(df_cnt.index.tolist())}

    feat_indices = [feat_idx_map[sid] for sid in common_spots]
    expr_indices = [expr_idx_map[sid] for sid in common_spots]

    # Extract aligned data
    features = feat_data["features"][torch.tensor(feat_indices, dtype=torch.long)]      # [S, D]
    coords = feat_data["coords"][torch.tensor(feat_indices, dtype=torch.long)]           # [S, 2]

    # --- 4. Gene alignment ---
    df_aligned = df_cnt.reindex(index=common_spots, columns=gene_list, fill_value=0.0)
    y_raw = torch.tensor(df_aligned.values, dtype=torch.float32)  # [S, G]

    S = features.size(0)
    D = features.size(1)
    G = y_raw.size(1)

    # --- 5. Normalize y ---
    if normalize_y:
        y_df = normalize_expression(df_aligned)
        y = torch.tensor(y_df.values, dtype=torch.float32)
    else:
        y = y_raw

    # --- 6. Build unified MCFG-style low/high graphs ---
    edge_index_low, edge_weight_low, edge_index_high, edge_weight_high, graph_info = build_graph_by_ablation_mode(
        coords=coords,
        features=features,
        mode=graph_ablation,
        k_spatial=k_spatial,
        k_low=k_low,
        k_high=k_high,
        morph_dim=morph_dim,
        sigma_d=sigma_d,
        sigma_m=sigma_m,
        make_undirected=make_undirected,
    )
    edge_attr = torch.linalg.norm(
        coords[edge_index_low[0]] - coords[edge_index_low[1]], dim=-1, keepdim=True
    )


    # --- 7. Assemble PyG Data ---
    data = Data(
        x=features,              # [S, D] — raw deep patch features
        pos=coords,              # [S, 2]
        edge_index=edge_index_low,
        edge_attr=edge_attr,
        edge_weight=edge_weight_low,
        edge_index_low=edge_index_low,
        edge_weight_low=edge_weight_low,
        edge_index_high=edge_index_high,
        edge_weight_high=edge_weight_high,
        y=y,                     # [S, G]
    )

    # --- Attach metadata ---
    data.spot_id = common_spots
    data.sample_name = sample_name
    data.gene_names = gene_list
    data.feature_type = f"deep_patch_{feat_data['encoder']}"
    data.encoder = feat_data["encoder"]
    data.patch_size = feat_data["patch_size"]
    data.resize_size = feat_data["resize_size"]
    data.y_is_normalized = normalize_y
    data.y_normalization = (
        "scprep: log(scp.normalize.library_size_normalize(counts))"
        if normalize_y else "raw"
    )
    data.y_is_spatially_smoothed = bool(normalize_y)
    data.y_smoothing = "3x3 grid neighborhood mean including self" if normalize_y else "none"
    data.graph_type = "mcfg"
    data.graph_ablation_mode = graph_ablation
    data.graph_info = graph_info

    # --- 8. Log summary ---
    logging.info(f"  Graph summary for {sample_name}:")
    logging.info(f"    Feature dim:    {D}")
    logging.info(f"    Gene num:       {G}")
    logging.info(f"    x mean/std:     {features.mean().item():.4f} / {features.std().item():.4f}")
    logging.info(f"    y mean/std:     {y.mean().item():.4f} / {y.std().item():.4f}")
    logging.info(f"    G_low edges:    {edge_index_low.size(1)}")
    logging.info(f"    G_high edges:   {edge_index_high.size(1)}")
    logging.info(f"    Graph mode:     {graph_ablation}")

    return data


# ============================================================================
# Batch processing
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Build Spot-level Graph from Deep Features"
    )
    # I/O
    parser.add_argument("--feature_dir", required=True,
                        help="Directory containing *_deep_features.pt files")
    parser.add_argument("--cnt_dir", required=True,
                        help="Directory containing spot-level count matrices")
    parser.add_argument("--output_dir", "--out_dir", required=True,
                        help="Directory to save *_spot_graph_deep.pt files")
    # Gene selection
    parser.add_argument("--gene_list", default="/media/npu/Data/lyl/dataset/her2st/genes_her2st.npy",
                        help="Path to .npy or .json gene list file")
    # Graph
    parser.add_argument(
        "--graph_ablation",
        default="mcfg_full",
        choices=[
            "spatial_knn",
            "spatial_gaussian",
            "mcfg_low_only",
            "mcfg_high_only",
            "mcfg_full",
        ],
        help="Graph ablation mode. All modes still output edge_index_low/high.",
    )
    parser.add_argument("--k_spatial", type=int, default=6)
    parser.add_argument("--k_low", type=int, default=6)
    parser.add_argument("--k_high", type=int, default=4)
    parser.add_argument("--morph_dim", type=int, default=64)
    parser.add_argument("--sigma_d", type=float, default=0.0)
    parser.add_argument("--sigma_m", type=float, default=0.0)
    parser.add_argument("--make_undirected", action="store_true", default=True)
    # Normalization
    parser.add_argument("--normalize_y", action="store_true", default=True,
                        help=(
                            "Apply y preprocessing: library-size normalize by "
                            "scprep defaults, log transform, then 3x3 grid smoothing"
                        ))
    # File matching
    parser.add_argument("--count_file_exts", default=".csv,.tsv,.txt,.parquet,.pq",
                        help="Comma-separated count file extensions")

    args = parser.parse_args()

    count_exts = tuple(args.count_file_exts.split(","))

    # --- Find feature files ---
    feature_files = sorted([
        f for f in os.listdir(args.feature_dir)
        if f.endswith("_deep_features.pt")
    ])
    if not feature_files:
        logging.error(f"No *_deep_features.pt files found in {args.feature_dir}")
        sys.exit(1)
    logging.info(f"Found {len(feature_files)} deep feature files")

    # Map sample_name → feature path
    feature_map = {}
    for f in feature_files:
        sample_name = f.replace("_deep_features.pt", "")
        feature_map[sample_name] = os.path.join(args.feature_dir, f)

    # --- Find count files ---
    cnt_files_found = []
    cnt_map = {}
    for f in sorted(os.listdir(args.cnt_dir)):
        if f.endswith(count_exts):
            full_path = os.path.join(args.cnt_dir, f)
            cnt_files_found.append(full_path)
            # Extract sample name: strip extension
            stem = f
            for ext in count_exts:
                if stem.endswith(ext):
                    stem = stem[:-len(ext)]
                    break
            cnt_map[stem] = full_path
    logging.info(f"Found {len(cnt_files_found)} count files")

    # --- Match feature and count samples ---
    common_samples = sorted(set(feature_map.keys()) & set(cnt_map.keys()))
    if not common_samples:
        logging.error("No matching samples between feature and count files!")
        logging.error(f"  Feature samples: {list(feature_map.keys())}")
        logging.error(f"  Count samples:   {list(cnt_map.keys())}")
        # Try case-insensitive matching
        cnt_lower = {k.lower(): v for k, v in cnt_map.items()}
        for feat_name in feature_map:
            if feat_name.lower() in cnt_lower:
                cnt_map[feat_name] = cnt_lower[feat_name.lower()]
                common_samples.append(feat_name)
        common_samples = sorted(set(common_samples))
        if common_samples:
            logging.info(f"  Case-insensitive match found {len(common_samples)} samples.")
        else:
            sys.exit(1)

    logging.info(f"Matched {len(common_samples)} samples:")
    for s in common_samples:
        logging.info(f"  {s}")

    # --- Load gene list ---
    gene_list = load_gene_list(gene_list_path=args.gene_list)
    logging.info(f"Gene list size: {len(gene_list)}")

    # --- Build graphs ---
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    for sample_name in tqdm(common_samples, desc="Building graphs"):
        logging.info(f"\n--- {sample_name} ---")
        try:
            df_cnt = read_count_matrix(cnt_map[sample_name])

            data = build_sample_graph(
                sample_name=sample_name,
                feature_path=feature_map[sample_name],
                df_cnt=df_cnt,
                gene_list=gene_list,
                graph_ablation=args.graph_ablation,
                k_spatial=args.k_spatial,
                k_low=args.k_low,
                k_high=args.k_high,
                morph_dim=args.morph_dim,
                sigma_d=args.sigma_d,
                sigma_m=args.sigma_m,
                make_undirected=args.make_undirected,
                normalize_y=args.normalize_y,
            )

            # Save
            out_path = os.path.join(args.output_dir, f"{sample_name}_spot_graph_deep.pt")
            torch.save(data, out_path)
            file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
            logging.info(f"  Saved to {out_path} ({file_size_mb:.1f} MB)")
            results[sample_name] = out_path

        except Exception as e:
            logging.error(f"  Failed to build graph for {sample_name}: {e}", exc_info=True)

    logging.info(f"\n{'=' * 60}")
    logging.info(f"Done! Built {len(results)}/{len(common_samples)} graphs.")
    for name, path in results.items():
        logging.info(f"  {name}: {path}")
    logging.info(f"  Gene list: {len(gene_list)} genes")
    logging.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
