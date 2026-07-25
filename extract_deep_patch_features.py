import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    resnet18,
    resnet50,
    resnet101,
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
)
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

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
# Image I/O
# ============================================================================

def read_image_robust(
    path: str,
    target_mode: str = "RGB",
) -> Image.Image:
    ext = os.path.splitext(path)[1].lower()
    try:
        img = Image.open(path)
        # Force load pixels (deferred loading in some TIFFs)
        img.load()
        if img.mode != target_mode:
            if img.mode == "RGBA":
                # Paste onto white background
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            elif img.mode == "L":
                img = img.convert("RGB")
            elif img.mode == "I;16" or img.mode == "I":
                # 16-bit grayscale → 8-bit → RGB
                img = ImageOps.autocontrast(img.convert("L"))
                img = img.convert("RGB")
            else:
                img = img.convert(target_mode)
        logging.info(f"Loaded image {os.path.basename(path)}: {img.size}, mode={img.mode}")
        return img
    except Exception as e_pil:
        logging.warning(f"PIL failed to open {path}: {e_pil}")

    # --- Fallback: tifffile ---
    if ext in (".tif", ".tiff"):
        try:
            import tifffile
            arr = tifffile.imread(path)  # [H, W] or [H, W, C]
            if arr.ndim == 2:
                img = Image.fromarray(arr).convert("L")
                img = img.convert(target_mode)
            elif arr.ndim == 3 and arr.shape[2] >= 3:
                img = Image.fromarray(arr[:, :, :3])
                if img.mode != target_mode:
                    img = img.convert(target_mode)
            else:
                img = Image.fromarray(arr[:, :, 0]).convert(target_mode)
            logging.info(f"Loaded via tifffile {os.path.basename(path)}: {img.size}")
            return img
        except ImportError:
            logging.error("tifffile not installed. Try: pip install tifffile")
        except Exception as e_tif:
            logging.error(f"tifffile also failed: {e_tif}")

    # --- Fallback: OpenSlide (for WSI) ---
    if ext in (".tif", ".tiff", ".svs", ".ndpi"):
        try:
            import openslide
            slide = openslide.OpenSlide(path)
            # Read at level 0 (full resolution) – may be huge; read a downsample
            # For Phase 1, read thumbnail as a guide if full image too large
            level_dims = slide.level_dimensions
            logging.info(f"OpenSlide levels: {level_dims}")
            # Read at highest resolution level 0
            region = slide.read_region((0, 0), 0, level_dims[0])
            img = region.convert(target_mode)
            logging.info(f"Loaded via OpenSlide: {img.size}")
            slide.close()
            return img
        except ImportError:
            logging.warning("openslide-python not installed.")
        except Exception as e_os:
            logging.error(f"OpenSlide also failed: {e_os}")

    raise RuntimeError(f"Could not read image: {path}")


# ============================================================================
# Spot Coordinate I/O
# ============================================================================

def read_spot_coords(
    path: str,
    spot_id_col: str = "spot_id",
    x_col: str = "x",
    y_col: str = "y",
    coord_scale: float = 1.0,
    coord_offset_x: float = 0.0,
    coord_offset_y: float = 0.0,
) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    sep = "\t" if ext == ".tsv" else ","

    df = pd.read_csv(path, sep=sep, index_col=None)

    # --- Validate coordinate columns (required) ---
    missing = []
    for col in [x_col, y_col]:
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise KeyError(f"Missing coordinate columns in {path}: {missing}. "
                       f"Available: {list(df.columns)}")

    # --- Resolve spot_id ---
    if spot_id_col in df.columns:
        spot_ids = df[spot_id_col].astype(str).tolist()
    elif not isinstance(df.index, pd.RangeIndex) and df.index.name is not None:
        # Index has meaningful values (e.g. barcodes)
        spot_ids = df.index.astype(str).tolist()
        logging.info(f"spot_id_col '{spot_id_col}' not found; using DataFrame index "
                     f"(name='{df.index.name}') as spot_id")
    elif "x" in df.columns and "y" in df.columns:
        # Construct "{x}x{y}" from grid coordinates (HER2st convention)
        grid_x = df["x"].astype(int).astype(str)
        grid_y = df["y"].astype(int).astype(str)
        spot_ids = (grid_x + "x" + grid_y).tolist()
        logging.info(f"spot_id_col '{spot_id_col}' not found; "
                     f"constructed '{{x}}x{{y}}' from grid x,y columns as spot_id")
    else:
        # Last resort: use row number
        spot_ids = [str(i) for i in range(len(df))]
        logging.info(f"spot_id_col '{spot_id_col}' not found; using row index as spot_id")

    # --- Extract coordinates ---
    x_raw = pd.to_numeric(df[x_col], errors="coerce")
    y_raw = pd.to_numeric(df[y_col], errors="coerce")

    # Drop missing coords
    valid = x_raw.notna() & y_raw.notna()
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        logging.warning(f"Dropping {n_dropped} spots with missing coordinates in {path}")

    spot_ids = [sid for sid, v in zip(spot_ids, valid) if v]
    x_raw = x_raw[valid].values
    y_raw = y_raw[valid].values

    # Apply scale and offset
    px = x_raw * coord_scale + coord_offset_x
    py = y_raw * coord_scale + coord_offset_y

    out = pd.DataFrame({
        "spot_id": spot_ids,
        "px": px.astype(float),
        "py": py.astype(float),
    })

    logging.info(f"Read {len(out)} spots from {path} "
                 f"(x∈[{px.min():.0f}, {px.max():.0f}], "
                 f"y∈[{py.min():.0f}, {py.max():.0f}])")
    return out


# ============================================================================
# Patch Cropping
# ============================================================================

def crop_patch(
    img: Image.Image,
    cx: float,
    cy: float,
    patch_size: int,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    left = int(round(cx - patch_size / 2))
    top = int(round(cy - patch_size / 2))
    right = left + patch_size
    bottom = top + patch_size

    # Check if fully within image
    W, H = img.size
    if left >= 0 and top >= 0 and right <= W and bottom <= H:
        return img.crop((left, top, right, bottom))

    # Need padding
    # Crop the overlapping region first
    crop_left = max(0, left)
    crop_top = max(0, top)
    crop_right = min(W, right)
    crop_bottom = min(H, bottom)

    cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom)) if \
        (crop_right > crop_left and crop_bottom > crop_top) else \
        Image.new("RGB", (0, 0), bg_color)

    # Pad back to patch_size × patch_size
    pad_left = crop_left - left
    pad_top = crop_top - top
    result = Image.new("RGB", (patch_size, patch_size), bg_color)
    result.paste(cropped, (pad_left, pad_top))

    return result


# ============================================================================
# Dataset for batch inference
# ============================================================================

class PatchDataset(Dataset):

    def __init__(
        self,
        patches: List[Image.Image],
        resize_size: int = 224,
    ):
        self.patches = patches
        self.transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        return self.transform(self.patches[idx])


# ============================================================================
# Feature Extractor
# ============================================================================

def build_encoder(
    encoder_name: str = "resnet50",
    pretrained: str = "imagenet",
    uni_ckpt: Optional[str] = None,
) -> Tuple[nn.Module, int]:
    # --- UNI (ViT-Large pathology foundation model) ---
    if encoder_name == "uni":
        if uni_ckpt is None:
            raise ValueError("--uni_ckpt is required when --encoder uni")
        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        state_dict = torch.load(uni_ckpt, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        feature_dim = 1024
        model.eval()
        return model, feature_dim

    # --- ResNet family ---
    weights_map = {
        "resnet18": ResNet18_Weights.IMAGENET1K_V1,
        "resnet50": ResNet50_Weights.IMAGENET1K_V2,
        "resnet101": ResNet101_Weights.IMAGENET1K_V2,
    }

    dim_map = {
        "resnet18": 512,
        "resnet50": 2048,
        "resnet101": 2048,
    }

    if encoder_name not in weights_map:
        raise ValueError(f"Unsupported encoder: {encoder_name}. "
                         f"Choose from {list(weights_map.keys())} or 'uni'.")

    weights = weights_map[encoder_name] if pretrained else None

    if encoder_name == "resnet18":
        model = resnet18(weights=weights)
    elif encoder_name == "resnet50":
        model = resnet50(weights=weights)
    elif encoder_name == "resnet101":
        model = resnet101(weights=weights)
    else:
        raise ValueError(encoder_name)

    # Remove final fc layer — keep the feature extractor up to avgpool
    model.fc = nn.Identity()

    feature_dim = dim_map[encoder_name]
    model.eval()

    return model, feature_dim


@torch.no_grad()
def extract_features_batch(
    encoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> torch.Tensor:
    encoder.to(device)
    encoder.eval()
    all_features = []

    amp_ctx = torch.cuda.amp.autocast if (use_amp and device.type == "cuda") else None

    for batch in tqdm(dataloader, desc="Extracting features"):
        batch = batch.to(device)
        if amp_ctx:
            with amp_ctx():
                feats = encoder(batch)
        else:
            feats = encoder(batch)
        all_features.append(feats.cpu())

    features = torch.cat(all_features, dim=0)  # [N, D]
    return features


# ============================================================================
# Sanity check
# ============================================================================

def check_features(features: torch.Tensor, sample_name: str) -> None:
    """Check features for NaN/Inf and print statistics."""
    n_nan = torch.isnan(features).sum().item()
    n_inf = torch.isinf(features).sum().item()

    logging.info(f"[{sample_name}] Feature stats:")
    logging.info(f"  Shape: {list(features.shape)}")
    logging.info(f"  Mean:  {features.mean().item():.6f}")
    logging.info(f"  Std:   {features.std().item():.6f}")
    logging.info(f"  Min:   {features.min().item():.6f}")
    logging.info(f"  Max:   {features.max().item():.6f}")
    logging.info(f"  Per-dim std mean: {features.std(dim=0).mean().item():.6f}")
    logging.info(f"  NaN count: {n_nan}")
    logging.info(f"  Inf count: {n_inf}")

    if n_nan > 0 or n_inf > 0:
        raise ValueError(
            f"[{sample_name}] Features contain {n_nan} NaN and {n_inf} Inf values! "
            f"Check input patches."
        )


# ============================================================================
# Main processing for one sample
# ============================================================================

def process_sample(
    sample_name: str,
    image_path: str,
    coord_path: str,
    output_dir: str,
    patch_dir: Optional[str],
    encoder: nn.Module,
    feature_dim: int,
    encoder_name: str,
    patch_size: int,
    resize_size: int,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    save_patches: bool,
    coord_scale: float,
    coord_offset_x: float,
    coord_offset_y: float,
    spot_id_col: str,
    x_col: str,
    y_col: str,
) -> Optional[str]:
    logging.info(f"\n{'=' * 60}")
    logging.info(f"Processing sample: {sample_name}")
    logging.info(f"{'=' * 60}")

    # --- 1. Read image ---
    logging.info(f"Reading image: {image_path}")
    img = read_image_robust(image_path)
    W, H = img.size
    logging.info(f"  Image size: {W} × {H}")

    # --- 2. Read coordinates ---
    logging.info(f"Reading coordinates: {coord_path}")
    df_coords = read_spot_coords(
        coord_path,
        spot_id_col=spot_id_col,
        x_col=x_col,
        y_col=y_col,
        coord_scale=coord_scale,
        coord_offset_x=coord_offset_x,
        coord_offset_y=coord_offset_y,
    )
    n_spots = len(df_coords)
    logging.info(f"  Total spots: {n_spots}")

    if n_spots == 0:
        logging.warning(f"No valid spots for {sample_name}, skipping.")
        return None

    # --- 3. Crop patches ---
    logging.info(f"Cropping patches ({patch_size}×{patch_size})...")
    patches = []
    spot_ids = []
    coords_list = []

    for _, row in tqdm(df_coords.iterrows(), total=n_spots, desc="Cropping"):
        spot_id = row["spot_id"]
        cx, cy = float(row["px"]), float(row["py"])

        patch = crop_patch(img, cx, cy, patch_size)

        patches.append(patch)
        spot_ids.append(spot_id)
        coords_list.append([cx, cy])

    # --- 4. Save patches as tensor ---
    logging.info(f"Saving patches as tensor...")
    patches_tensor = torch.stack([
        transforms.ToTensor()(p) for p in patches
    ])  # [N, 3, patch_size, patch_size]
    coords_tensor = torch.tensor(coords_list, dtype=torch.float32)
    os.makedirs(output_dir, exist_ok=True)
    patch_tensor_path = os.path.join(output_dir, f"{sample_name}_patches.pt")
    torch.save({"patches": patches_tensor, "spot_ids": spot_ids, "coords": coords_tensor, "sample_name": sample_name,}, patch_tensor_path)
    logging.info(f"  Saved patches tensor ({list(patches_tensor.shape)}) to {patch_tensor_path}")

    # --- 5. Save patches as PNGs if requested ---
    patch_paths = [""] * n_spots
    if save_patches and patch_dir:
        sample_patch_dir = os.path.join(patch_dir, sample_name)
        os.makedirs(sample_patch_dir, exist_ok=True)
        for i, patch in enumerate(tqdm(patches, desc="Saving patches")):
            p = os.path.join(sample_patch_dir, f"{spot_ids[i]}.png")
            patch.save(p)
            patch_paths[i] = p
        logging.info(f"  Saved {n_spots} patches to {sample_patch_dir}")

    # --- 6. Extract deep features ---
    logging.info(f"Extracting deep features (encoder={encoder_name}, dim={feature_dim})...")
    ds = PatchDataset(patches, resize_size=resize_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    features = extract_features_batch(encoder, dl, device)  # [N, D]

    # --- 7. Sanity check ---
    check_features(features, sample_name)

    # --- 8. Save ---
    os.makedirs(output_dir, exist_ok=True)

    coords_t = torch.tensor(coords_list, dtype=torch.float32)

    # Main feature file
    save_dict = {
        "sample_name": sample_name,
        "spot_ids": spot_ids,
        "features": features,               # [N, D] float32
        "coords": coords_t,                 # [N, 2]
        "encoder": f"{encoder_name}_imagenet",
        "feature_dim": feature_dim,
        "patch_size": patch_size,
        "resize_size": resize_size,
        "coord_scale": coord_scale,
        "coord_offset_x": coord_offset_x,
        "coord_offset_y": coord_offset_y,
    }
    pt_path = os.path.join(output_dir, f"{sample_name}_deep_features.pt")
    torch.save(save_dict, pt_path)
    logging.info(f"  Saved features to {pt_path}")

    # Metadata CSV
    rows = []
    for i in range(n_spots):
        rows.append({
            "sample_name": sample_name,
            "spot_id": spot_ids[i],
            "x": coords_list[i][0],
            "y": coords_list[i][1],
            "feature_file": pt_path,
            "patch_path": patch_paths[i] if patch_paths[i] else "",
        })
    df_meta = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"{sample_name}_deep_features_metadata.csv")
    df_meta.to_csv(csv_path, index=False)
    logging.info(f"  Saved metadata to {csv_path}")

    return pt_path


# ============================================================================
# Batch processing
# ============================================================================

def find_pairs(
    image_dir: str,
    coord_dir: str,
    sample_glob: str = "*",
    image_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    coord_exts: Tuple[str, ...] = (".csv", ".tsv"),
) -> Dict[str, Dict[str, str]]:
    import fnmatch

    image_files = []
    for f in sorted(os.listdir(image_dir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in image_exts and fnmatch.fnmatch(stem, sample_glob):
            image_files.append((stem, os.path.join(image_dir, f)))

    coord_files = []
    for f in sorted(os.listdir(coord_dir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in coord_exts and fnmatch.fnmatch(stem, sample_glob):
            coord_files.append((stem, os.path.join(coord_dir, f)))

    # Match by stem
    coord_stems = {stem: path for stem, path in coord_files}
    pairs = {}
    for stem, img_path in image_files:
        if stem in coord_stems:
            pairs[stem] = {"image": img_path, "coord": coord_stems[stem]}
        else:
            logging.warning(f"No matching coord file for image {stem}, skipping.")

    logging.info(f"Found {len(pairs)} (image, coord) pairs.")
    for name in pairs:
        logging.info(f"  {name}: image={os.path.basename(pairs[name]['image'])}, "
                     f"coord={os.path.basename(pairs[name]['coord'])}")
    return pairs


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Extract Deep Patch Features from H&E Images"
    )
    # I/O
    parser.add_argument("--image_dir", required=True,
                        help="Directory containing H&E images")
    parser.add_argument("--coord_dir", required=True,
                        help="Directory containing spot coordinate files")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save deep feature .pt files")
    parser.add_argument("--patch_dir", default="/home/lyl/lyl/dataset/her2st/processed_data/patch",
                        help="Optional directory to save patch PNGs")
    # Encoder
    parser.add_argument("--encoder", default="resnet50",
                        choices=["resnet18", "resnet50", "resnet101", "uni"],
                        help="Pretrained encoder (resnet* or uni)")
    parser.add_argument("--pretrained", default="imagenet",
                        help="Pretrained weights (currently only imagenet, ignored for uni)")
    parser.add_argument("--uni_ckpt", default=None,
                        help="Path to UNI checkpoint .pth/.bin (required when --encoder uni)")
    # Patch
    parser.add_argument("--patch_size", type=int, default=224,
                        help="Patch size in pixels (square)")
    parser.add_argument("--resize_size", type=int, default=224,
                        help="Resize patch to this size before encoder")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for feature extraction")
    parser.add_argument("--device", default="cuda:0",
                        help="Device for inference")
    parser.add_argument("--save_patches", action="store_true", default=False,
                        help="Save cropped patch PNGs")
    # Coordinate parsing
    parser.add_argument("--coord_scale", type=float, default=1.0,
                        help="Scale factor for coordinates")
    parser.add_argument("--coord_offset_x", type=float, default=0.0,
                        help="Offset for x coordinates")
    parser.add_argument("--coord_offset_y", type=float, default=0.0,
                        help="Offset for y coordinates")
    parser.add_argument("--coord_x_col", default="pixel_x",
                        help="Column name for x coordinate")
    parser.add_argument("--coord_y_col", default="pixel_y",
                        help="Column name for y coordinate")
    parser.add_argument("--spot_id_col", default="spot_id",
                        help="Column name for spot ID")
    # File matching
    parser.add_argument("--sample_glob", default="*",
                        help="Glob pattern to filter sample names")
    parser.add_argument("--image_exts", default=".png,.jpg,.jpeg,.tif,.tiff",
                        help="Comma-separated image extensions")
    # Performance
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--no_amp", action="store_true", default=False,
                        help="Disable AMP")

    args = parser.parse_args()

    image_exts = tuple(args.image_exts.split(","))

    # --- Device ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # --- Build encoder ---
    logging.info(f"Building encoder: {args.encoder} (pretrained={args.pretrained})")
    encoder, feature_dim = build_encoder(args.encoder, args.pretrained, uni_ckpt=args.uni_ckpt)
    encoder.to(device)
    encoder.eval()
    logging.info(f"  Feature dimension: {feature_dim}")

    # --- Find (image, coord) pairs ---
    pairs = find_pairs(
        args.image_dir, args.coord_dir,
        sample_glob=args.sample_glob,
        image_exts=image_exts,
    )
    if not pairs:
        logging.error("No (image, coord) pairs found. Check --image_dir, --coord_dir, "
                      "--sample_glob, --image_exts.")
        sys.exit(1)

    # --- Process each sample ---
    results = {}
    for sample_name, paths in pairs.items():
        try:
            out_path = process_sample(
                sample_name=sample_name,
                image_path=paths["image"],
                coord_path=paths["coord"],
                output_dir=args.output_dir,
                patch_dir=args.patch_dir,
                encoder=encoder,
                feature_dim=feature_dim,
                encoder_name=args.encoder,
                patch_size=args.patch_size,
                resize_size=args.resize_size,
                batch_size=args.batch_size,
                device=device,
                num_workers=args.num_workers,
                save_patches=args.save_patches,
                coord_scale=args.coord_scale,
                coord_offset_x=args.coord_offset_x,
                coord_offset_y=args.coord_offset_y,
                spot_id_col=args.spot_id_col,
                x_col=args.coord_x_col,
                y_col=args.coord_y_col,
            )
            if out_path:
                results[sample_name] = out_path
        except Exception as e:
            logging.error(f"Failed to process {sample_name}: {e}", exc_info=True)

    logging.info(f"\n{'=' * 60}")
    logging.info(f"Done! Processed {len(results)}/{len(pairs)} samples.")
    for name, path in results.items():
        logging.info(f"  {name}: {path}")
    logging.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
