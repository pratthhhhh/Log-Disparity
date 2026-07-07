"""
Evaluation / inference for the log-loss StereoTransformer on CARLA.

There is no KITTI fine-tuned checkpoint, so this runs a single pass:

  CARLA : accuracy metrics (EPE, RMSE, bad-pixel rates, D1) + a per-disparity-bin
          EPE breakdown on Dataset/testCARLA using best_model.pth.tar, plus
          GFLOPs and a couple of prediction figures (left / GT / pred).

Run from the model root:
    cd log-loss
    python scripts/eval.py
"""

import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless: save figures instead of showing them
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.profiler import profile, ProfilerActivity

# Model root on PYTHONPATH so `src` imports resolve.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import CustomStereoDataset
from src.model import StereoTransformer

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

MODEL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_ROOT = os.path.abspath(os.path.join(MODEL_ROOT, "..", "Dataset"))
OUTPUT_DIR = os.path.join(MODEL_ROOT, "eval_outputs")

CARLA_CKPT = os.path.join(MODEL_ROOT, "best_model.pth.tar")   # CARLA-trained weights
TEST_CARLA = os.path.join(DATASET_ROOT, "testCARLA")          # has GT disparity

# Architecture must match the checkpoint exactly.
HIDDEN_DIM = 64
NHEAD = 8
NUM_ATTN_LAYERS = 3
MAX_DISP = 256

# Inference resolution (multiples of 8). MUST match the training resolution:
# the disparity head predicts in the pixel-scale it was trained at, so evaluating
# at a different size mis-scales disparities (esp. large ones).
CARLA_H, CARLA_W = 512, 1024    # matches train.py (CARLA frames are 1600x900)

# Bad-pixel accuracy thresholds (px) and KITTI D1 rule (>3px AND >5% of GT).
BAD_THRESHOLDS = (1.0, 2.0, 3.0)
D1_ABS_PX = 3.0
D1_REL = 0.05

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Model / transforms
# ---------------------------------------------------------------------------

def build_model(checkpoint_path, device):
    """Instantiate StereoTransformer and load weights from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = StereoTransformer(
        hidden_dim=HIDDEN_DIM,
        nhead=NHEAD,
        num_attn_layers=NUM_ATTN_LAYERS,
        max_disp=MAX_DISP,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device).eval()
    print(f"Loaded {os.path.basename(checkpoint_path)} "
          f"(epoch {ckpt.get('epoch')}, best_loss {ckpt.get('best_loss')})")
    return model


def make_transforms(height, width):
    from PIL import Image
    img_tf = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    depth_tf = transforms.Compose([
        transforms.Resize((height, width), interpolation=Image.NEAREST),
    ])
    return img_tf, depth_tf


def unnormalize(img_chw):
    """(C,H,W) normalized tensor/array -> (H,W,C) displayable image in [0,1]."""
    img = np.transpose(np.asarray(img_chw), (1, 2, 0))
    img = np.array(IMAGENET_STD) * img + np.array(IMAGENET_MEAN)
    return np.clip(img, 0, 1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    """
    Single-pass evaluation over valid pixels (gt >= 0).

    Returns accuracy metrics (EPE, RMSE, bad-pixel rates, KITTI D1) plus a
    per-disparity-bin EPE breakdown.
    """
    bin_edges = (0, 2, 4, 8, 16, 32, 64, 128, 1e9)
    bin_sums = [0.0] * (len(bin_edges) - 1)
    bin_counts = [0] * (len(bin_edges) - 1)

    total_abs, total_sq, total_n = 0.0, 0.0, 0
    bad_counts = {t: 0 for t in BAD_THRESHOLDS}
    d1_count = 0

    for left, right, disp_gt in loader:
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        disp_gt = disp_gt.to(device, non_blocking=True).float()

        pred = model(left, right).float()

        valid = torch.isfinite(disp_gt) & (disp_gt >= 0) & torch.isfinite(pred)
        if valid.sum() == 0:
            continue

        gt = disp_gt[valid]
        epe = (pred[valid] - gt).abs()

        total_abs += epe.sum().item()
        total_sq += (epe ** 2).sum().item()
        total_n += epe.numel()

        # Bad-pixel accuracy: fraction of pixels with error above each threshold.
        for t in BAD_THRESHOLDS:
            bad_counts[t] += int((epe > t).sum().item())

        # KITTI D1: error > 3px AND > 5% of the true disparity.
        d1_count += int(((epe > D1_ABS_PX) & (epe > D1_REL * gt)).sum().item())

        # Per-disparity-bin EPE.
        for i in range(len(bin_edges) - 1):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            m = (gt >= lo) & (gt < hi)
            if m.any():
                bin_sums[i] += epe[m].sum().item()
                bin_counts[i] += int(m.sum().item())

    results = {
        "EPE_all": (total_abs / total_n) if total_n > 0 else None,
        "RMSE": (total_sq / total_n) ** 0.5 if total_n > 0 else None,
        "valid_px": total_n,
    }
    for t in BAD_THRESHOLDS:
        results[f"bad{t:g}px_%"] = (100.0 * bad_counts[t] / total_n) if total_n else None
    results["D1_all_%"] = (100.0 * d1_count / total_n) if total_n else None

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        key = f"EPE_disp[{lo},{hi})"
        results[key] = (bin_sums[i] / bin_counts[i]) if bin_counts[i] else None
        results[key + "_n"] = bin_counts[i]
    return results


def print_metrics(title, metrics):
    print(f"\n=== {title} ===")

    if metrics["EPE_all"] is not None:
        print("-- Accuracy metrics --")
        print(f"  EPE (mean abs err) : {metrics['EPE_all']:.4f} px")
        print(f"  RMSE               : {metrics['RMSE']:.4f} px")
        for t in BAD_THRESHOLDS:
            print(f"  bad>{t:g}px          : {metrics[f'bad{t:g}px_%']:.2f} %")
        print(f"  D1-all (>3px &>5%) : {metrics['D1_all_%']:.2f} %")
        print(f"  valid pixels       : {metrics['valid_px']:,}")
    else:
        print("  n/a (no valid pixels)")

    print("-- Per-disparity-bin EPE --")
    for k, v in metrics.items():
        if k.startswith("EPE_disp") and not k.endswith("_n"):
            n = metrics[k + "_n"]
            v_str = f"{v:.4f}" if v is not None else "n/a"
            print(f"  {k:22s} {v_str:>8s} px  (n={n:,})")


@torch.no_grad()
def compute_gflops(model, device, height, width):
    """GFLOPs for a single forward pass at the given resolution."""
    left = torch.randn(1, 3, height, width, device=device)
    right = torch.randn(1, 3, height, width, device=device)

    _ = model(left, right)  # warmup

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, with_flops=True) as prof:
        out = model(left, right)

    total_flops = sum(e.flops for e in prof.key_averages() if e.flops)
    print(f"\n=== GFLOPs (1x3x{height}x{width}) ===")
    print(f"Output shape: {tuple(out.shape)}")
    print(f"GFLOPs: {total_flops / 1e9:.3f}   (GMACs: {total_flops / 2 / 1e9:.3f})")
    return total_flops / 1e9


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_with_gt(model, dataset, device, num_samples, out_path, indices=None):
    """Save a left / GT / pred figure for the given (or evenly-spaced) samples."""
    if indices is None:
        # Spread samples across the dataset so we don't get near-duplicate frames.
        n = min(num_samples, len(dataset))
        indices = np.linspace(0, len(dataset) - 1, n, dtype=int).tolist()
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(15, n * 4))
    if n == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(indices):
        i = row  # axis row
        left, right, gt = dataset[idx]
        pred = model(left.unsqueeze(0).to(device),
                     right.unsqueeze(0).to(device))[0].cpu().numpy()
        gt = gt.numpy()

        axes[i, 0].imshow(unnormalize(left))
        axes[i, 0].set_title(f"Left (sample {idx})")
        axes[i, 0].axis("off")

        vmax = max(gt.max(), 1e-6)
        im = axes[i, 1].imshow(np.where(gt >= 0, gt, np.nan), cmap="magma", vmin=0, vmax=vmax)
        axes[i, 1].set_title(f"GT disparity (max {vmax:.1f})")
        axes[i, 1].axis("off")
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

        im = axes[i, 2].imshow(pred, cmap="magma", vmin=0, vmax=max(pred.max(), 1e-6))
        axes[i, 2].set_title(f"Pred disparity (max {pred.max():.1f})")
        axes[i, 2].axis("off")
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

class _Tee:
    """Write to multiple streams (stdout + log file) at once."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------- CARLA ----------------
    print("\n" + "=" * 70 + "\nCARLA (testCARLA, best_model.pth.tar)\n" + "=" * 70)
    carla_model = build_model(CARLA_CKPT, device)
    img_tf, depth_tf = make_transforms(CARLA_H, CARLA_W)
    carla_ds = CustomStereoDataset(
        left_dir=os.path.join(TEST_CARLA, "left"),
        right_dir=os.path.join(TEST_CARLA, "right"),
        depth_dir=os.path.join(TEST_CARLA, "disparity"),
        transform=img_tf, depth_transform=depth_tf,
        zero_is_invalid=False,  # CARLA: 0 disparity (sky) is valid
    )
    carla_loader = DataLoader(carla_ds, batch_size=2, shuffle=False)
    print_metrics("CARLA metrics", evaluate(carla_model, carla_loader, device))
    compute_gflops(carla_model, device, CARLA_H, CARLA_W)
    plot_with_gt(carla_model, carla_ds, device, num_samples=3,
                 out_path=os.path.join(OUTPUT_DIR, "carla_samples.png"))

    print("\nDone. Figures written to:", OUTPUT_DIR)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "eval.log")
    with open(log_path, "w") as log_file:
        original_stdout = sys.stdout
        sys.stdout = _Tee(original_stdout, log_file)
        try:
            main()
            print(f"\nLog written to: {log_path}")
        finally:
            sys.stdout = original_stdout
