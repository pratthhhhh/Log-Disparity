import torch
from torch import Tensor


def masked_l1_disparity_loss(pred_disp: Tensor, gt_disp: Tensor, valid_mask=None) -> Tensor:
    """
    L1 loss on valid pixels.

    Valid pixels are those with gt >= 0.  This includes zero-disparity pixels
    (sky / infinite depth).  Truly invalid pixels carry sentinel -1 and are
    excluded.
    """
    if valid_mask is None:
        valid_mask = gt_disp >= 0

    return torch.abs(pred_disp - gt_disp)[valid_mask].mean()


def log_disparity_loss(
    pred_disp: Tensor,
    gt_disp: Tensor,
    eps: float = 1e-6,
    sky_weight: float = 1.0,
) -> Tensor:
    """
    Hybrid log-disparity loss.

    Positive ground truth is supervised in log space, which balances near
    (large-disparity) and far (small-disparity) regions more evenly than plain L1.
    Zero-disparity pixels — sky / infinite depth, where ``log`` is undefined — are
    handled separately with a linear L1 term that simply pulls the prediction
    toward zero (the same treatment plain L1 gave them).  The two terms are then
    averaged together over all valid pixels:

        gt  > 0 : |log(pred) - log(gt)|    (log space, as usual)
        gt == 0 : sky_weight * |pred|      (sky; linear pull toward 0)
        gt  < 0 : sentinel (-1), ignored

    ``sky_weight`` damps the sky term if it starts to dominate the mean — log-space
    errors are typically << 1, so a large sky region can outweigh the informative
    pixels early in training.
    """
    finite = torch.isfinite(gt_disp)
    pos = finite & (gt_disp > 0)
    sky = finite & (gt_disp == 0)

    terms = []
    if pos.any():
        pred = pred_disp[pos].clamp_min(eps)
        gt   = gt_disp[pos].clamp_min(eps)
        terms.append((torch.log(pred) - torch.log(gt)).abs())
    if sky.any():
        terms.append(sky_weight * pred_disp[sky].abs())

    # Nothing valid in this batch -> return 0 so the training loop can skip it.
    if not terms:
        return pred_disp.new_tensor(0.0)

    return torch.cat(terms).mean()


def compute_epe(pred: Tensor, gt: Tensor) -> float:
    """End-Point Error on valid pixels (gt >= 0)."""
    mask = gt >= 0
    if mask.sum() == 0:
        return 0.0
    return torch.abs(pred[mask] - gt[mask]).mean().item()


@torch.no_grad()
def evaluate_long_range_metrics(model, loader, device):
    model.eval()

    total_abs = 0.0
    total_n = 0

    bin_edges = (0, 2, 4, 8, 16, 32, 64, 128, 1e9)
    bin_sums = [0.0] * (len(bin_edges) - 1)
    bin_counts = [0] * (len(bin_edges) - 1)

    far_sum = 0.0
    far_count = 0

    for left, right, gt_disp in loader:
        left    = left.to(device)
        right   = right.to(device)
        gt_disp = gt_disp.to(device).float()

        pred_disp = model(left, right)  # (B, H, W)

        valid = (gt_disp >= 0) & torch.isfinite(pred_disp)
        abs_err = (pred_disp - gt_disp).abs()

        n = valid.sum().item()
        if n > 0:
            total_abs += abs_err[valid].sum().item()
            total_n   += int(n)

        for i in range(len(bin_edges) - 1):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            m = valid & (gt_disp >= lo) & (gt_disp < hi)
            c = m.sum().item()
            if c > 0:
                bin_sums[i]   += abs_err[m].sum().item()
                bin_counts[i] += int(c)

        # Far region: very small disparity (< 4 px ≈ very distant objects)
        mfar  = valid & (gt_disp < 4.0)
        cfar  = mfar.sum().item()
        if cfar > 0:
            far_sum   += abs_err[mfar].sum().item()
            far_count += int(cfar)

    results = {}
    results["EPE_all"]   = (total_abs / total_n) if total_n > 0 else None
    results["valid_px"]  = total_n

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        key = f"EPE_disp[{lo},{hi})"
        results[key]           = (bin_sums[i] / bin_counts[i]) if bin_counts[i] > 0 else None
        results[key + "_n"]    = bin_counts[i]

    results["EPE_far(<4px)"] = (far_sum / far_count) if far_count > 0 else None
    results["EPE_far_n"]     = far_count

    return results
