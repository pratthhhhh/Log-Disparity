import torch
import sys
import os

# Make `src` importable whether the model root or its parent is on sys.path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from src.loss import compute_epe
except ModuleNotFoundError:  # fallback for legacy L1/WHT_L1/WHT_L1DCT layouts
    from loss import compute_epe

# bf16 autocast on CUDA (tensor cores + Flash Attention). bf16 has the same
# exponent range as fp32, so no GradScaler is needed.
_AMP_DTYPE = torch.bfloat16


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_epe = 0
    use_amp = (device.type == 'cuda')
    for left, right, depth_gt in dataloader:
        left, right, depth_gt = left.to(device), right.to(device), depth_gt.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=_AMP_DTYPE, enabled=use_amp):
            depth_pred = model(left, right)
            loss = criterion(depth_pred, depth_gt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_epe += compute_epe(depth_pred.float(), depth_gt)

    avg_loss = total_loss / len(dataloader)
    avg_epe = total_epe / len(dataloader)
    return avg_loss, avg_epe


def test_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_epe = 0
    use_amp = (device.type == 'cuda')
    with torch.no_grad():
        for left, right, depth_gt in dataloader:
            left, right, depth_gt = left.to(device), right.to(device), depth_gt.to(device)
            with torch.autocast(device_type='cuda', dtype=_AMP_DTYPE, enabled=use_amp):
                depth_pred = model(left, right)
                loss = criterion(depth_pred, depth_gt)
            total_loss += loss.item()
            total_epe += compute_epe(depth_pred.float(), depth_gt)

    avg_loss = total_loss / len(dataloader)
    avg_epe = total_epe / len(dataloader)
    return avg_loss, avg_epe
