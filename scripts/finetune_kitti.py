"""
Fine-tune the CARLA-pretrained STTR (updated-baseline / L1) on KITTI 2015.

Run from the model root:
    cd 2.0/updated-baseline
    python scripts/finetune_kitti.py

Strategy:
  - Load ONLY the model weights from the CARLA checkpoint (not the optimizer).
  - Freeze the backbone for the first FREEZE_EPOCHS at lr=1e-5 (adapt the head).
  - Unfreeze and continue at lr=1e-6 (gently adapt features to real images).
  - Early-stop on the best KITTI test EPE; save to *_kitti.pth.tar.
"""
import os
import sys
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# One level up from scripts/ -> model root; import the local `src` package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import CustomStereoDataset
from src.loss import log_disparity_loss
from src.model import StereoTransformer
from src.train_eval import train_epoch, test_epoch
from src.checkpoint import save_checkpoint

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Dataset', 'KITTI'))
left_dir  = os.path.join(DATA_ROOT, 'left')
right_dir = os.path.join(DATA_ROOT, 'right')
# Sparse LiDAR GT. Swap to 'disparity_dense' to train on densified GT instead.
depth_dir = os.path.join(DATA_ROOT, 'disparity')

PRETRAINED           = os.path.join(os.path.dirname(__file__), '..', 'best_model.pth.tar')          # CARLA weights
LAST_CHECKPOINT_FILE = os.path.join(os.path.dirname(__file__), '..', 'last_checkpoint_kitti.pth.tar')
BEST_CHECKPOINT_FILE = os.path.join(os.path.dirname(__file__), '..', 'best_model_kitti.pth.tar')

# KITTI 2015 camera intrinsics (NOT CARLA's 1385.64 / 1.0)
focal_length = 721.5    # px
baseline     = 0.54     # m

batch_size   = 4        # KITTI images are large
image_height = 384      # 375 -> 384 (multiple of 8)
image_width  = 1248     # 1242 -> 1248 (multiple of 8)

FREEZE_EPOCHS = 10      # backbone frozen, lr=1e-5
TOTAL_EPOCHS  = 40      # then unfrozen, lr=1e-6
LR_FROZEN     = 1e-5
LR_UNFROZEN   = 1e-6
PATIENCE      = 10      # early stop after this many epochs with no test-EPE gain

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
depth_transform = transforms.Compose([
    transforms.Resize((image_height, image_width), interpolation=Image.NEAREST),
])

dataset = CustomStereoDataset(
    left_dir=left_dir, right_dir=right_dir, depth_dir=depth_dir,
    transform=transform, depth_transform=depth_transform,
    focal_length=focal_length, baseline=baseline,
    zero_is_invalid=True,   # KITTI: disparity 0 = no LiDAR return, must be masked out
)
assert len(dataset) > 0, f"No triplets found under {DATA_ROOT} — check paths / extensions."

train_size = int(0.9 * len(dataset))
test_size  = len(dataset) - train_size
train_dataset, test_dataset = random_split(
    dataset, [train_size, test_size],
    generator=torch.Generator().manual_seed(42),  # reproducible split on 200 samples
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# ----------------------------------------------------------------------------
# Model + load CARLA weights only
# ----------------------------------------------------------------------------
model = StereoTransformer(hidden_dim=64, nhead=8, num_attn_layers=3, max_disp=256).to(device)

ckpt = torch.load(os.path.abspath(PRETRAINED), map_location=device)
model.load_state_dict(ckpt['model_state_dict'])   # weights only — NOT the optimizer
print(f"Loaded CARLA weights from {PRETRAINED} (epoch {ckpt.get('epoch')})")

criterion = log_disparity_loss


def make_optimizer(lr):
    return optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


def set_backbone_frozen(frozen: bool):
    for p in model.backbone.parameters():
        p.requires_grad = not frozen


# ----------------------------------------------------------------------------
# Fine-tune loop with freeze schedule + early stopping
# ----------------------------------------------------------------------------
set_backbone_frozen(True)
optimizer = make_optimizer(LR_FROZEN)
print(f"Phase 1: backbone frozen, lr={LR_FROZEN} for {FREEZE_EPOCHS} epochs")

train_loss_history, test_loss_history = [], []
train_epe_history,  test_epe_history  = [], []
best_test_epe = float('inf')
epochs_no_improve = 0

for epoch in range(TOTAL_EPOCHS):
    if epoch == FREEZE_EPOCHS:
        set_backbone_frozen(False)
        optimizer = make_optimizer(LR_UNFROZEN)
        print(f"Phase 2: backbone unfrozen, lr={LR_UNFROZEN}")

    train_loss, train_epe = train_epoch(model, train_loader, optimizer, criterion, device)
    test_loss,  test_epe  = test_epoch(model, test_loader, criterion, device)

    train_loss_history.append(train_loss); test_loss_history.append(test_loss)
    train_epe_history.append(train_epe);   test_epe_history.append(test_epe)

    print(f"Epoch {epoch+1}/{TOTAL_EPOCHS} - "
          f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
          f"Train EPE: {train_epe:.4f}, Test EPE: {test_epe:.4f}")

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_test_epe,
        'train_loss_history': train_loss_history,
        'test_loss_history': test_loss_history,
        'train_epe_history': train_epe_history,
        'test_epe_history': test_epe_history,
    }
    save_checkpoint(checkpoint, filename=os.path.abspath(LAST_CHECKPOINT_FILE))

    if test_epe < best_test_epe:
        best_test_epe = test_epe
        epochs_no_improve = 0
        import shutil
        shutil.copyfile(os.path.abspath(LAST_CHECKPOINT_FILE), os.path.abspath(BEST_CHECKPOINT_FILE))
        print(f"  New best KITTI test EPE: {best_test_epe:.4f} -> saved best_model_kitti.pth.tar")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping: no test-EPE improvement for {PATIENCE} epochs.")
            break

print(f"Fine-tuning finished. Best KITTI test EPE: {best_test_epe:.4f}")
