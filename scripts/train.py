# Updated main training block
import numpy as np
import os
import sys
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True  # fixed input sizes -> autotune fastest convs

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.checkpoint import save_checkpoint, load_checkpoint
from src.data import CustomStereoDataset
from src.loss import log_disparity_loss
from src.model import StereoTransformer
from src.train_eval import train_epoch, test_epoch
from src.visualize import plot_loss_epe_curves

# Updated main training block
if __name__ == '__main__':
    DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Dataset', 'CARLA'))
    left_dir = os.path.join(DATA_ROOT, 'left')
    right_dir = os.path.join(DATA_ROOT, 'right')
    depth_dir = os.path.join(DATA_ROOT, 'disparity')
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 200
    image_height = 512 #384
    image_width = 1024 #768

    # Camera intrinsics (from user)
    K = np.array([[1.38564065e+03, 0.00000000e+00, 8.00000000e+02],
                  [0.00000000e+00, 1.38564065e+03, 4.50000000e+02],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    focal_length = K[0, 0]  # fx in pixels
    baseline = 1.0          # 100 cm = 1.0 m

    # --- Checkpoint file paths ---
    MODEL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    LAST_CHECKPOINT_FILE = os.path.join(MODEL_ROOT, 'last_checkpoint.pth.tar')
    BEST_CHECKPOINT_FILE = os.path.join(MODEL_ROOT, 'best_model.pth.tar')

    # --- Data Loading and Preprocessing ---
    transform = transforms.Compose([

        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    depth_transform = transforms.Compose([
        transforms.Resize((image_height, image_width), interpolation=Image.NEAREST)
    ])

    # Check if the data directories exist
    if not (os.path.isdir(left_dir) and os.path.isdir(right_dir) and os.path.isdir(depth_dir)):
        print(f"Error: One or more data directories do not exist. Please update the left_dir, right_dir, and depth_dir variables.")
    else:
        dataset = CustomStereoDataset(
            left_dir=left_dir,
            right_dir=right_dir,
            depth_dir=depth_dir,
            transform=transform,
            depth_transform=depth_transform,
            focal_length=focal_length,
            baseline=baseline
        )

        # Train-Test Split
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True, persistent_workers=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device:", device)

        if device.type == 'cuda':
            print("CUDA available:", torch.cuda.is_available())
            print("GPU name:", torch.cuda.get_device_name(0))
            print("Number of GPUs:", torch.cuda.device_count())
    
        # Create model with proper STTR architecture
        model = StereoTransformer(
            hidden_dim=64,
            nhead=8,
            num_attn_layers=3,
            max_disp=256,
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = log_disparity_loss
    
        # --- Initialize loss history lists ---
        train_loss_history = []
        test_loss_history = []
        train_epe_history = []
        test_epe_history = []

        # --- Load Checkpoint if it exists ---
        start_epoch = 0
        best_test_loss = float('inf')
        if os.path.exists(LAST_CHECKPOINT_FILE):
            start_epoch, best_test_loss, train_loss_history, test_loss_history = load_checkpoint(
                LAST_CHECKPOINT_FILE, model, optimizer
            )
            print(f"Resuming training from epoch {start_epoch}")
            print(f"Loaded {len(train_loss_history)} epochs of loss history.")

        # --- Training and Testing Loop ---
        SAVE_EVERY = 50   # also save a periodic resumable checkpoint every N epochs
        for epoch in range(start_epoch, num_epochs):
            train_loss, train_epe = train_epoch(model, train_loader, optimizer, criterion, device)
            test_loss, test_epe = test_epoch(model, test_loader, criterion, device)

            train_loss_history.append(train_loss)
            test_loss_history.append(test_loss)
            train_epe_history.append(train_epe)
            test_epe_history.append(test_epe)

            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train EPE: {train_epe:.4f}, Test EPE: {test_epe:.4f}")

            # --- Checkpointing (inside the loop) ---
            is_best = test_loss < best_test_loss
            if is_best:
                best_test_loss = test_loss
                print(f"  New best model found with test loss: {best_test_loss:.4f}")

            checkpoint = {
                'epoch': epoch + 1,   # number of completed epochs -> correct resume point
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_test_loss,
                'train_loss_history': train_loss_history,
                'test_loss_history': test_loss_history,
                'train_epe_history': train_epe_history,
                'test_epe_history': test_epe_history
            }

            # Always keep best_model.pth.tar current
            if is_best:
                save_checkpoint(checkpoint, filename=BEST_CHECKPOINT_FILE)

            # Periodic + final: update last_checkpoint and write a numbered restore point
            is_last = (epoch + 1) == num_epochs
            if (epoch + 1) % SAVE_EVERY == 0 or is_last:
                save_checkpoint(checkpoint, filename=LAST_CHECKPOINT_FILE)
                epoch_ckpt = os.path.join(MODEL_ROOT, f'checkpoint_epoch_{epoch + 1}.pth.tar')
                save_checkpoint(checkpoint, filename=epoch_ckpt)
                print(f"  Saved periodic checkpoint at epoch {epoch + 1}")

    print("Training finished.")

        # Plot both loss and EPE curves (only available after training)
    try:
        plot_loss_epe_curves(train_loss_history, test_loss_history, train_epe_history, test_epe_history)
    except Exception:
        pass