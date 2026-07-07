import matplotlib.pyplot as plt
import numpy as np
import random
import torch

def plot_loss_epe_curves(train_loss_history, test_loss_history):
    """
    Plots the training/testing loss and EPE curves.
    """
    fig, axs = plt.subplots(1, 1, figsize=(16, 5))
    # Loss plot
    axs[0].plot(train_loss_history, label='Train Loss')
    axs[0].plot(test_loss_history, label='Test Loss')
    axs[0].set_title('Training and Test Loss Over Epochs')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss (L1)')
    axs[0].legend()
    axs[0].grid(True)
    plt.show()

def visualize_results(model, dataloader, device, num_samples=3):
    """
    Visualizes the model's predictions on a few samples from the dataloader.
    """
    model.eval()
    
    # Get a batch of data
    try:
        left_batch, right_batch, disp_gt_batch = next(iter(dataloader))
    except StopIteration:
        print("Test loader is empty. Cannot visualize results.")
        return
    
    # Determine how many samples to show
    batch_size = left_batch.size(0)
    num_to_show = min(num_samples, batch_size)
    
    # Select random indices from the batch
    if num_to_show < batch_size:
        sample_indices = random.sample(range(batch_size), num_to_show)
    else:
        sample_indices = list(range(num_to_show))
    
    # Get the selected samples
    left_images = left_batch[sample_indices].to(device)
    right_images = right_batch[sample_indices].to(device)
    disp_gts = disp_gt_batch[sample_indices].to(device)
    
    # Get predictions
    with torch.no_grad():
        disp_preds = model(left_images, right_images)
    
    # Move tensors to CPU for plotting
    left_images_np = left_images.cpu().numpy()
    disp_gts_np = disp_gts.cpu().numpy()
    disp_preds_np = disp_preds.cpu().numpy()
    
    # --- Plotting ---
    fig, axes = plt.subplots(num_to_show, 3, figsize=(15, num_to_show * 4))
    
    # Handle case when num_to_show == 1
    if num_to_show == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_to_show):
        # Transpose image from (C, H, W) to (H, W, C) for displaying
        img = np.transpose(left_images_np[i], (1, 2, 0))
        
        # Un-normalize the image for visualization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Plot Left Image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Left Image')
        axes[i, 0].axis('off')
        
        # Plot Ground Truth Disparity
        gt_disp = axes[i, 1].imshow(disp_gts_np[i], cmap='magma', vmin=0, vmax=disp_gts_np[i].max())
        axes[i, 1].set_title(f'GT Disparity (max: {disp_gts_np[i].max():.2f})')
        axes[i, 1].axis('off')
        plt.colorbar(gt_disp, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # Plot Predicted Disparity
        pred_disp = axes[i, 2].imshow(disp_preds_np[i], cmap='magma', vmin=0, vmax=disp_preds_np[i].max())
        axes[i, 2].set_title(f'Predicted Disparity (max: {disp_preds_np[i].max():.2f})')
        axes[i, 2].axis('off')
        plt.colorbar(pred_disp, ax=axes[i, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print("\n=== Disparity Statistics ===")
    for i in range(num_to_show):
        gt_valid = disp_gts_np[i][disp_gts_np[i] > 0]
        pred_valid = disp_preds_np[i][disp_gts_np[i] > 0]
        
        if len(gt_valid) > 0:
            mae = np.abs(pred_valid - gt_valid).mean()
            print(f"Sample {i+1}:")
            print(f"  GT range: [{gt_valid.min():.2f}, {gt_valid.max():.2f}]")
            print(f"  Pred range: [{pred_valid.min():.2f}, {pred_valid.max():.2f}]")
            print(f"  MAE: {mae:.2f}")

def visualize_results_grid(model, dataloader, device, num_samples=6):
    """
    Visualizes multiple samples in a larger grid format.
    Shows only GT and Predicted disparity side by side.
    """
    model.eval()
    
    # Collect samples
    samples_collected = 0
    all_left = []
    all_right = []
    all_gt = []
    all_pred = []
    
    with torch.no_grad():
        for left_batch, right_batch, disp_gt_batch in dataloader:
            if samples_collected >= num_samples:
                break
            
            batch_size = left_batch.size(0)
            num_to_take = min(num_samples - samples_collected, batch_size)
            
            left = left_batch[:num_to_take].to(device)
            right = right_batch[:num_to_take].to(device)
            gt = disp_gt_batch[:num_to_take]
            
            pred = model(left, right)
            
            all_left.append(left.cpu())
            all_right.append(right.cpu())
            all_gt.append(gt)
            all_pred.append(pred.cpu())
            
            samples_collected += num_to_take
    
    if samples_collected == 0:
        print("No samples to visualize.")
        return
    
    # Concatenate all samples
    all_left = torch.cat(all_left, dim=0).numpy()
    all_gt = torch.cat(all_gt, dim=0).numpy()
    all_pred = torch.cat(all_pred, dim=0).numpy()
    
    # Create grid
    n_rows = (samples_collected + 1) // 2
    n_cols = 4  # Left Image | GT | Pred | Error
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(samples_collected):
        row = idx // 2
        col_offset = (idx % 2) * 2
        
        # Get image
        img = np.transpose(all_left[idx], (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Calculate error
        valid_mask = all_gt[idx] > 0
        error_map = np.abs(all_pred[idx] - all_gt[idx])
        error_map[~valid_mask] = 0
        
        # Left image (only show for first column)
        if col_offset == 0:
            axes[row, 0].imshow(img)
            axes[row, 0].set_title(f'Sample {idx+1}')
            axes[row, 0].axis('off')
        
        # GT Disparity
        gt_im = axes[row, col_offset + 1].imshow(all_gt[idx], cmap='magma')
        axes[row, col_offset + 1].set_title(f'GT (max: {all_gt[idx].max():.1f})')
        axes[row, col_offset + 1].axis('off')
        
        # Predicted Disparity
        pred_im = axes[row, col_offset + 2].imshow(all_pred[idx], cmap='magma')
        axes[row, col_offset + 2].set_title(f'Pred (max: {all_pred[idx].max():.1f})')
        axes[row, col_offset + 2].axis('off')
        
        # Error map
        if col_offset == 2:
            err_im = axes[row, 3].imshow(error_map, cmap='hot', vmin=0, vmax=10)
            mae = error_map[valid_mask].mean() if valid_mask.any() else 0
            axes[row, 3].set_title(f'Error (MAE: {mae:.2f})')
            axes[row, 3].axis('off')
    
    # Hide any unused subplots
    for idx in range(samples_collected, n_rows * 2):
        row = idx // 2
        col_offset = (idx % 2) * 2
        if col_offset == 0 and row < n_rows:
            axes[row, 0].axis('off')
        if row < n_rows:
            axes[row, col_offset + 1].axis('off')
            axes[row, col_offset + 2].axis('off')
        if col_offset == 2 and row < n_rows:
            axes[row, 3].axis('off')
    
    plt.tight_layout()
    plt.show()