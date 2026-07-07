import torch

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
    Saves the current model and training state to a file.
    
    Args:
        state (dict): A dictionary containing model state, optimizer state, epoch, and loss.
        filename (str): The name of the file to save the checkpoint to.
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint_file, model, optimizer):
    """
    Loads a checkpoint and restores the model, optimizer, and loss history.
    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint.get('best_loss', float('inf'))
    
    # Load the loss history, defaulting to an empty list if not found
    train_history = checkpoint.get('train_loss_history', [])
    test_history = checkpoint.get('test_loss_history', [])
    
    return start_epoch, best_loss, train_history, test_history

def inspect_checkpoint(checkpoint_path):
    """
    Inspect a checkpoint to determine the model architecture.
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['model_state_dict']
    
    print("=" * 70)
    print("Checkpoint Architecture Inspector")
    print("=" * 70)
    
    # Detect hidden_dim
    if 'transformer.norm.weight' in state_dict:
        hidden_dim = state_dict['transformer.norm.weight'].shape[0]
        print(f"✓ hidden_dim: {hidden_dim}")
    
    # Count layers
    layer_keys = [k for k in state_dict.keys() if 'transformer.layers.' in k]
    max_layer = max([int(k.split('.')[2]) for k in layer_keys if k.split('.')[2].isdigit()]) + 1
    print(f"✓ num_attn_layers: {max_layer}")
    
    # Detect max_disp from regression head
    if 'regression_head.disp_pred.weight' in state_dict:
        disp_channels = state_dict['regression_head.disp_pred.weight'].shape[0]
        max_disp = disp_channels * 8  # Assuming 8x downsampling
        print(f"✓ max_disp: ~{max_disp} (estimated)")
    
    # Other info
    if 'epoch' in ckpt:
        print(f"✓ Trained for {ckpt['epoch']} epochs")
    if 'best_loss' in ckpt:
        print(f"✓ Best loss: {ckpt['best_loss']:.4f}")
    
    print()
    print("To load this checkpoint, use:")
    print(f"model = StereoTransformer(hidden_dim={hidden_dim}, nhead=8, num_attn_layers={max_layer})")
    print("=" * 70)
    
    return hidden_dim, max_layer