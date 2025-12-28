"""
Quantization-Aware Training Utilities

This module provides training utilities specifically for quantization-aware training,
including:
- Joint Quantization training with random bitwidth sampling
- Random bitwidths validation with statistical reporting
- Layer-wise quantization setup and bitwidth management
"""

import os
import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# Import validation function from train_test_utils
from train_test.train_test_utils import validate, calculate_confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def get_conv_class_from_model(model):
    """
    Extract the Conv class being used in the model by inspecting its modules.
    
    Args:
        model: PyTorch model
    
    Returns:
        Conv class type, or None if no quantized Conv layers found
    """
    return model.get_conv_class()


def get_average_bitwidth(model):
    """
    Calculate the average bitwidth across all quantized Conv layers in the model.
    
    Args:
        model: PyTorch model containing quantized Conv layers
    
    Returns:
        float: Average bitwidth across all layers, or None if no quantized layers found
    """
    conv_class = get_conv_class_from_model(model)
    if conv_class is None:
        return None
    
    bitwidths = []
    for module in model.modules():
        if isinstance(module, conv_class):
            if hasattr(module, 'curr_bitwidth') and module.curr_bitwidth is not None:
                bitwidths.append(module.curr_bitwidth)
    
    if len(bitwidths) == 0:
        return None
    
    return np.mean(bitwidths)


def set_random_bitwidth_all_layers(model, bitwidth_options):
    """
    Set random bitwidths for all quantized Conv layers in the model.
    Each layer independently samples from the provided bitwidth options.
    
    Args:
        model: PyTorch model containing quantized Conv layers
        bitwidth_options: List of bitwidth options to sample from (e.g., [4, 6, 8])
    """
    conv_class = get_conv_class_from_model(model)
    if conv_class is None:
        return  # No quantized Conv layers found
    
    for module in model.modules():
        if isinstance(module, conv_class):
            # Each layer samples independently
            bw = bitwidth_options[torch.randint(0, len(bitwidth_options), (1,)).item()]
            module.set_bitwidth(bw)


def setup_quantization_layers(model, quant_config):
    """
    Setup quantization functions for all quantized Conv layers in the model.
    
    Args:
        model: PyTorch model containing quantized Conv layers
        quant_config: Quantization configuration dictionary
    
    Returns:
        model: Model with quantization layers configured
    """
    conv_class = get_conv_class_from_model(model)
    if conv_class is None:
        logging.warning("No quantized Conv layers found in model")
        return model
    
    # Setup quantization for all quantized Conv layers
    for module in model.modules():
        if isinstance(module, conv_class):
            module.setup_quantize_funcs(quant_config)
    
    logging.info("Quantization layers configured successfully")
    logging.info(f"  Weight quantization: {quant_config.get('weight_quantization', 'N/A')}")
    logging.info(f"  Activation quantization: {quant_config.get('activation_quantization', 'N/A')}")
    logging.info(f"  Bitwidth options: {quant_config.get('bitwidth_options', 'N/A')}")
    
    return model


def validate_simple(model, val_loader, loss_fn, device, augmenter, apply_augmentation_fn, bitwidth):
    """
    Simple validation with a single fixed bitwidth.
    
    Args:
        model: PyTorch model to validate
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to run validation on
        augmenter: Data augmenter object
        apply_augmentation_fn: Function to apply augmentation
        bitwidth: Fixed bitwidth to use for validation
    
    Returns:
        dict: Validation metrics (accuracy, loss)
    """
    model.eval()
    
    # Set fixed bitwidth for all layers
    conv_class = get_conv_class_from_model(model)
    if conv_class is not None:
        for module in model.modules():
            if isinstance(module, conv_class):
                module.set_bitwidth(bitwidth)
    
    logging.info(f"Validating with fixed bitwidth: {bitwidth}")
    
    # Run standard validation
    val_result = validate(
        model, val_loader, loss_fn, device,
        augmenter, apply_augmentation_fn
    )
    
    logging.info(f"  Validation - Acc: {val_result['accuracy']:.4f}, Loss: {val_result['loss']:.4f}")
    
    return val_result


def validate_random_bitwidths(model, val_loader, loss_fn, device, 
                               augmenter, apply_augmentation_fn, num_configs, 
                               bitwidth_options, bin_tolerance=0.5):
    """
    Validate model with multiple random bitwidth configurations and report statistics.
    
    Args:
        model: PyTorch model to validate
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to run validation on
        augmenter: Data augmenter object
        apply_augmentation_fn: Function to apply augmentation
        num_configs: Number of random configurations to test
        bitwidth_options: List of bitwidth options
        bin_tolerance: Tolerance for grouping bitwidths into bins (default: 0.5)
    
    Returns:
        dict: Statistics including mean, min, max, std for accuracy, loss, and bitwidth
    """
    model.eval()
    results = []
    
    logging.info(f"Validating with {num_configs} random bitwidth configurations...")
    logging.info(f" validate_random_bitwidths: Bitwidth options: {bitwidth_options}")
    logging.info(f" validate_random_bitwidths: Bin tolerance: {bin_tolerance}")
    
    for i in range(num_configs):
        # Set random bitwidths for all layers
        set_random_bitwidth_all_layers(model, bitwidth_options)
        
        # Get average bitwidth for this configuration
        avg_bitwidth = get_average_bitwidth(model)
        
        # Run validation
        val_result = validate(
            model, val_loader, loss_fn, device,
            augmenter, apply_augmentation_fn
        )
        
        results.append({
            'accuracy': val_result['accuracy'],
            'loss': val_result['loss'],
            'avg_bitwidth': avg_bitwidth if avg_bitwidth is not None else 0.0
        })
        
        bitwidth_str = f", Avg Bitwidth={avg_bitwidth:.2f}" if avg_bitwidth is not None else ""
        logging.info(f"  Config {i+1}/{num_configs}: Acc={val_result['accuracy']:.4f}, "
                    f"Loss={val_result['loss']:.4f}{bitwidth_str}")
    
    # Calculate statistics
    accuracies = [r['accuracy'] for r in results]
    losses = [r['loss'] for r in results]
    avg_bitwidths = [r['avg_bitwidth'] for r in results]
    
    # Calculate bin-level statistics
    # Create bins based on unique bitwidths (rounded to 1 decimal)
    unique_bitwidths = np.unique(np.round(avg_bitwidths, 1))
    
    bin_stats = []
    # Assign each data point to the closest bin center (no overlapping bins)
    avg_bitwidths_array = np.array(avg_bitwidths)
    accuracies_array = np.array(accuracies)
    
    for bw in unique_bitwidths:
        # Find data points closest to this bin center
        distances = np.abs(avg_bitwidths_array - bw)
        
        # Assign each point to its closest bin
        # For each data point, check if this bin is the closest
        mask = np.array([np.abs(avg_bw - bw) == np.min([np.abs(avg_bw - ub) for ub in unique_bitwidths]) 
                         for avg_bw in avg_bitwidths_array])
        
        if np.sum(mask) > 0:
            bin_accs = accuracies_array[mask]
            bin_stats.append({
                'bitwidth': float(bw),
                'mean_acc': float(np.mean(bin_accs)),
                'std_acc': float(np.std(bin_accs)),
                'min_acc': float(np.min(bin_accs)),
                'max_acc': float(np.max(bin_accs)),
                'count': int(np.sum(mask))
            })
    
    # Calculate average validation std across all bins
    avg_val_std = np.mean([b['std_acc'] for b in bin_stats]) if bin_stats else 0.0
    
    stats = {
        'mean_acc': np.mean(accuracies),
        'min_acc': np.min(accuracies),
        'max_acc': np.max(accuracies),
        'std_acc': np.std(accuracies),
        'mean_loss': np.mean(losses),
        'min_loss': np.min(losses),
        'max_loss': np.max(losses),
        'std_loss': np.std(losses),
        'mean_bitwidth': np.mean(avg_bitwidths),
        'min_bitwidth': np.min(avg_bitwidths),
        'max_bitwidth': np.max(avg_bitwidths),
        'std_bitwidth': np.std(avg_bitwidths),
        'accuracy': np.mean(accuracies),  # For compatibility with standard validation
        'loss': np.mean(losses),
        # Store raw data for plotting
        'all_accuracies': accuracies,
        'all_bitwidths': avg_bitwidths,
        # Store bin-level statistics
        'bin_stats': bin_stats,
        'avg_val_std': avg_val_std
    }
    
    # Log summary statistics
    logging.info("Random Bitwidths Validation Statistics:")
    logging.info(f"  Accuracy - Mean: {stats['mean_acc']:.4f}, Min: {stats['min_acc']:.4f}, "
                f"Max: {stats['max_acc']:.4f}, Std: {stats['std_acc']:.4f}")
    logging.info(f"  Loss - Mean: {stats['mean_loss']:.4f}, Min: {stats['min_loss']:.4f}, "
                f"Max: {stats['max_loss']:.4f}, Std: {stats['std_loss']:.4f}")
    logging.info(f"  Bitwidth - Mean: {stats['mean_bitwidth']:.2f}, Min: {stats['min_bitwidth']:.2f}, "
                f"Max: {stats['max_bitwidth']:.2f}, Std: {stats['std_bitwidth']:.4f}")
    
    # Log bin-level statistics
    if bin_stats:
        logging.info(f"  Bin Statistics ({len(bin_stats)} bins, tolerance=±{bin_tolerance}):")
        for bin_stat in bin_stats:
            logging.info(f"    Bitwidth {bin_stat['bitwidth']:.1f}: "
                        f"Acc={bin_stat['mean_acc']:.4f}±{bin_stat['std_acc']:.4f} "
                        f"(n={bin_stat['count']})")
        logging.info(f"  Average Validation Std across bins: {avg_val_std:.4f}")
    
    return stats


def plot_bitwidth_vs_accuracy(bitwidths, accuracies, epoch, title="Bitwidth vs Accuracy", bin_tolerance=0.5):
    """
    Create a scatter plot or candlestick-style plot of bitwidth vs accuracy.
    
    Groups results by bitwidth bins and shows mean and std deviation.
    
    Args:
        bitwidths: List of average bitwidths
        accuracies: List of corresponding accuracies
        epoch: Current epoch number (for title)
        title: Plot title
        bin_tolerance: Tolerance for grouping bitwidths into bins (default: 0.5)
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to numpy arrays
    bitwidths = np.array(bitwidths)
    accuracies = np.array(accuracies)
    
    # Create bins based on unique bitwidths (rounded to 1 decimal)
    unique_bitwidths = np.unique(np.round(bitwidths, 1))
    
    if len(unique_bitwidths) > 1:
        # Create bins - assign each point to its closest bin center
        bin_means = []
        bin_stds = []
        bin_mins = []
        bin_maxs = []
        bin_centers = []
        
        for bw in unique_bitwidths:
            # Assign each point to its closest bin (no overlapping bins)
            mask = np.array([np.abs(bw_val - bw) == np.min([np.abs(bw_val - ub) for ub in unique_bitwidths]) 
                             for bw_val in bitwidths])
            
            if np.sum(mask) > 0:
                bin_accs = accuracies[mask]
                bin_centers.append(bw)
                bin_means.append(np.mean(bin_accs))
                bin_stds.append(np.std(bin_accs))
                bin_mins.append(np.min(bin_accs))
                bin_maxs.append(np.max(bin_accs))
        
        bin_centers = np.array(bin_centers)
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)
        bin_mins = np.array(bin_mins)
        bin_maxs = np.array(bin_maxs)
        
        # Plot individual points with transparency
        ax.scatter(bitwidths, accuracies, alpha=0.3, s=50, c='lightblue', 
                  edgecolors='blue', linewidth=0.5, label='Individual configs')
        
        # Plot mean line
        ax.plot(bin_centers, bin_means, 'ro-', linewidth=2, markersize=8, 
               label='Mean accuracy', zorder=5)
        
        # Plot error bars (std deviation)
        ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='none', 
                   ecolor='red', capsize=5, capthick=2, alpha=0.7, zorder=4)
        
        # Plot min-max range as vertical lines
        for i in range(len(bin_centers)):
            ax.plot([bin_centers[i], bin_centers[i]], [bin_mins[i], bin_maxs[i]], 
                   'k-', alpha=0.3, linewidth=1, zorder=3)
    else:
        # If only one bitwidth, just show scatter
        ax.scatter(bitwidths, accuracies, alpha=0.6, s=100, c='blue', 
                  edgecolors='darkblue', linewidth=1)
        ax.axhline(y=np.mean(accuracies), color='r', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(accuracies):.4f}')
    
    ax.set_xlabel('Average Bitwidth', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'{title} (Epoch {epoch + 1})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9)
    
    # Add statistics text box
    stats_text = f'Mean Acc: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n'
    stats_text += f'Mean BW: {np.mean(bitwidths):.2f} ± {np.std(bitwidths):.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig


def train_epoch_vanilla_single_precision(model, train_loader, optimizer, loss_fn, 
                                         quant_config, augmenter, apply_augmentation_fn, 
                                         device, writer, epoch, clip_grad=None):
    """
    Train one epoch using vanilla single precision training with fixed bitwidth.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        quant_config: Quantization configuration dictionary
        augmenter: Data augmenter object
        apply_augmentation_fn: Function to apply augmentation
        device: Device to run training on
        writer: TensorBoard writer
        epoch: Current epoch number
        clip_grad: Gradient clipping value (optional)
    
    Returns:
        dict: Training metrics (loss, accuracy)
    """
    model.train()
    
    # Extract bitwidth (should be a single value)
    bitwidth_options = quant_config.get('bitwidth_options', [8])
    assert len(bitwidth_options) == 1, "For training_method: vanilla_single_precision_training, bitwidth_options should contain only one value"
    bitwidth = bitwidth_options[0] if isinstance(bitwidth_options, list) else bitwidth_options
    
    # Set fixed bitwidth for all layers at the start
    conv_class = get_conv_class_from_model(model)
    if conv_class is not None:
        for module in model.modules():
            if isinstance(module, conv_class):
                module.set_bitwidth(bitwidth)
    
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    logging.info(f"Training with fixed bitwidth: {bitwidth}")
    
    for batch_idx, batch_data in enumerate(train_loader):
        # Unpack batch
        if len(batch_data) == 3:
            data, labels, idx = batch_data
        else:
            data, labels = batch_data[0], batch_data[1]
        
        # Apply augmentation if provided
        if augmenter is not None and apply_augmentation_fn is not None:
            data, labels = apply_augmentation_fn(augmenter, data, labels)
        
        # Move to device
        labels = labels.to(device)
        if isinstance(data, dict):
            # Multi-modal data
            for loc in data:
                for mod in data[loc]:
                    data[loc][mod] = data[loc][mod].to(device)
        else:
            data = data.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)
        
        # Handle one-hot labels if needed
        if len(labels.shape) == 2 and labels.shape[1] > 1:
            loss_labels = torch.argmax(labels, dim=1)
        else:
            loss_labels = labels
        
        loss = loss_fn(outputs, loss_labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        # Calculate metrics
        train_loss += loss.item() * labels.size(0)
        predictions = torch.argmax(outputs, dim=1)
        if len(labels.shape) == 2 and labels.shape[1] > 1:
            labels_idx = torch.argmax(labels, dim=1)
        else:
            labels_idx = labels
        
        train_correct += (predictions == labels_idx).sum().item()
        train_total += labels.size(0)
        
        # Log to TensorBoard every 50 batches
        if batch_idx % 50 == 0 and writer is not None:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)
    
    # Calculate epoch metrics
    epoch_train_loss = train_loss / train_total
    epoch_train_acc = train_correct / train_total
    
    return {
        'loss': epoch_train_loss,
        'accuracy': epoch_train_acc
    }


def train_epoch_joint_quantization(model, train_loader, optimizer, loss_fn, 
                                   quant_config, augmenter, apply_augmentation_fn, 
                                   device, writer, epoch, clip_grad=None):
    """
    Train one epoch using joint quantization strategy.
    
    For each batch, multiple forward passes are performed with different random
    bitwidth configurations, and the losses are averaged.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        quant_config: Quantization configuration dictionary
        augmenter: Data augmenter object
        apply_augmentation_fn: Function to apply augmentation
        device: Device to run training on
        writer: TensorBoard writer
        epoch: Current epoch number
        clip_grad: Gradient clipping value (optional)
    
    Returns:
        dict: Training metrics (loss, accuracy)
    """
    model.train()
    
    # Extract joint quantization parameters
    joint_quant_config = quant_config.get('joint_quantization', {})
    joint_quantization_batch_size = joint_quant_config.get('joint_quantization_batch_size', 2)
    bitwidth_options = quant_config.get('bitwidth_options', [4, 6, 8])
    
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    logging.info(f"Training with joint quantization (batch_size={joint_quantization_batch_size})")
    
    for batch_idx, batch_data in enumerate(train_loader):
        # Unpack batch
        if len(batch_data) == 3:
            data, labels, idx = batch_data
        else:
            data, labels = batch_data[0], batch_data[1]
        
        # Apply augmentation if provided
        if augmenter is not None and apply_augmentation_fn is not None:
            data, labels = apply_augmentation_fn(augmenter, data, labels)
        
        # Move to device
        labels = labels.to(device)
        if isinstance(data, dict):
            # Multi-modal data
            for loc in data:
                for mod in data[loc]:
                    data[loc][mod] = data[loc][mod].to(device)
        else:
            data = data.to(device)
        
        # Joint quantization: accumulate loss over multiple forward passes
        accumulated_loss = 0.0
        
        for _ in range(joint_quantization_batch_size):
            # Set random bitwidths for all layers
            set_random_bitwidth_all_layers(model, bitwidth_options)
            
            # Forward pass
            outputs = model(data)
            
            # Handle one-hot labels if needed
            if len(labels.shape) == 2 and labels.shape[1] > 1:
                loss_labels = torch.argmax(labels, dim=1)
            else:
                loss_labels = labels
            
            loss = loss_fn(outputs, loss_labels)
            accumulated_loss += loss
        
        # Average the loss
        avg_loss = accumulated_loss / joint_quantization_batch_size
        
        # Backward pass
        optimizer.zero_grad()
        avg_loss.backward()
        
        # Gradient clipping
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        # Calculate metrics (using last outputs from loop)
        train_loss += avg_loss.item() * labels.size(0)
        predictions = torch.argmax(outputs, dim=1)
        if len(labels.shape) == 2 and labels.shape[1] > 1:
            labels_idx = torch.argmax(labels, dim=1)
        else:
            labels_idx = labels
        
        train_correct += (predictions == labels_idx).sum().item()
        train_total += labels.size(0)
        
        # Log to TensorBoard every 50 batches
        if batch_idx % 50 == 0 and writer is not None:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Batch_Loss', avg_loss.item(), global_step)
    
    # Calculate epoch metrics
    epoch_train_loss = train_loss / train_total
    epoch_train_acc = train_correct / train_total
    
    return {
        'loss': epoch_train_loss,
        'accuracy': epoch_train_acc
    }


def train_with_quantization(model, train_loader, val_loader, config, experiment_dir,
                            loss_fn, optimizer, scheduler, augmenter, apply_augmentation_fn):
    """
    Main training function for quantization-aware training.
    
    This function handles:
    - Setup of quantization layers
    - Training loop with appropriate training method (joint quantization, etc.)
    - Validation with appropriate validation function (random bitwidths, etc.)
    - Checkpointing and best model tracking
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Full configuration dictionary
        experiment_dir: Path to experiment directory
        loss_fn: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        augmenter: Data augmenter object
        apply_augmentation_fn: Function to apply augmentation
    
    Returns:
        model: Trained model
        train_history: Dictionary with training history
    """
    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    
    # Extract quantization configuration
    quantization_enabled = config.get('quantization', {}).get('enable', False)
    if not quantization_enabled:
        raise ValueError("Quantization is not enabled in config. Use standard train() function.")
    
    quantization_method = config.get('quantization_method', 'dorefa')
    logging.info(f"Using quantization method: {quantization_method}")
    
    # Get nested quantization config
    if quantization_method not in config['quantization']:
        raise ValueError(f"Quantization method '{quantization_method}' not found in config. "
                        f"Available methods: {list(config['quantization'].keys())}")
    
    quant_config = config['quantization'][quantization_method]
    
    # Setup quantization layers
    model = setup_quantization_layers(model, quant_config)
    model = model.to(device)
    
    # Extract training and validation methods
    training_method = quant_config.get('training_method', 'joint_quantization')
    validation_function = quant_config.get('validation_function', 'random_bitwidths')
    
    logging.info(f"Training method: {training_method}")
    logging.info(f"Validation function: {validation_function}")
    
    # Setup directories
    experiment_path = Path(experiment_dir)
    logs_dir = experiment_path / "logs"
    models_dir = experiment_path / "models"
    tensorboard_dir = experiment_path / "tensorboard"
    
    # Note: File logging is already set up in train.py, so we don't duplicate it here
    
    # Setup TensorBoard
    writer = SummaryWriter(str(tensorboard_dir))
    
    # Training parameters
    model_name = config.get('model', 'ResNet')
    num_epochs = config.get(model_name, {}).get('lr_scheduler', {}).get('train_epochs', 50)
    num_classes = config.get('vehicle_classification', {}).get('num_classes', 7)
    class_names = config.get('vehicle_classification', {}).get('class_names', None)
    clip_grad = config.get(model_name, {}).get('optimizer', {}).get('clip_grad', None)
    
    # Training history
    train_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_mean_acc': [],
        'val_std_acc': [],
        'learning_rates': [],
        'bitwidth_bin_stats': [],  # Store bin stats for each epoch
        'avg_val_std_history': []  # Store avg val std for each epoch
    }
    
    # Best model tracking
    best_val_acc = 0.0
    best_epoch = 0
    
    logging.info("=" * 80)
    logging.info("Starting Quantization-Aware Training")
    logging.info(f"Device: {device}")
    logging.info(f"Number of epochs: {num_epochs}")
    logging.info(f"Quantization method: {quantization_method}")
    logging.info(f"Training method: {training_method}")
    logging.info(f"Validation function: {validation_function}")
    logging.info("=" * 80)
    
    for epoch in range(num_epochs):
        # ====================================================================
        # Training Phase
        # ====================================================================
        logging.info(f"Epoch [{epoch+1}/{num_epochs}]")
        
        if training_method == "joint_quantization":
            train_metrics = train_epoch_joint_quantization(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                quant_config=quant_config,
                augmenter=augmenter,
                apply_augmentation_fn=apply_augmentation_fn,
                device=device,
                writer=writer,
                epoch=epoch,
                clip_grad=clip_grad
            )
        elif training_method == "vanilla_single_precision_training":
            train_metrics = train_epoch_vanilla_single_precision(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                quant_config=quant_config,
                augmenter=augmenter,
                apply_augmentation_fn=apply_augmentation_fn,
                device=device,
                writer=writer,
                epoch=epoch,
                clip_grad=clip_grad
            )
        else:
            raise ValueError(f"Unknown training method: {training_method}")
        
        epoch_train_loss = train_metrics['loss']
        epoch_train_acc = train_metrics['accuracy']
        
        train_history['train_loss'].append(epoch_train_loss)
        train_history['train_acc'].append(epoch_train_acc)
        
        # ====================================================================
        # Validation Phase
        # ====================================================================
        if validation_function == "random_bitwidths":
            # Get validation config from quantization.random_bitwidths
            val_config = config.get('quantization', {}).get('random_bitwidths', {})
            num_configs = val_config.get('number_of_configs', 4)
            bitwidth_options = val_config.get('bitwidth_options', [4, 6, 8])
            bin_tolerance = val_config.get('bin_tolerance', 0.5)  # Configurable tolerance
            
            val_stats = validate_random_bitwidths(
                model=model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                augmenter=augmenter,
                apply_augmentation_fn=apply_augmentation_fn,
                num_configs=num_configs,
                bitwidth_options=bitwidth_options,
                bin_tolerance=bin_tolerance
            )
            
            epoch_val_loss = val_stats['mean_loss']
            epoch_val_acc = val_stats['mean_acc']
            
            train_history['val_mean_acc'].append(val_stats['mean_acc'])
            train_history['val_std_acc'].append(val_stats['std_acc'])
            train_history['bitwidth_bin_stats'].append(val_stats.get('bin_stats', []))
            train_history['avg_val_std_history'].append(val_stats.get('avg_val_std', 0.0))
        elif validation_function == "simple_validation":
            # Simple validation with fixed bitwidth
            bitwidth_options = quant_config.get('bitwidth_options', [8])
            bitwidth = bitwidth_options[0] if isinstance(bitwidth_options, list) else bitwidth_options
            
            val_stats = validate_simple(
                model=model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                augmenter=augmenter,
                apply_augmentation_fn=apply_augmentation_fn,
                bitwidth=bitwidth
            )
            
            epoch_val_loss = val_stats['loss']
            epoch_val_acc = val_stats['accuracy']
        else:
            raise ValueError(f"Unknown validation function: {validation_function}")
        
        train_history['val_loss'].append(epoch_val_loss)
        train_history['val_acc'].append(epoch_val_acc)
        
        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        train_history['learning_rates'].append(current_lr)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # ====================================================================
        # Logging
        # ====================================================================
        logging.info(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        logging.info(f"  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        logging.info(f"  Learning Rate: {current_lr:.6f}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/val', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)
        writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Additional validation metrics for random_bitwidths
        if validation_function == "random_bitwidths":
            writer.add_scalar('Validation/mean_acc', val_stats['mean_acc'], epoch)
            writer.add_scalar('Validation/std_acc', val_stats['std_acc'], epoch)
            writer.add_scalar('Validation/min_acc', val_stats['min_acc'], epoch)
            writer.add_scalar('Validation/max_acc', val_stats['max_acc'], epoch)
            
            # Log bitwidth statistics
            writer.add_scalar('Bitwidth/mean_bitwidth', val_stats['mean_bitwidth'], epoch)
            writer.add_scalar('Bitwidth/std_bitwidth', val_stats['std_bitwidth'], epoch)
            writer.add_scalar('Bitwidth/min_bitwidth', val_stats['min_bitwidth'], epoch)
            writer.add_scalar('Bitwidth/max_bitwidth', val_stats['max_bitwidth'], epoch)
            
            # Create and log bitwidth vs accuracy plot (every 5 epochs or last epoch)
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                bitwidth_acc_fig = plot_bitwidth_vs_accuracy(
                    bitwidths=val_stats['all_bitwidths'],
                    accuracies=val_stats['all_accuracies'],
                    epoch=epoch,
                    title="Bitwidth vs Validation Accuracy",
                    bin_tolerance=bin_tolerance
                )
                writer.add_figure('Bitwidth_Analysis/bitwidth_vs_accuracy', bitwidth_acc_fig, epoch)
                plt.close(bitwidth_acc_fig)
                logging.info(f"  Bitwidth vs Accuracy plot logged to TensorBoard")
        
        # ====================================================================
        # Save Checkpoints
        # ====================================================================
        # Save best model (based on mean accuracy)
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            best_model_path = models_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'val_loss': epoch_val_loss,
                'config': config,
                'quantization_method': quantization_method
            }, best_model_path)
            logging.info(f"  Best model saved! (Val Acc: {best_val_acc:.4f})")
        
        # Save last epoch
        last_model_path = models_dir / "last_epoch.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': epoch_val_acc,
            'val_loss': epoch_val_loss,
            'config': config,
            'quantization_method': quantization_method
        }, last_model_path)
    
    # Final summary
    logging.info("=" * 80)
    logging.info("Training Complete!")
    logging.info(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch + 1}")
    logging.info(f"Models saved to: {models_dir}")
    logging.info(f"TensorBoard logs: {tensorboard_dir}")
    logging.info("=" * 80)
    
    writer.close()
    
    return model, train_history

