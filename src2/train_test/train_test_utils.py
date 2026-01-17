"""
Training and Testing Utilities

This module provides core training/testing functionality with:
- Experiment tracking and directory management
- Training loop with checkpointing and logging
- Testing function with flexible evaluation
- Metrics calculation (accuracy, confusion matrix)
- TensorBoard and text file logging
"""

import os
import logging
import yaml
import shutil
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter


# ============================================================================
# Optimizer and Scheduler Setup
# ============================================================================

def setup_optimizer(model, config):
    """
    Create optimizer based on configuration.
    
    Args:
        model: PyTorch model
        config: Configuration dictionary
    
    Returns:
        optimizer: Configured optimizer
    """
    model_name = config.get("model", "ResNet")
    optimizer_config = config.get(model_name, {}).get("optimizer", {})
    
    optimizer_name = optimizer_config.get("name", "AdamW")
    start_lr = optimizer_config.get("start_lr", 0.001)
    weight_decay = optimizer_config.get("weight_decay", 0.0)
    
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=start_lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=start_lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == "SGD":
        momentum = optimizer_config.get("momentum", 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=start_lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    logging.info(f"Optimizer created: {optimizer_name}")
    logging.info(f"  Learning rate: {start_lr}")
    logging.info(f"  Weight decay: {weight_decay}")
    
    return optimizer


def setup_scheduler(optimizer, config):
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary
    
    Returns:
        scheduler: Learning rate scheduler (or None)
    """
    model_name = config.get("model", "ResNet")
    scheduler_config = config.get(model_name, {}).get("lr_scheduler", {})
    
    scheduler_name = scheduler_config.get("name", "cosine")
    train_epochs = scheduler_config.get("train_epochs", 50)
    warmup_epochs = scheduler_config.get("warmup_epochs", 0)
    
    if scheduler_name == "cosine":
        # Cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_epochs - warmup_epochs,
            eta_min=scheduler_config.get("min_lr", 1e-6)
        )
        logging.info(f"Scheduler created: CosineAnnealingLR")
        logging.info(f"  Train epochs: {train_epochs}, Warmup epochs: {warmup_epochs}, Min LR: {scheduler_config.get('min_lr', 1e-6)}")
    
    elif scheduler_name == "step":
        # Step decay
        decay_epochs = scheduler_config.get("decay_epochs", 30)
        decay_rate = scheduler_config.get("decay_rate", 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=decay_epochs,
            gamma=decay_rate
        )
        logging.info(f"Scheduler created: StepLR")
        logging.info(f"  Step size: {decay_epochs}, Gamma: {decay_rate}")
    
    elif scheduler_name == "multistep":
        # Multi-step decay
        milestones = scheduler_config.get("milestones", [30, 60, 90])
        decay_rate = scheduler_config.get("decay_rate", 0.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=decay_rate
        )
        logging.info(f"Scheduler created: MultiStepLR")
        logging.info(f"  Milestones: {milestones}, Gamma: {decay_rate}")
    
    elif scheduler_name == "none" or scheduler_name is None:
        scheduler = None
        logging.info("No learning rate scheduler")
    
    else:
        logging.warning(f"Unknown scheduler: {scheduler_name}. Using no scheduler.")
        scheduler = None
    
    return scheduler


# ============================================================================
# Experiment Management
# ============================================================================

def create_experiment_id(model_name, model_variant=None):
    """
    Generate a unique experiment ID with timestamp and model information.
    
    Format: YYYYMMDD_HHMMSS_modelname_variant
    
    Args:
        model_name: Name of the model (e.g., "resnet", "deepsense")
        model_variant: Optional model variant (e.g., "resnet18", "resnet50")
    
    Returns:
        experiment_id: String identifier for this experiment
    
    Example:
        >>> create_experiment_id("resnet", "resnet18")
        "20231118_143052_resnet_resnet18"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if model_variant:
        experiment_id = f"{timestamp}_{model_name}_{model_variant}"
    else:
        experiment_id = f"{timestamp}_{model_name}"
    
    return experiment_id


def setup_experiment_dir(config):
    """
    Create experiment directory structure and save configuration.
    
    Structure:
        experiments/
        └── <experiment_id>/
            ├── config.yaml
            ├── models/
            ├── logs/
            └── tensorboard/
    
    Args:
        config: Configuration dictionary
        base_experiments_dir: Base directory for all experiments
    
    Returns:
        experiment_dir: Path to the created experiment directory
        tensorboard_dir: Path to tensorboard logs
    """
    # Create experiment ID
    base_experiments_dir = config.get("base_experiment_dir", "/home/misra8/sensing-nn/src2/experiments")
    model_name = config.get("model", "model")
    model_variant = config.get("model_variant", None)
    experiment_id = create_experiment_id(model_name, model_variant)
    
    # Create directory structure
    experiment_dir = Path(base_experiments_dir) / experiment_id
    models_dir = experiment_dir / "models"
    logs_dir = experiment_dir / "logs"
    tensorboard_dir = experiment_dir / "tensorboard"
    
    # Create directories
    experiment_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    tensorboard_dir.mkdir(exist_ok=True)
    
    # Save configuration
    config_path = experiment_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logging.info(f"Experiment directory created: {experiment_dir}")
    logging.info(f"  Experiment ID: {experiment_id}")
    
    return str(experiment_dir), str(tensorboard_dir)


# ============================================================================
# Metrics Functions
# ============================================================================

def calculate_accuracy(outputs, labels):
    """
    Calculate classification accuracy.
    
    Args:
        outputs: Model outputs (logits) of shape (batch_size, num_classes)
        labels: Ground truth labels of shape (batch_size,)
    
    Returns:
        accuracy: Accuracy as a float between 0 and 1
    """
    predictions = torch.argmax(outputs, dim=1)
    
    # Handle one-hot encoded labels
    if len(labels.shape) == 2 and labels.shape[1] > 1:
        labels = torch.argmax(labels, dim=1)
    
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    
    return accuracy


def calculate_confusion_matrix(all_predictions, all_labels, num_classes):
    """
    Calculate confusion matrix.
    
    Args:
        all_predictions: Numpy array or list of predicted class indices
        all_labels: Numpy array or list of true class indices
        num_classes: Number of classes
    
    Returns:
        cm: Confusion matrix as numpy array of shape (num_classes, num_classes)
    """
    cm = sklearn_confusion_matrix(all_labels, all_predictions, labels=range(num_classes))
    return cm


def plot_confusion_matrix(cm, class_names=None, normalize=False):
    """
    Create a matplotlib figure of the confusion matrix.
    
    Args:
        cm: Confusion matrix as numpy array
        class_names: List of class names (optional)
        normalize: Whether to normalize the confusion matrix
    
    Returns:
        fig: Matplotlib figure object
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                cmap='Blues', ax=ax, cbar=True)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    
    if class_names:
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names, rotation=0)
    
    plt.tight_layout()
    return fig


# ============================================================================
# Validation Function
# ============================================================================

def validate(model, val_loader, loss_fn, device, augmenter=None, apply_augmentation_fn=None):
    """
    Default validation function.
    
    Args:
        model: PyTorch model to validate
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to run validation on
        augmenter: Data augmenter object (optional)
        apply_augmentation_fn: Function to apply augmentation (optional)
    
    Returns:
        val_results: Dictionary with validation metrics
            - 'loss': float
            - 'accuracy': float
            - 'predictions': list
            - 'labels': list
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_val_preds = []
    all_val_labels = []
    
    with torch.no_grad():
        for batch_data in val_loader:
            # Unpack batch
            if len(batch_data) == 3:
                data, labels, idx = batch_data
            else:
                data, labels = batch_data[0], batch_data[1]
            
            # Apply augmentation if provided (for frequency transformation)
            if augmenter is not None and apply_augmentation_fn is not None:
                data, labels = apply_augmentation_fn(augmenter, data, labels)
            
            # Move to device
            labels = labels.to(device)
            if isinstance(data, dict):
                for loc in data:
                    for mod in data[loc]:
                        data[loc][mod] = data[loc][mod].to(device)
            else:
                data = data.to(device)
            
            # Forward pass
            outputs = model(data)
            
            # Handle one-hot labels
            if len(labels.shape) == 2 and labels.shape[1] > 1:
                loss_labels = torch.argmax(labels, dim=1)
            else:
                loss_labels = labels
            
            loss = loss_fn(outputs, loss_labels)
            
            val_loss += loss.item() * labels.size(0)
            predictions = torch.argmax(outputs, dim=1)
            val_correct += (predictions == loss_labels).sum().item()
            val_total += labels.size(0)
            
            all_val_preds.extend(predictions.cpu().numpy())
            all_val_labels.extend(loss_labels.cpu().numpy())
    
    epoch_val_loss = val_loss / val_total
    epoch_val_acc = val_correct / val_total
    
    return {
        'loss': epoch_val_loss,
        'accuracy': epoch_val_acc,
        'predictions': all_val_preds,
        'labels': all_val_labels
    }


# ============================================================================
# Training Function
# ============================================================================

def train(model, train_loader, val_loader, config, experiment_dir, 
          loss_fn=None, val_fn=None,
          augmenter=None, apply_augmentation_fn=None):
    """
    Train the model with comprehensive logging and checkpointing.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        experiment_dir: Path to experiment directory
        loss_fn: Loss function (if None, uses CrossEntropyLoss)
        val_fn: Custom validation function (optional)
        augmenter: Data augmenter object (optional)
        apply_augmentation_fn: Function to apply augmentation (optional)
    
    Returns:
        model: Trained model
        train_history: Dictionary with training history
    """
    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(optimizer, config)
    
    # Setup loss function
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    
    # Setup optimizer
    if optimizer is None:
        lr = config.get(config['model'], {}).get('optimizer', {}).get('start_lr', 0.001)
        weight_decay = config.get(config['model'], {}).get('optimizer', {}).get('weight_decay', 0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Setup directories
    experiment_path = Path(experiment_dir)
    logs_dir = experiment_path / "logs"
    models_dir = experiment_path / "models"
    tensorboard_dir = experiment_path / "tensorboard"
    
    # Setup logging
    log_file = logs_dir / "train.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger('train')
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    # Setup TensorBoard
    writer = SummaryWriter(str(tensorboard_dir))
    
    # Training parameters
    num_epochs = config.get(config['model'], {}).get('lr_scheduler', {}).get('train_epochs', 50)
    num_classes = config.get('vehicle_classification', {}).get('num_classes', 7)
    class_names = config.get('vehicle_classification', {}).get('class_names', None)
    
    # Training history
    train_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    # Best model tracking
    best_val_acc = 0.0
    best_epoch = 0
    
    logger.info("=" * 80)
    logger.info("Starting Training")
    logger.info(f"Device: {device}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info("=" * 80)
    
    for epoch in range(num_epochs):
        # ====================================================================
        # Training Phase
        # ====================================================================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []
        
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
            clip_grad = config.get(config['model'], {}).get('optimizer', {}).get('clip_grad', None)
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            optimizer.step()
            
            # Metrics
            train_loss += loss.item() * labels.size(0)
            predictions = torch.argmax(outputs, dim=1)
            if len(labels.shape) == 2 and labels.shape[1] > 1:
                labels_idx = torch.argmax(labels, dim=1)
            else:
                labels_idx = labels
            
            train_correct += (predictions == labels_idx).sum().item()
            train_total += labels.size(0)
            
            all_train_preds.extend(predictions.cpu().numpy())
            all_train_labels.extend(labels_idx.cpu().numpy())
        
        # Calculate epoch training metrics
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        train_history['train_loss'].append(epoch_train_loss)
        train_history['train_acc'].append(epoch_train_acc)
        
        # ====================================================================
        # Validation Phase
        # ====================================================================
        if val_fn is not None:
            # Use custom validation function
            val_results = val_fn(model, val_loader, loss_fn, device, config)
        else:
            # Use default validation function
            val_results = validate(model, val_loader, loss_fn, device, augmenter, apply_augmentation_fn)
        
        epoch_val_loss = val_results['loss']
        epoch_val_acc = val_results['accuracy']
        all_val_preds = val_results['predictions']
        all_val_labels = val_results['labels']
        
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
        logger.info(f"Epoch [{epoch+1}/{num_epochs}]")
        logger.info(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        logger.info(f"  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        logger.info(f"  Learning Rate: {current_lr:.6f}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/val', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)
        writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Confusion matrix logging (every 5 epochs or last epoch)
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            # Training confusion matrix
            train_cm = calculate_confusion_matrix(all_train_preds, all_train_labels, num_classes)
            train_cm_fig = plot_confusion_matrix(train_cm, class_names=class_names, normalize=True)
            writer.add_figure('Confusion_Matrix/train', train_cm_fig, epoch)
            plt.close(train_cm_fig)
            
            # Validation confusion matrix
            val_cm = calculate_confusion_matrix(all_val_preds, all_val_labels, num_classes)
            val_cm_fig = plot_confusion_matrix(val_cm, class_names=class_names, normalize=True)
            writer.add_figure('Confusion_Matrix/val', val_cm_fig, epoch)
            plt.close(val_cm_fig)
            
            logger.info(f"  Confusion matrices logged to TensorBoard")
        
        # ====================================================================
        # Save Checkpoints
        # ====================================================================
        # Save best model
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
                'config': config
            }, best_model_path)
            logger.info(f"  Best model saved! (Val Acc: {best_val_acc:.4f})")
        
        # Save last epoch
        last_model_path = models_dir / "last_epoch.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': epoch_val_acc,
            'val_loss': epoch_val_loss,
            'config': config
        }, last_model_path)
    
    # Final summary
    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch + 1}")
    logger.info(f"Models saved to: {models_dir}")
    logger.info(f"TensorBoard logs: {tensorboard_dir}")
    logger.info("=" * 80)
    
    writer.close()
    
    return model, train_history


# ============================================================================
# Testing Function
# ============================================================================

def test(model, test_loader, config, experiment_dir, checkpoint_path=None,
         loss_fn=None, test_fn=None, augmenter=None, apply_augmentation_fn=None):
    """
    Test the model and save results.
    
    Args:
        model: PyTorch model to test
        test_loader: Test data loader
        config: Configuration dictionary
        experiment_dir: Path to experiment directory
        checkpoint_path: Path to checkpoint file (optional, if None uses current model)
        loss_fn: Loss function (if None, uses CrossEntropyLoss)
        test_fn: Custom test function (optional)
        augmenter: Data augmenter for transformations (optional)
        apply_augmentation_fn: Function to apply augmentation (optional)
    
    Returns:
        test_results: Dictionary with test metrics
    """
    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded checkpoint from: {checkpoint_path}")
        if 'epoch' in checkpoint:
            logging.info(f"  Checkpoint epoch: {checkpoint['epoch']}")
        if 'val_acc' in checkpoint:
            logging.info(f"  Checkpoint val accuracy: {checkpoint['val_acc']:.4f}")
    
    model = model.to(device)
    
    # Setup loss function
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    
    # Setup logging
    experiment_path = Path(experiment_dir)
    logs_dir = experiment_path / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    log_file = logs_dir / "test_results.txt"
    
    num_classes = config.get('vehicle_classification', {}).get('num_classes', 7)
    class_names = config.get('vehicle_classification', {}).get('class_names', None)
    
    # Use custom test function if provided
    if test_fn is not None:
        test_results = test_fn(model, test_loader, loss_fn, device, config)
        test_loss = test_results['loss']
        test_acc = test_results['accuracy']
        all_preds = test_results['predictions']
        all_labels = test_results['labels']
    else:
        # Default testing
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                # Unpack batch
                if len(batch_data) == 3:
                    data, labels, idx = batch_data
                else:
                    data, labels = batch_data[0], batch_data[1]
                
                # Apply augmentation if provided (for frequency transformation)
                if augmenter is not None and apply_augmentation_fn is not None:
                    data, labels = apply_augmentation_fn(augmenter, data, labels)
                
                # Move to device
                labels = labels.to(device)
                if isinstance(data, dict):
                    for loc in data:
                        for mod in data[loc]:
                            data[loc][mod] = data[loc][mod].to(device)
                else:
                    data = data.to(device)
                
                # Forward pass
                outputs = model(data)
                
                # Handle one-hot labels
                if len(labels.shape) == 2 and labels.shape[1] > 1:
                    loss_labels = torch.argmax(labels, dim=1)
                else:
                    loss_labels = labels
                
                loss = loss_fn(outputs, loss_labels)
                
                test_loss += loss.item() * labels.size(0)
                predictions = torch.argmax(outputs, dim=1)
                test_correct += (predictions == loss_labels).sum().item()
                test_total += labels.size(0)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(loss_labels.cpu().numpy())
        
        test_loss = test_loss / test_total
        test_acc = test_correct / test_total
    
    # Calculate confusion matrix
    cm = calculate_confusion_matrix(all_preds, all_labels, num_classes)
    
    # Calculate per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Save results to file
    with open(log_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TEST RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        
        f.write("Per-Class Accuracy:\n")
        for i, acc in enumerate(per_class_acc):
            class_name = class_names[i] if class_names else f"Class {i}"
            f.write(f"  {class_name}: {acc:.4f}\n")
        
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        
        if checkpoint_path:
            f.write(f"Checkpoint: {checkpoint_path}\n")
        
        f.write("=" * 80 + "\n")
    
    # Save confusion matrix plot
    cm_fig = plot_confusion_matrix(cm, class_names=class_names, normalize=True)
    cm_fig.savefig(logs_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close(cm_fig)
    
    # Print results
    logging.info("=" * 80)
    logging.info("TEST RESULTS")
    logging.info("=" * 80)
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test Accuracy: {test_acc:.4f}")
    logging.info(f"Results saved to: {log_file}")
    logging.info(f"Confusion matrix saved to: {logs_dir / 'confusion_matrix.png'}")
    logging.info("=" * 80)
    
    # Return results
    test_results = {
        'loss': test_loss,
        'accuracy': test_acc,
        'confusion_matrix': cm,
        'per_class_accuracy': per_class_acc,
        'predictions': all_preds,
        'labels': all_labels
    }
    
    return test_results

