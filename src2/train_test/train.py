"""
Training Script

This script orchestrates the training process:
1. Parse configuration and command-line arguments
2. Create dataloaders
3. Create model and augmenter
4. Setup experiment directory and logging
5. Initialize optimizer and scheduler
6. Train the model with checkpointing and logging
"""

import sys
import logging
import torch
from pathlib import Path

# Add src2 to path for imports
src2_path = Path(__file__).parent.parent
sys.path.insert(0, str(src2_path))

from dataset_utils.parse_args_utils import get_config
from dataset_utils.MultiModalDataLoader import create_dataloaders
from data_augmenter import create_augmenter, apply_augmentation
from models.create_models import create_model
from train_test.loss import get_loss_function
from train_test.train_test_utils import setup_experiment_dir, train

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


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


def main():
    """Main training function."""
    
    # ========================================================================
    # 1. Load Configuration
    # ========================================================================
    logging.info("=" * 80)
    logging.info("TRAINING SCRIPT")
    logging.info("=" * 80)
    
    config = get_config()
    logging.info("Configuration loaded successfully")
    logging.info(f"  Model: {config.get('model', 'Unknown')}")
    logging.info(f"  Model variant: {config.get('model_variant', 'None')}")
    logging.info(f"  Dataset: {config.get('yaml_path', 'Unknown')}")
    logging.info(f"  Device: {config.get('device', 'cpu')}")
    
    # ========================================================================
    # 2. Create Dataloaders
    # ========================================================================
    logging.info("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config=config)
    logging.info("  Train batches: {}".format(len(train_loader)))
    logging.info("  Val batches: {}".format(len(val_loader)))
    logging.info("  Test batches: {}".format(len(test_loader)))
    
    # ========================================================================
    # 3. Create Augmenter
    # ========================================================================
    logging.info("\nCreating augmenter...")
    # Get augmentation mode from config or use default
    model_name = config.get("model", "ResNet")
    augmenters_config = config.get(model_name, {}).get("fixed_augmenters", {})
    
    # For training, we typically want augmentation enabled
    augmenter = create_augmenter(config, augmentation_mode="fixed")
    logging.info("Augmenter created successfully")
    
    # ========================================================================
    # 4. Create Model
    # ========================================================================
    logging.info("\nCreating model...")
    model = create_model(config)
    
    # ========================================================================
    # 5. Setup Experiment Directory
    # ========================================================================
    logging.info("\nSetting up experiment directory...")
    experiment_dir, tensorboard_dir = setup_experiment_dir(config)
    
    # ========================================================================
    # 5b. Setup File Logging (as early as possible after experiment dir)
    # ========================================================================
    from pathlib import Path
    logs_dir = Path(experiment_dir) / "logs"
    
    # Determine log file name based on whether quantization is enabled
    quantization_enabled = config.get("quantization", {}).get("enable", False)
    log_filename = "train_quantization.log" if quantization_enabled else "train.log"
    log_file = logs_dir / log_filename
    
    # Add file handler to root logger so all logging goes to file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logging.info(f"Logging to file: {log_file}")
    
    # Now log the command line (after file logging is set up)
    import sys
    command_line = " ".join(sys.argv)
    logging.info("")
    logging.info("Command line used to run this script:")
    logging.info(f"  {command_line}")
    
    # Display TensorBoard command (after file logging is set up)
    logging.info("")
    logging.info("=" * 80)
    logging.info("TENSORBOARD")
    logging.info("=" * 80)
    logging.info("To monitor training in real-time, run this command in a separate terminal:")
    logging.info(f"  tensorboard --logdir={tensorboard_dir}")
    logging.info("=" * 80)
    
    # ========================================================================
    # 6. Setup Training Components
    # ========================================================================
    logging.info("\nSetting up training components...")
    
    # Loss function
    loss_fn = get_loss_function(config=config)
    
    # Optimizer
    optimizer = setup_optimizer(model, config)
    
    # Scheduler
    scheduler = setup_scheduler(optimizer, config)
    
    # ========================================================================
    # 7. Train Model
    # ========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("STARTING TRAINING")
    logging.info("=" * 80 + "\n")
    
    # Check if quantization is enabled
    quantization_enabled = config.get("quantization", {}).get("enable", False)
    
    try:
        if quantization_enabled:
            # Use quantization-aware training
            from train_test.quantization_train_test_utils import train_with_quantization
            
            quantization_method = config.get("quantization_method")
            if quantization_method not in config.get("quantization", {}):
                raise ValueError(f"Quantization method '{quantization_method}' not found in config. "
                               f"Available methods: {list(config.get('quantization', {}).keys())}")
            
            logging.info(f"Using quantization-aware training with method: {quantization_method}")
            
            model, train_history = train_with_quantization(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                experiment_dir=experiment_dir,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                augmenter=augmenter,
                apply_augmentation_fn=apply_augmentation
            )
        else:
            # Use standard training
            logging.info("Using standard training (quantization disabled)")
            
            model, train_history = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                experiment_dir=experiment_dir,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                val_fn=None,  # Use default validation
                augmenter=augmenter,
                apply_augmentation_fn=apply_augmentation
            )
        
        logging.info("\n" + "=" * 80)
        logging.info("TRAINING COMPLETED SUCCESSFULLY!")
        logging.info("=" * 80)
        logging.info(f"Experiment directory: {experiment_dir}")
        logging.info(f"TensorBoard logs: {tensorboard_dir}")
        logging.info("\nTo view training logs in TensorBoard, run:")
        logging.info(f"  tensorboard --logdir={tensorboard_dir}")
        logging.info("=" * 80)
        
    except KeyboardInterrupt:
        logging.info("\n" + "=" * 80)
        logging.warning("Training interrupted by user")
        logging.info("=" * 80)
        logging.info(f"Experiment directory: {experiment_dir}")
        logging.info(f"TensorBoard logs: {tensorboard_dir}")
        logging.info("\nTo view partial training logs in TensorBoard, run:")
        logging.info(f"  tensorboard --logdir={tensorboard_dir}")
        logging.info("=" * 80)
        sys.exit(0)
    
    except Exception as e:
        logging.error("\n" + "=" * 80)
        logging.error("ERROR DURING TRAINING")
        logging.error("=" * 80)
        logging.error(f"Error: {e}")
        logging.error(f"\nCommand that failed:")
        command_line = " ".join(sys.argv)
        logging.error(f"  {command_line}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

