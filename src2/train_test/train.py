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
import yaml
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
from train_test.normalize import setup_normalization

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
    # 2b. Setup Normalization
    # ========================================================================
    logging.info("\nSetting up normalization...")
    train_loader, val_loader, test_loader = setup_normalization(
        train_loader, val_loader, test_loader, config
    )
    logging.info("Normalization setup complete")
    
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
    # 7. Log Hyperparameters to TensorBoard
    # ========================================================================
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(tensorboard_dir))
    
    # Prepare hyperparameters for logging
    hparams = {
        'model': config.get('model', 'Unknown'),
        'model_variant': config.get('model_variant', 'None'),
        'dataset': Path(config.get('yaml_path', 'Unknown')).stem,
        'batch_size': config.get('batch_size', 'Unknown'),
        'learning_rate': optimizer.param_groups[0]['lr'],
        'weight_decay': config.get(model_name, {}).get('optimizer', {}).get('weight_decay', 0.0),
        'optimizer': config.get(model_name, {}).get('optimizer', {}).get('name', 'Unknown'),
        'scheduler': config.get(model_name, {}).get('lr_scheduler', {}).get('name', 'Unknown'),
        'loss_function': config.get('loss_name', 'cross_entropy'),
        'num_epochs': config.get(model_name, {}).get('lr_scheduler', {}).get('train_epochs', 50),
    }
    
    # Add quantization info if enabled
    quantization_enabled = config.get("quantization", {}).get("enable", False)
    if quantization_enabled:
        quantization_method = config.get('quantization_method', 'Unknown')
        quant_config = config.get('quantization', {}).get(quantization_method, {})
        
        hparams['quantization_enabled'] = 'True'
        hparams['quantization_method'] = quantization_method
        
        # Get bitwidth options from the quantization method config
        bitwidth_options = quant_config.get('bitwidth_options', [])
        hparams['bitwidth_options'] = str(bitwidth_options)
        
        # Add other quantization details
        hparams['training_method'] = quant_config.get('training_method', 'Unknown')
        hparams['validation_function'] = quant_config.get('validation_function', 'Unknown')
        hparams['weight_quantization'] = quant_config.get('weight_quantization', 'Unknown')
        hparams['activation_quantization'] = quant_config.get('activation_quantization', 'Unknown')
    else:
        hparams['quantization_enabled'] = 'False'
        hparams['quantization_method'] = 'None'
        hparams['bitwidth_options'] = 'N/A'
    
    # Log hyperparameters to TensorBoard (will be updated with metrics after training)
    logging.info("\nLogging hyperparameters to TensorBoard...")
    for key, value in hparams.items():
        logging.info(f"  {key}: {value}")
    
    # ========================================================================
    # 8. Train Model
    # ========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("STARTING TRAINING")
    logging.info("=" * 80 + "\n")
    
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
        
        # ====================================================================
        # Log final metrics to TensorBoard hyperparameters
        # ====================================================================
        final_train_acc = train_history['train_acc'][-1] if train_history['train_acc'] else 0.0
        final_val_acc = train_history['val_acc'][-1] if train_history['val_acc'] else 0.0
        best_val_acc = max(train_history['val_acc']) if train_history['val_acc'] else 0.0
        
        # Update hparams with final metrics
        metrics = {
            'hparam/final_train_acc': final_train_acc,
            'hparam/final_val_acc': final_val_acc,
            'hparam/best_val_acc': best_val_acc,
        }
        
        writer.add_hparams(hparams, metrics)
        writer.close()
        
        logging.info(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
        logging.info(f"Final Validation Accuracy: {final_val_acc:.4f}")
        logging.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
        
        # ====================================================================
        # Save Experiment Summary
        # ====================================================================
        summary_file = Path(experiment_dir) / "experiment_summary.yaml"
        summary = {
            'experiment_id': Path(experiment_dir).name,
            'model': hparams['model'],
            'model_variant': hparams['model_variant'],
            'dataset': hparams['dataset'],
            'quantization_enabled': hparams['quantization_enabled'],
            'quantization_method': hparams['quantization_method'],
            'bitwidth_options': hparams['bitwidth_options'],
            'training_method': hparams.get('training_method', 'N/A'),
            'validation_function': hparams.get('validation_function', 'N/A'),
            'num_epochs': hparams['num_epochs'],
            'batch_size': hparams['batch_size'],
            'learning_rate': hparams['learning_rate'],
            'optimizer': hparams['optimizer'],
            'scheduler': hparams['scheduler'],
            'loss_function': hparams['loss_function'],
            'final_train_accuracy': float(final_train_acc),
            'final_val_accuracy': float(final_val_acc),
            'best_val_accuracy': float(best_val_acc),
            'training_status': 'completed',
        }
        
        # Add bitwidth bin statistics if using random_bitwidths validation
        if (quantization_enabled and 
            hparams.get('validation_function') == 'random_bitwidths' and
            'bitwidth_bin_stats' in train_history and 
            train_history['bitwidth_bin_stats']):
            # Get the final epoch's bin stats
            final_bin_stats = train_history['bitwidth_bin_stats'][-1]
            summary['bitwidth_bin_stats'] = final_bin_stats
            
            # Calculate and add average validation std
            if 'avg_val_std_history' in train_history and train_history['avg_val_std_history']:
                summary['avg_val_std'] = float(train_history['avg_val_std_history'][-1])
            else:
                # Fallback: calculate from bin stats
                if final_bin_stats:
                    avg_val_std = sum(b['std_acc'] for b in final_bin_stats) / len(final_bin_stats)
                    summary['avg_val_std'] = float(avg_val_std)
                else:
                    summary['avg_val_std'] = 0.0
            
            logging.info(f"Bitwidth bin statistics added to summary ({len(final_bin_stats)} bins)")
            logging.info(f"Average validation std: {summary['avg_val_std']:.4f}")
        else:
            summary['avg_val_std'] = 'N/A'
        
        with open(summary_file, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logging.info(f"\nExperiment summary saved to: {summary_file}")
        
        logging.info(f"\nExperiment directory: {experiment_dir}")
        logging.info(f"TensorBoard logs: {tensorboard_dir}")
        logging.info("\nTo view training logs in TensorBoard, run:")
        logging.info(f"  tensorboard --logdir={tensorboard_dir}")
        logging.info("=" * 80)
        
    except KeyboardInterrupt:
        logging.info("\n" + "=" * 80)
        logging.warning("Training interrupted by user")
        logging.info("=" * 80)
        
        # Save partial summary
        summary_file = Path(experiment_dir) / "experiment_summary.yaml"
        final_train_acc = train_history['train_acc'][-1] if train_history.get('train_acc') else 0.0
        final_val_acc = train_history['val_acc'][-1] if train_history.get('val_acc') else 0.0
        best_val_acc = max(train_history['val_acc']) if train_history.get('val_acc') else 0.0
        
        summary = {
            'experiment_id': Path(experiment_dir).name,
            'model': hparams['model'],
            'model_variant': hparams['model_variant'],
            'dataset': hparams['dataset'],
            'quantization_enabled': hparams['quantization_enabled'],
            'quantization_method': hparams['quantization_method'],
            'bitwidth_options': hparams['bitwidth_options'],
            'training_method': hparams.get('training_method', 'N/A'),
            'validation_function': hparams.get('validation_function', 'N/A'),
            'num_epochs': hparams['num_epochs'],
            'batch_size': hparams['batch_size'],
            'learning_rate': hparams['learning_rate'],
            'optimizer': hparams['optimizer'],
            'scheduler': hparams['scheduler'],
            'loss_function': hparams['loss_function'],
            'final_train_accuracy': float(final_train_acc),
            'final_val_accuracy': float(final_val_acc),
            'best_val_accuracy': float(best_val_acc),
            'training_status': 'interrupted',
        }
        
        # Add bitwidth bin statistics if available
        if (quantization_enabled and 
            hparams.get('validation_function') == 'random_bitwidths' and
            train_history.get('bitwidth_bin_stats') and 
            train_history['bitwidth_bin_stats']):
            final_bin_stats = train_history['bitwidth_bin_stats'][-1]
            summary['bitwidth_bin_stats'] = final_bin_stats
            if train_history.get('avg_val_std_history'):
                summary['avg_val_std'] = float(train_history['avg_val_std_history'][-1])
            else:
                summary['avg_val_std'] = 'N/A'
        else:
            summary['avg_val_std'] = 'N/A'
        
        with open(summary_file, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        writer.close()
        
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
        
        # Save error summary
        summary_file = Path(experiment_dir) / "experiment_summary.yaml"
        summary = {
            'experiment_id': Path(experiment_dir).name,
            'model': hparams['model'],
            'model_variant': hparams['model_variant'],
            'dataset': hparams['dataset'],
            'quantization_enabled': hparams['quantization_enabled'],
            'quantization_method': hparams['quantization_method'],
            'bitwidth_options': hparams['bitwidth_options'],
            'training_method': hparams.get('training_method', 'N/A'),
            'validation_function': hparams.get('validation_function', 'N/A'),
            'num_epochs': hparams['num_epochs'],
            'batch_size': hparams['batch_size'],
            'learning_rate': hparams['learning_rate'],
            'optimizer': hparams['optimizer'],
            'scheduler': hparams['scheduler'],
            'loss_function': hparams['loss_function'],
            'final_train_accuracy': 0.0,
            'final_val_accuracy': 0.0,
            'best_val_accuracy': 0.0,
            'training_status': 'failed',
            'error_message': str(e),
            'avg_val_std': 'N/A',
        }
        
        with open(summary_file, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        writer.close()
        
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

