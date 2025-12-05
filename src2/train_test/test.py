"""
Testing Script

This script tests a trained model with two modes:
1. Load a specific checkpoint via --checkpoint_path
2. Auto-find the latest experiment with --auto_latest

Usage:
    # Test with specific checkpoint
    python test.py --checkpoint_path /path/to/checkpoint.pth
    
    # Auto-find latest experiment
    python test.py --auto_latest --model resnet --yaml_path /path/to/config.yaml
    
    # Test specific checkpoint with custom test function
    python test.py --checkpoint_path /path/to/checkpoint.pth --use_best
"""

import sys
import logging
import argparse
import torch
import yaml
from pathlib import Path

# Add src2 to path for imports
src2_path = Path(__file__).parent.parent
sys.path.insert(0, str(src2_path))

from dataset_utils.parse_args_utils import get_config, load_yaml_config
from dataset_utils.MultiModalDataLoader import create_dataloaders
from data_augmenter import create_augmenter, apply_augmentation
from models.create_models import create_model
from train_test.loss import get_loss_function
from train_test.train_test_utils import test

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def parse_test_args():
    """
    Parse command line arguments specific to testing.
    
    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Test a trained model'
    )
    
    # Checkpoint loading modes
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        '--checkpoint_path',
        type=str,
        help='Path to specific checkpoint file to test'
    )
    mode_group.add_argument(
        '--auto_latest',
        action='store_true',
        help='Automatically find and test the most recent experiment'
    )
    
    # Model selection for auto_latest mode
    parser.add_argument(
        '--use_best',
        action='store_true',
        default=True,
        help='Use best model checkpoint (default: True). If False, uses last epoch.'
    )
    
    # Standard arguments (needed for auto_latest or if no checkpoint specified)
    parser.add_argument(
        '--model',
        type=str,
        help='Model name (e.g., ResNet, DeepSense)'
    )
    
    parser.add_argument(
        '--model_variant',
        type=str,
        help='Model variant (e.g., resnet18, resnet50)'
    )
    
    parser.add_argument(
        '--yaml_path',
        type=str,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU to use'
    )
    
    parser.add_argument(
        '--experiments_dir',
        type=str,
        default='/home/misra8/sensing-nn/src2/experiments',
        help='Base directory for experiments'
    )
    
    return parser.parse_args()


def find_latest_experiment(experiments_dir):
    """
    Find the most recent experiment directory.
    
    Args:
        experiments_dir: Base directory containing experiments
    
    Returns:
        experiment_path: Path to the latest experiment directory
    """
    experiments_path = Path(experiments_dir)
    
    if not experiments_path.exists():
        raise FileNotFoundError(f"Experiments directory not found: {experiments_dir}")
    
    # Get all experiment directories (format: YYYYMMDD_HHMMSS_*)
    experiment_dirs = [d for d in experiments_path.iterdir() if d.is_dir()]
    
    if not experiment_dirs:
        raise FileNotFoundError(f"No experiments found in: {experiments_dir}")
    
    # Sort by directory name (which includes timestamp)
    experiment_dirs.sort(reverse=True)
    latest_experiment = experiment_dirs[0]
    
    logging.info(f"Found latest experiment: {latest_experiment.name}")
    
    return latest_experiment


def load_config_from_experiment(experiment_dir):
    """
    Load configuration from an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
    
    Returns:
        config: Configuration dictionary
    """
    config_path = Path(experiment_dir) / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found in experiment: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Loaded config from: {config_path}")
    
    return config


def get_checkpoint_path(experiment_dir, use_best=True):
    """
    Get the path to a checkpoint in an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        use_best: If True, return best_model.pth; else last_epoch.pth
    
    Returns:
        checkpoint_path: Path to the checkpoint file
    """
    models_dir = Path(experiment_dir) / "models"
    
    if use_best:
        checkpoint_path = models_dir / "best_model.pth"
    else:
        checkpoint_path = models_dir / "last_epoch.pth"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return checkpoint_path


def main():
    """Main testing function."""
    
    # ========================================================================
    # 1. Parse Arguments and Setup
    # ========================================================================
    logging.info("=" * 80)
    logging.info("TESTING SCRIPT")
    logging.info("=" * 80)
    
    test_args = parse_test_args()
    
    # Determine mode and setup paths
    if test_args.checkpoint_path:
        # Mode 1: Specific checkpoint provided
        logging.info("Mode: Testing specific checkpoint")
        checkpoint_path = Path(test_args.checkpoint_path)
        
        if not checkpoint_path.exists():
            logging.error(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        
        # Infer experiment directory from checkpoint path
        # Expected structure: experiment_dir/models/checkpoint.pth
        experiment_dir = checkpoint_path.parent.parent
        
        # Load config from experiment
        config = load_config_from_experiment(experiment_dir)
        
    elif test_args.auto_latest:
        # Mode 2: Auto-find latest experiment
        logging.info("Mode: Auto-finding latest experiment")
        
        experiment_dir = find_latest_experiment(test_args.experiments_dir)
        
        # Load config from experiment
        config = load_config_from_experiment(experiment_dir)
        
        # Get checkpoint path
        checkpoint_path = get_checkpoint_path(experiment_dir, use_best=test_args.use_best)
        
    else:
        # Mode 3: Use provided config, no checkpoint loading (test current model)
        logging.info("Mode: Testing with provided config (no checkpoint)")
        
        if not test_args.model or not test_args.yaml_path:
            logging.error("When not using --checkpoint_path or --auto_latest, "
                        "you must provide --model and --yaml_path")
            sys.exit(1)
        
        # Load config from arguments
        config = load_yaml_config(test_args.yaml_path)
        config['model'] = test_args.model
        config['yaml_path'] = test_args.yaml_path
        config['device'] = f'cuda:{test_args.gpu}'
        if test_args.model_variant:
            config['model_variant'] = test_args.model_variant
        
        # No checkpoint to load
        checkpoint_path = None
        
        # Create a temporary experiment directory for results
        from train_test.train_test_utils import setup_experiment_dir
        experiment_dir, _ = setup_experiment_dir(config)
    
    # Update device if GPU specified
    if test_args.gpu is not None:
        config['device'] = f'cuda:{test_args.gpu}'
    
    logging.info(f"Experiment directory: {experiment_dir}")
    if checkpoint_path:
        logging.info(f"Checkpoint: {checkpoint_path}")
    logging.info(f"Device: {config.get('device', 'cpu')}")
    
    # ========================================================================
    # 2. Create Dataloaders (test set only)
    # ========================================================================
    logging.info("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config=config)
    logging.info(f"  Test batches: {len(test_loader)}")
    
    # ========================================================================
    # 3. Create Augmenter
    # ========================================================================
    logging.info("\nCreating augmenter...")
    # Create augmenter for data transformation (time -> frequency domain)
    augmenter = create_augmenter(config, augmentation_mode="fixed")
    logging.info("Augmenter created successfully")
    
    # ========================================================================
    # 4. Create Model
    # ========================================================================
    logging.info("\nCreating model...")
    model = create_model(config)
    
    # ========================================================================
    # 5. Setup Loss Function
    # ========================================================================
    loss_fn = get_loss_function(config=config)
    
    # ========================================================================
    # 6. Run Testing
    # ========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("STARTING TESTING")
    logging.info("=" * 80 + "\n")
    
    try:
        test_results = test(
            model=model,
            test_loader=test_loader,
            config=config,
            experiment_dir=experiment_dir,
            checkpoint_path=checkpoint_path,
            loss_fn=loss_fn,
            test_fn=None,  # Use default test function
            augmenter=augmenter,
            apply_augmentation_fn=apply_augmentation
        )
        
        logging.info("\n" + "=" * 80)
        logging.info("TESTING COMPLETED SUCCESSFULLY!")
        logging.info("=" * 80)
        logging.info(f"Test Loss: {test_results['loss']:.4f}")
        logging.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
        logging.info(f"Results saved to: {experiment_dir}/logs/test_results.txt")
        logging.info("=" * 80)
        
    except KeyboardInterrupt:
        logging.info("\n" + "=" * 80)
        logging.warning("Testing interrupted by user")
        logging.info("=" * 80)
        sys.exit(0)
    
    except Exception as e:
        logging.error("\n" + "=" * 80)
        logging.error("ERROR DURING TESTING")
        logging.error("=" * 80)
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

