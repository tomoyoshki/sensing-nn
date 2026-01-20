"""
Testing Script

This script tests a trained model with multiple modes:
1. Load a specific checkpoint via --checkpoint_path
2. Auto-find the latest experiment with --auto_latest
3. Test from an experiment directory with --experiments_dir

Usage:
    # Mode 1: Test with specific checkpoint
    python test.py --checkpoint_path /path/to/checkpoint.pth --test_function float
    
    # Mode 2: Auto-find latest experiment (commented out in current version)
    # python test.py --auto_latest --model resnet --yaml_path /path/to/config.yaml --test_function float
    
    # Mode 3: Test from experiment directory (uses best model by default)
    python test.py --experiment_dir /path/to/experiment --test_function float
    
    # Mode 3a: Test specific checkpoint by epoch number
    python test.py --experiment_dir /path/to/experiment --run_checkpoint 10 --test_function float
    
    # Mode 3b: Test multiple specific checkpoints by epoch numbers
    python test.py --experiment_dir /path/to/experiment --run_checkpoint 10 20 30 --test_function float
    
    # Mode 3c: Test all checkpoints in the experiment
    python test.py --experiment_dir /path/to/experiment --run_all_checkpoints --test_function float
"""

import sys
import logging
import argparse
import torch
import yaml
from pathlib import Path
import datetime

# Add src2 to path for imports
src2_path = Path(__file__).parent.parent
sys.path.insert(0, str(src2_path))

from dataset_utils.parse_args_utils import get_config, load_yaml_config, parse_test_args
from dataset_utils.MultiModalDataLoader import create_dataloaders
from data_augmenter import create_augmenter, apply_augmentation
from models.create_models import create_model
from train_test.loss import get_loss_function
from train_test.train_test_utils import test
from train_test.quantization_train_test_utils import setup_quantization_for_testing
from train_test.quantization_test_functions import (
    test_float,
    test_simple,
    test_random_bitwidths,
    load_and_test_float,
    load_and_test_single_precision,
    load_and_test_random_bitwidths
)
from train_test.normalize import setup_normalization

from train_test.quantization_test_utils import (
    find_latest_experiment,
    find_all_epoch_checkpoints,
    get_checkpoint_by_epoch,
    load_config_from_experiment,
    get_checkpoint_path,
    setup_test_run_directory,
    setup_checkpoint_from_path,
    setup_checkpoints_from_experiment_dir,
    setup_test_function_directory,
    save_single_precision_results_to_csv,
    validate_test_args
)

# Configure logging (console only initially, file handler added later)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Only console handler initially
)


def main():
    """Main testing function."""
    
    # ========================================================================
    # 1. Parse Arguments and Setup
    # ========================================================================
    logging.info("=" * 80)
    logging.info("TESTING SCRIPT")
    logging.info("=" * 80)
    
    test_args = parse_test_args()
    
    # Validate argument combinations
    validate_test_args(test_args)
    # Determine mode and setup paths
    try:
        if test_args.checkpoint_path:
            # Mode 1: Specific checkpoint provided
            experiment_dir, checkpoint_paths, config = setup_checkpoint_from_path(
                test_args
            )
        
        elif test_args.experiments_dir:
            # Mode 2: Experiment directory with checkpoint selection
            experiment_dir, checkpoint_paths, config = setup_checkpoints_from_experiment_dir(
                test_args
            )
        else:
            raise ValueError("Must provide either --checkpoint_path or --experiments_dir")
    
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(1)
    
    # Update device if GPU specified
    if test_args.gpu is not None:
        config['device'] = f'cuda:{test_args.gpu}'
    
    logging.info(f"Experiment directory: {experiment_dir}")
    logging.info(f"Number of checkpoints to test: {len(checkpoint_paths)}")
    for cp in checkpoint_paths:
        logging.info(f"  - {cp.name}")
    logging.info(f"Device: {config.get('device', 'cpu')}")
    
    # ========================================================================
    # 2. Create Dataloaders (test set only) - Shared across all checkpoints
    # ========================================================================
    logging.info("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config=config)
    logging.info(f"  Test batches: {len(test_loader)}")

    logging.info("\nSetting up normalization...")
    train_loader, val_loader, test_loader = setup_normalization(
        train_loader, val_loader, test_loader, config
    )
    logging.info("Normalization setup complete")
    
    # ========================================================================
    # 3. Create Augmenter - Shared across all checkpoints
    # ========================================================================
    logging.info("\nCreating augmenter...")
    # Create augmenter for data transformation (time -> frequency domain)
    augmenter = create_augmenter(config, augmentation_mode="fixed")
    logging.info("Augmenter created successfully")
    
    # ========================================================================
    # 4. Setup Loss Function - Shared across all checkpoints
    # ========================================================================
    # Create a temporary model for loss function setup
    temp_model = create_model(config)
    loss_fn, loss_fn_name = get_loss_function(config=config, model=temp_model)
    logging.info(f"Loss function: {loss_fn_name}")
    del temp_model
    
    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    
    # ========================================================================
    # 5. Loop Through Checkpoints
    # ========================================================================
    all_results = {}
    
    model = create_model(config)
    
    # ========================================================================
    # 5. Setup Quantization (if applicable) - ONCE for all checkpoints
    # ========================================================================
    logging.info("\nSetting up model for testing...")
    model = setup_quantization_for_testing(model, config, test_args.test_function, device)
    
    # ========================================================================
    # 6. Loop Through Checkpoints
    # ========================================================================
    for checkpoint_idx, checkpoint_path in enumerate(checkpoint_paths):
        logging.info("\n" + "=" * 80)
        logging.info(f"TESTING CHECKPOINT {checkpoint_idx + 1}/{len(checkpoint_paths)}")
        logging.info(f"Checkpoint: {checkpoint_path.name}")
        logging.info("=" * 80 + "\n")
        
        # ========================================================================
        # 6a. Setup Test Run Directory and File Logging for this checkpoint
        # ========================================================================
        logging.info("Setting up test run directory...")
        test_function_directory_path = setup_test_function_directory(experiment_dir, test_args)
        logging.info(f"Test run directory: {test_function_directory_path}")
        
        
        # ========================================================================
        # 6c. Run Testing for this checkpoint
        # ========================================================================
        logging.info("\nSTARTING TESTING")
        logging.info("=" * 80 + "\n")
        
        try:
            # Run appropriate test function based on test_function argument
            if test_args.test_function == 'float':
                test_results = load_and_test_float(
                    model=model,
                    checkpoint_path=checkpoint_path,
                    test_loader=test_loader,
                    loss_fn=loss_fn,
                    device=device,
                    augmenter=augmenter,
                    apply_augmentation_fn=apply_augmentation
                )
                
            elif test_args.test_function == 'single_precision_quantized':
                test_results = load_and_test_single_precision(
                    model=model,
                    checkpoint_path=checkpoint_path,
                    test_loader=test_loader,
                    loss_fn=loss_fn,
                    device=device,
                    augmenter=augmenter,
                    apply_augmentation_fn=apply_augmentation,
                    config=config,
                    test_args=test_args
                )
                
                # Save single precision results to CSV
                csv_filename = f"single_precision_results_{checkpoint_path.stem}.csv"
                csv_path = test_function_directory_path / csv_filename
                save_single_precision_results_to_csv(
                    test_results=test_results,
                    output_path=csv_path,
                    checkpoint_name=checkpoint_path.name
                )
                
            elif test_args.test_function == 'random_bitwidth':
                # breakpoint()
                test_results = load_and_test_random_bitwidths(
                    model=model,
                    checkpoint_path=checkpoint_path,
                    test_loader=test_loader,
                    loss_fn=loss_fn,
                    device=device,
                    augmenter=augmenter,
                    apply_augmentation_fn=apply_augmentation,
                    config=config,
                    test_args=test_args
                )
                
            else:
                raise ValueError(f"Invalid test function: {test_args.test_function}, "
                              f"must be one of: float, single_precision_quantized, random_bitwidth")
            
            # ========================================================================
            # 6d. Log Results for this checkpoint
            # ========================================================================
            logging.info("\n" + "=" * 80)
            logging.info(f"TESTING COMPLETED FOR CHECKPOINT: {checkpoint_path.name}")
            logging.info("=" * 80)
            for key, value in (test_results.items() if test_results else []):
                logging.info(f"{key}: {value} \n")
            logging.info(f"Test run directory: {test_function_directory_path}")
            logging.info("=" * 80)
            
            # Store results for this checkpoint
            all_results[checkpoint_path.name] = test_results
            
        except KeyboardInterrupt:
            logging.info("\n" + "=" * 80)
            logging.warning(f"Testing interrupted by user on checkpoint: {checkpoint_path.name}")
            logging.info("=" * 80)
            sys.exit(0)
        
        except Exception as e:
            logging.error("\n" + "=" * 80)
            logging.error(f"ERROR DURING TESTING OF CHECKPOINT: {checkpoint_path.name}")
            logging.error("=" * 80)
            logging.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            # Continue to next checkpoint instead of exiting
            logging.info("Continuing to next checkpoint...")
            continue
    
    # ========================================================================
    # 7. Final Summary
    # ========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("ALL TESTING COMPLETED!")
    logging.info("=" * 80)
    # logging.info(f"Tested {len(all_results)} checkpoint(s)")
    # for checkpoint_name, results in all_results.items():
    #     logging.info(f"\n{checkpoint_name}:")
    #     for key, value in (results.items() if results else []):
    #         logging.info(f"  {key}: {value}")
    # logging.info("=" * 80)


if __name__ == "__main__":
    main()

