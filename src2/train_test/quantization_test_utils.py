import logging
from pathlib import Path
import yaml
import datetime
import numpy as np


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


def find_all_epoch_checkpoints(models_dir):
    """
    Find all checkpoint_epoch_*.pth files in a models directory.
    
    Args:
        models_dir: Path to the models directory
    
    Returns:
        list: Sorted list of checkpoint paths (by epoch number)
    """
    models_path = Path(models_dir)
    
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # Find all checkpoint_epoch_*.pth files
    checkpoint_files = list(models_path.glob("checkpoint_epoch_*.pth"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint_epoch_*.pth files found in: {models_dir}")
    
    # Sort by epoch number
    def get_epoch_num(path):
        # Extract number from checkpoint_epoch_10.pth -> 10
        return int(path.stem.split('_')[-1])
    
    checkpoint_files.sort(key=get_epoch_num)
    
    logging.info(f"Found {len(checkpoint_files)} checkpoint(s): {[f.name for f in checkpoint_files]}")
    
    return checkpoint_files


def get_checkpoint_by_epoch(models_dir, epoch_num):
    """
    Get a specific checkpoint by epoch number.
    
    Args:
        models_dir: Path to the models directory
        epoch_num: Epoch number to find
    
    Returns:
        Path: Path to the checkpoint file
    """
    models_path = Path(models_dir)
    checkpoint_path = models_path / f"checkpoint_epoch_{epoch_num}.pth"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logging.info(f"Found checkpoint: {checkpoint_path.name}")
    
    return checkpoint_path


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


def setup_test_run_directory(experiment_dir, suffix_to_test_run_dir_name):
    """
    Create a test run directory under the experiment directory and setup file logging.
    
    Args:
        experiment_dir: Path to the experiment directory
        suffix_to_test_run_dir_name: Suffix to add to the test run directory name
    
    Returns:
        test_run_dir: Path to the created test run directory
        log_file_path: Path to the logs.txt file
    """
    experiment_path = Path(experiment_dir)
    
    # Create directory name with date and test function name
    # Format: test_run_YYYYMMDD_HHMMSS_{test_function_name}
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_run_dir_name = f"test_run_{timestamp}_{suffix_to_test_run_dir_name}"
    test_run_dir = experiment_path / test_run_dir_name
    
    # Create the directory
    test_run_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created test run directory: {test_run_dir}")
    
    # Setup file logging to logs.txt in the test run directory
    log_file_path = test_run_dir / "logs.txt"
    file_handler = logging.FileHandler(log_file_path, mode='w')  # 'w' mode overwrites, 'a' would append
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Add file handler to root logger
    # This ensures all subsequent logging.info(), logging.error(), etc. go to both console and file
    logging.getLogger().addHandler(file_handler)
    
    # Write a header to the log file to mark when file logging started
    logging.info("=" * 80)
    logging.info(f"TEST RUN LOG FILE - {suffix_to_test_run_dir_name}")
    logging.info(f"Test run directory: {test_run_dir}")
    logging.info(f"Log file: {log_file_path}")
    logging.info("=" * 80)
    
    return test_run_dir, log_file_path


def setup_checkpoint_from_path(test_args):
    """
    Setup testing from a specific checkpoint path.
    Infers experiment directory from checkpoint path structure.
    
    Args:
        checkpoint_path: Path to checkpoint file (str or Path)
    
    Returns:
        tuple: (experiment_dir, checkpoint_paths, config)
            - experiment_dir: Path to experiment directory
            - checkpoint_paths: List containing single checkpoint path
            - config: Configuration dictionary loaded from experiment
    
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
    """
    logging.info("Mode: Testing specific checkpoint")
    checkpoint_path = Path(test_args.checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint_paths = [checkpoint_path]
    
    # Infer experiment directory from checkpoint path
    # Expected structure: experiment_dir/models/checkpoint.pth
    experiment_dir = checkpoint_path.parent.parent
    
    # Load config from experiment
    config = load_config_from_experiment(experiment_dir)
    
    return experiment_dir, checkpoint_paths, config


def setup_checkpoints_from_experiment_dir(test_args):
    """
    Setup testing from an experiment directory with checkpoint selection.
    
    Args:
        experiments_dir: Path to experiment directory (str or Path)
        run_checkpoint: List of epoch numbers to test, or None
        run_all_checkpoints: If True, test all checkpoint_epoch_*.pth files
    
    Returns:
        tuple: (experiment_dir, checkpoint_paths, config)
            - experiment_dir: Path to experiment directory
            - checkpoint_paths: List of checkpoint paths to test
            - config: Configuration dictionary loaded from experiment
    
    Raises:
        FileNotFoundError: If experiment dir, models dir, or checkpoints don't exist
    """
    logging.info("Mode: Testing from experiment directory")
    experiment_dir = Path(test_args.experiments_dir)
    
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    models_dir = experiment_dir / "models"
    
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # Load config from experiment
    config = load_config_from_experiment(experiment_dir)
    
    # Determine which checkpoints to test
    if test_args.run_checkpoint:
        # Run specific checkpoint(s) by epoch number(s)
        epoch_nums = test_args.run_checkpoint  # This is a list
        logging.info(f"Running checkpoints for epochs: {epoch_nums}")
        checkpoint_paths = []
        for epoch_num in epoch_nums:
            checkpoint_path = get_checkpoint_by_epoch(models_dir, epoch_num)
            checkpoint_paths.append(checkpoint_path)
    
    elif test_args.run_all_checkpoints:
        # Run all checkpoints
        logging.info("Running all checkpoints")
        checkpoint_paths = find_all_epoch_checkpoints(models_dir)
    elif test_args.use_best:
        # Run best model
        logging.info("Running best model")
        checkpoint_path = models_dir / "best_model.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Best model not found: {checkpoint_path}")
        checkpoint_paths = [checkpoint_path]
    elif test_args.use_last_epoch:
        # Run last epoch
        logging.info("Running last epoch")
        checkpoint_path = models_dir / "last_epoch.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Last epoch not found: {checkpoint_path}")
        checkpoint_paths = [checkpoint_path]
    else:
        raise ValueError("No checkpoint specified, use --run_checkpoint, --run_all_checkpoints, --use_best, or --use_last_epoch")
    
    return experiment_dir, checkpoint_paths, config



def validate_test_args(test_args):
    """
    Validate test arguments.
    
    Args:
        test_args: Test arguments
    
    Returns:
        tuple: (experiment_dir, checkpoint_paths, config)
    """
    if test_args.run_checkpoint and test_args.run_all_checkpoints:
        raise ValueError("Cannot use both --run_checkpoint and --run_all_checkpoints, choose one")
    if test_args.use_best and test_args.use_last_epoch:
        raise ValueError("Cannot use both --use_best and --use_last_epoch, choose one")
    if test_args.override_single_bitwidth and test_args.num_test_configs != 4:
        raise ValueError("Cannot use both --override_single_bitwidth and --num_test_configs, choose one")
    if test_args.run_checkpoint and not test_args.experiments_dir:
        raise ValueError("Cannot use --run_checkpoint without --experiments_dir")
    if test_args.run_all_checkpoints and not test_args.experiments_dir:
        raise ValueError("Cannot use --run_all_checkpoints without --experiments_dir")
    if test_args.use_best and not test_args.experiments_dir:
        raise ValueError("Cannot use --use_best without --experiments_dir")
    if test_args.use_last_epoch and not test_args.experiments_dir:
        raise ValueError("Cannot use --use_last_epoch without --experiments_dir")
    if (test_args.run_checkpoint or test_args.run_all_checkpoints) and (test_args.use_best or test_args.use_last_epoch):
        raise ValueError("Cannot use --run_checkpoint or --run_all_checkpoints with --use_best or --use_last_epoch, choose one")
    if (test_args.test_function == "random_bitwidths") and (test_args.bitwidth_bin_size is None):
        raise ValueError("Cannot use --test_function random_bitwidths without --bitwidth_bin_size")
    return True



def setup_test_function_directory(experiment_dir, test_args):
    """
    Setup test function directory.
    
    Args:
        experiment_dir: Path to experiment directory
        test_args: Test arguments
    """
    test_function_suffix = ""
    if test_args.run_checkpoint:
        test_function_suffix = f"_run_checkpoint_{test_args.run_checkpoint}"
    elif test_args.run_all_checkpoints:
        test_function_suffix = f"_run_all_checkpoints"
    elif test_args.use_best:
        test_function_suffix = f"_use_best"
    elif test_args.use_last_epoch:
        test_function_suffix = f"_use_last_epoch"
    else:
        raise ValueError("No test function specified, use --run_checkpoint, --run_all_checkpoints, --use_best, or --use_last_epoch")
    test_function_directory_path = experiment_dir / "test_functions" / f"test_run_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_{test_args.test_function}_{test_function_suffix}"
    test_function_directory_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created test function directory: {test_function_directory_path}")
    return test_function_directory_path


# =============================================================================
# Bitwidth Configuration Generation Functions
# =============================================================================

def validate_and_parse_bitwidth_ranges(bitwidth_values, parser=None):
    """
    Validate and parse bitwidth range values into list of (min, max) tuples.
    
    Args:
        bitwidth_values: List of float values from command line
        parser: Optional argparse.ArgumentParser for error reporting
    
    Returns:
        list of tuples: Each tuple is (min, max) for a bitwidth range
    
    Raises:
        ValueError or parser.error: If validation fails
    """
    # Check if we have an even number of values
    if len(bitwidth_values) % 2 != 0:
        error_msg = (
            f"--bitwidth_bin_size requires an even number of values (pairs of min/max), "
            f"but got {len(bitwidth_values)} values: {bitwidth_values}"
        )
        if parser:
            parser.error(error_msg)
        else:
            raise ValueError(error_msg)
    
    # Group into pairs and validate each pair
    ranges = []
    for i in range(0, len(bitwidth_values), 2):
        min_val = bitwidth_values[i]
        max_val = bitwidth_values[i + 1]
        
        # Validate: min <= max
        if min_val > max_val:
            error_msg = (
                f"--bitwidth_bin_size: min must be <= max, "
                f"but got range [{min_val}, {max_val}] at position {i//2 + 1}"
            )
            if parser:
                parser.error(error_msg)
            else:
                raise ValueError(error_msg)
        
        # Validate: both positive
        if min_val <= 0 or max_val <= 0:
            error_msg = (
                f"--bitwidth_bin_size: both min and max must be > 0, "
                f"but got range [{min_val}, {max_val}] at position {i//2 + 1}"
            )
            if parser:
                parser.error(error_msg)
            else:
                raise ValueError(error_msg)
        
        # Validate: reasonable upper bound (bitwidths typically <= 32)
        if min_val > 32 or max_val > 32:
            error_msg = (
                f"--bitwidth_bin_size: both min and max must be <= 32, "
                f"but got range [{min_val}, {max_val}] at position {i//2 + 1}"
            )
            if parser:
                parser.error(error_msg)
            else:
                raise ValueError(error_msg)
        
        ranges.append((min_val, max_val))
    
    return ranges


def get_layer_weight_counts(model, conv_class):
    """
    Extract the number of weights for each quantized conv layer in the model.
    
    Args:
        model: PyTorch model containing quantized Conv layers
        conv_class: The quantized Conv class type
    
    Returns:
        list: Number of weights per layer (in module order)
    """
    weight_counts = []
    for module in model.modules():
        if isinstance(module, conv_class):
            weight_counts.append(module.weight.numel())
    return weight_counts


def compute_relative_memory(scheme, weight_counts, max_bitwidth=8):
    """
    Compute relative memory consumption for a quantization scheme.
    
    Relative memory = (Σ Nw_i × bw_i) / (Σ Nw_i × max_bitwidth)
    
    Args:
        scheme: List of bitwidths, one per layer (quantization scheme)
        weight_counts: List of weight counts per layer
        max_bitwidth: Maximum bitwidth (default 8 for 8-bit reference)
    
    Returns:
        float: Relative memory consumption (0 to 1, where 1 = max_bitwidth for all)
    """
    assert len(scheme) == len(weight_counts), \
        f"Scheme length ({len(scheme)}) must match weight_counts length ({len(weight_counts)})"
    
    current_memory = sum(nw * bw for nw, bw in zip(weight_counts, scheme))
    max_memory = sum(nw * max_bitwidth for nw in weight_counts)
    
    if max_memory == 0:
        return 0.0
    
    return current_memory / max_memory


def generate_scheme_for_target_relative_memory(model, conv_class, bitwidth_options, target_relative_memory):
    """
    Generate a quantization scheme (bitwidth per layer) that achieves a target relative memory.
    
    Relative memory = current_memory / max_memory, where max_memory is at max_bitwidth.
    
    For 2 options: uses weighted linear interpolation
    For 3+ options: uses greedy upgrade from minimum (weighted by layer sizes)
    
    Args:
        model: PyTorch model containing quantized Conv layers
        conv_class: The quantized Conv class type
        bitwidth_options: List of available bitwidths (e.g., [2, 4, 8])
        target_relative_memory: Target relative memory (0 to 1, where 1 = max bitwidth all layers)
    
    Returns:
        list: Quantization scheme (bitwidth per layer)
    """
    options = sorted(bitwidth_options)
    max_bw = options[-1]  # Reference bitwidth (typically 8)
    min_bw = options[0]
    
    # Get weight counts for each layer
    weight_counts = get_layer_weight_counts(model, conv_class)
    num_layers = len(weight_counts)
    
    if num_layers == 0:
        return []
    
    # Compute memory bounds
    max_memory = sum(nw * max_bw for nw in weight_counts)
    min_memory = sum(nw * min_bw for nw in weight_counts)
    
    # Target memory in absolute terms
    target_memory = target_relative_memory * max_memory
    target_memory = np.clip(target_memory, min_memory, max_memory)
    
    if len(options) == 2:
        # Weighted linear interpolation for 2 bitwidth options
        # We need to decide which layers get high bitwidth vs low bitwidth
        # such that total memory = target_memory
        # 
        # Sort layers by weight count (largest first) and greedily assign
        b_low, b_high = options
        
        # Start with all low bitwidth
        scheme = [b_low] * num_layers
        current_memory = min_memory
        
        # Sort layer indices by weight count (largest first for efficiency)
        sorted_indices = sorted(range(num_layers), key=lambda i: weight_counts[i], reverse=True)
        np.random.shuffle(sorted_indices)  # Add randomness
        
        for idx in sorted_indices:
            if current_memory >= target_memory:
                break
            delta = weight_counts[idx] * (b_high - b_low)
            if current_memory + delta <= target_memory:
                scheme[idx] = b_high
                current_memory += delta
            else:
                # Check if upgrading gets us closer to target
                if abs(current_memory + delta - target_memory) < abs(current_memory - target_memory):
                    scheme[idx] = b_high
                    current_memory += delta
    else:
        # Greedy approach for 3+ options (weighted by layer sizes)
        # Start with all layers at minimum bitwidth
        scheme = [options[0]] * num_layers
        current_memory = min_memory
        
        # Randomize order of layers to upgrade
        layer_order = np.random.permutation(num_layers).tolist()
        
        while current_memory < target_memory:
            upgraded = False
            for layer_idx in layer_order:
                if current_memory >= target_memory:
                    break
                
                curr_bw = scheme[layer_idx]
                curr_bw_idx = options.index(curr_bw)
                nw = weight_counts[layer_idx]
                
                if curr_bw_idx < len(options) - 1:
                    # Find best upgrade: largest that doesn't overshoot,
                    # or smallest if all upgrades overshoot but gets us closer
                    best_new_bw = None
                    for next_bw in options[curr_bw_idx + 1:]:
                        delta = nw * (next_bw - curr_bw)
                        if current_memory + delta <= target_memory:
                            best_new_bw = next_bw  # Keep looking for larger upgrade
                        else:
                            if best_new_bw is None:
                                # All upgrades overshoot; take smallest if it gets us closer
                                if abs(current_memory + delta - target_memory) < abs(current_memory - target_memory):
                                    best_new_bw = next_bw
                            break
                    
                    if best_new_bw is not None:
                        old_bw = scheme[layer_idx]
                        scheme[layer_idx] = best_new_bw
                        current_memory += nw * (best_new_bw - old_bw)
                        upgraded = True
            
            if not upgraded:
                break
    
    return scheme


def generate_schemes_in_relative_memory_bin(model, conv_class, bitwidth_options, 
                                            relative_memory_consumption_bin_min, relative_memory_consumption_bin_max, num_schemes,
                                            max_attempts_per_scheme=100):
    """
    Generate multiple quantization schemes with relative memory in [relative_memory_consumption_bin_min, relative_memory_consumption_bin_max].
    
    Uses rejection sampling: generates a scheme, checks if it's in range,
    if not regenerates (up to max_attempts_per_scheme times).
    
    Relative memory is the ratio of current scheme memory to max-bitwidth memory.
    
    Args:
        model: PyTorch model containing quantized Conv layers
        conv_class: The quantized Conv class type
        bitwidth_options: List of available bitwidths (e.g., [2, 4, 8])
        bin_min: Minimum relative memory (inclusive, 0 to 1)
        bin_max: Maximum relative memory (inclusive, 0 to 1)
        num_schemes: Number of quantization schemes to generate
        max_attempts_per_scheme: Max regeneration attempts per scheme (default: 10)
    
    Returns:
        list of lists: Each inner list is a quantization scheme (bitwidth per layer)
    """
    # Pre-compute weight counts and max bitwidth for efficiency
    weight_counts = get_layer_weight_counts(model, conv_class)
    max_bw = max(bitwidth_options)
    
    schemes = []
    total_attempts = 0
    successful_first_try = 0
    
    for i in range(num_schemes):
        for attempt in range(max_attempts_per_scheme):
            total_attempts += 1
            
            # Sample target uniformly in the bin range
            target_relative_memory = np.random.uniform(relative_memory_consumption_bin_min, relative_memory_consumption_bin_max)
            scheme = generate_scheme_for_target_relative_memory(
                model, conv_class, bitwidth_options, target_relative_memory
            )
            
            # Verify the scheme is actually in range
            actual_rel_mem = compute_relative_memory(scheme, weight_counts, max_bitwidth=max_bw)
            
            if relative_memory_consumption_bin_min <= actual_rel_mem <= relative_memory_consumption_bin_max:
                if attempt == 0:
                    successful_first_try += 1
                schemes.append(scheme)
                break
        else:
            # Max attempts reached, use last generated scheme (best effort)
            schemes.append(scheme)
    
    return schemes


def get_num_quantized_layers(model, conv_class):
    """
    Count the number of quantized convolutional layers in the model.
    
    Args:
        model: PyTorch model
        conv_class: The quantized Conv class type to count
    
    Returns:
        int: Number of quantized layers
    """
    return sum(1 for m in model.modules() if isinstance(m, conv_class))


# =============================================================================
# Results Saving Functions
# =============================================================================

def save_single_precision_results_to_csv(test_results, output_path, checkpoint_name=None):
    """
    Save single precision test results to CSV file.
    
    Args:
        test_results: Dictionary with bitwidth as keys and test metrics as values
                     Format: {4: {'accuracy': 0.85, 'loss': 0.23}, ...}
        output_path: Path to save CSV file (can be Path object or string)
        checkpoint_name: Optional checkpoint name to include in CSV
    
    Returns:
        Path: Path to saved CSV file
    """
    import csv
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for CSV
    rows = []
    for bitwidth, metrics in sorted(test_results.items()):
        row = {
            'bitwidth': bitwidth,
            'accuracy': metrics.get('accuracy', 'N/A'),
            'loss': metrics.get('loss', 'N/A')
        }
        if checkpoint_name:
            row['checkpoint'] = checkpoint_name
        rows.append(row)
    
    # Write to CSV
    if rows:
        fieldnames = ['checkpoint', 'bitwidth', 'accuracy', 'loss'] if checkpoint_name else ['bitwidth', 'accuracy', 'loss']
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        logging.info(f"Single precision results saved to: {output_path}")
    else:
        logging.warning("No test results to save")
    
    return output_path