import argparse
import yaml
from pathlib import Path


def parse_args():
    """
    Parse command line arguments for model training/testing.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Load configuration from YAML file for model training/testing'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name (e.g., ResNet, DeepSense, TransformerV4)'
    )

    parser.add_argument(
        '--model_variant',
        type=str,
        required=False,
        help='Model variant (e.g., resnet18, resnet34, resnet50, resnet101, resnet152)'
    )

    parser.add_argument(
        '--loss',
        type=str,
        required=False,
        default='cross_entropy',
        help='Loss function (e.g., cross_entropy)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        help='Dataset name (e.g., ACIDS, PAMAP2, Parkland, RealWorld_HAR)'
    )
    
    parser.add_argument(
        '--yaml_path',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        required=False,
        help='GPU to use'
    )
    
    parser.add_argument(
        '--quantization_method',
        type=str,
        default='dorefa',
        required=False,
        help='Quantization method to use (e.g., dorefa, any_precision). Only used if quantization is enabled in config.'
    )
    
    return parser.parse_args()


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
    
    # Run Checkpoints (requires --experiments_dir)
    run_checkpoint_group = parser.add_mutually_exclusive_group(required=False)
    run_checkpoint_group.add_argument(
        '--run_checkpoint',
        type=int,
        nargs='+',
        help='Run specific checkpoint(s) by epoch number (e.g., --run_checkpoint 10 20 30)'
    )
    run_checkpoint_group.add_argument(
        '--run_all_checkpoints',
        action='store_true',
        help='Run all checkpoint_epoch_*.pth files in the experiment models directory'
    )
    run_checkpoint_group.add_argument(
        '--use_best',
        action='store_true',
        help='Use best model checkpoint (default: True). If False, uses last epoch.'
    )

    run_checkpoint_group.add_argument(
        '--use_last_epoch',
        action='store_true',
        help='Use last epoch checkpoint (default: False). If True, uses last epoch.'
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

    parser.add_argument(
        '--test_function',
        type=str,
        required=True,
        help='Test function to use, float, single_precision_quantized, random_bitwidth'
    )

    # Bitwidth override options (mutually exclusive)
    bitwidth_override_group = parser.add_mutually_exclusive_group(required=False)
    bitwidth_override_group.add_argument(
        '--override_single_bitwidth',
        type=int,
        nargs='+',
        default=None,
        help=(
            "For multiple entries, all will be tested one by one - This will override the bitwidth options used during training - \n"
            "Currently do not put a bitwidth that it was not trained on it will throw an error, model will crash\n"
            "Example usage:\n"
            "  --override_single_bitwidth 4 6 8\n"
            "Note: Cannot be used with --override_bitwidth_options or --num_test_configs"
        )
    )
    bitwidth_override_group.add_argument(
        '--override_bitwidth_options',
        type=int,
        nargs='+',
        default=None,
        help=(
            "List of bitwidth options for random bitwidth testing.\n"
            "Example usage:\n"
            "  --override_bitwidth_options 4 6 8\n"
            "This will override the bitwidth options used during training - \n"
            "Currently do not put a bitwidth that it was not trained on it will throw an error, model will crash\n"
            "Note: Cannot be used with --override_single_bitwidth"
        )
    )

    parser.add_argument(
        '--num_test_configs',
        type=int,
        default=4,
        help='Number of random bitwidth configurations to test (for random_bitwidth test function). Ignored when --override_single_bitwidth is used.'
    )

    parser.add_argument(
        '--relative_memory_consumption_bin_size',
        type=float,
        nargs='+',  # Accept one or more values
        metavar='VALUE',
        default=None,
        required=False,
        help=(
            "Specify one or more relative memory ranges as pairs of [min, max] values.\n"
            "Each range requires exactly two values (min, max).\n"
            "Examples:\n"
            "  Single range:\n"
            "    --relative_memory_consumption_bin_size 0.5 0.7\n"
            "  Multiple ranges:\n"
            "    --relative_memory_consumption_bin_size 0.5 0.7 0.7 0.9 0.9 1.0\n"
            "    (Creates 3 ranges: [0.5, 0.7], [0.7, 0.9], [0.9, 1.0])\n"
            "Total number of values must be even (pairs of min/max)."
        )
    )

    
    args = parser.parse_args()
    
    # Validate and parse relative_memory_consumption_bin_size into pairs
    if hasattr(args, 'relative_memory_consumption_bin_size') and args.relative_memory_consumption_bin_size is not None:
        from train_test.quantization_test_utils import validate_and_parse_relative_memory_ranges
        args.relative_memory_consumption_bin_size = validate_and_parse_relative_memory_ranges(
            args.relative_memory_consumption_bin_size, 
            parser
        )

    # Backwards-compat: older testing code uses `bitwidth_bin_size` to mean
    # "relative memory bin ranges". Mirror the parsed value onto that attribute.
    # This keeps `quantization_test_functions.load_and_test_random_bitwidths()` working.
    if not hasattr(args, 'bitwidth_bin_size'):
        args.bitwidth_bin_size = None
    if args.relative_memory_consumption_bin_size is not None:
        args.bitwidth_bin_size = args.relative_memory_consumption_bin_size
    
    return args


def load_yaml_config(yaml_path):
    """
    Load YAML configuration file and return as dictionary.
    
    Args:
        yaml_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary loaded from YAML
        
    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    yaml_file = Path(yaml_path)
    
    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")
    
    with open(yaml_file, 'r') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {yaml_path}: {e}")


def get_config():
    """
    Single function to parse arguments and load configuration.
    This is the main entry point for getting configuration in main.py.
    
    Returns:
        tuple: (args, config_dict)
            - args: Namespace containing model, dataset, and yaml_path
            - config_dict: Dictionary containing the full YAML configuration
    """
    # Parse command line arguments
    args = parse_args()
    
    # Load YAML configuration
    config = load_yaml_config(args.yaml_path)
    
    # Add command line args to config for easy access
    config['model'] = args.model
    config['yaml_path'] = args.yaml_path
    config['device'] = f'cuda:{args.gpu}'
    config['model_variant'] = args.model_variant
    config['quantization_method'] = args.quantization_method
    return config

