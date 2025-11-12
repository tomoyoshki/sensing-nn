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
    
    return parser.parse_args()


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
    
    return config

