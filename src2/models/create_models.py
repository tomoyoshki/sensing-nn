import logging
import torch
from pathlib import Path
import sys

# Add src2 to path for imports
src2_path = Path(__file__).parent.parent
sys.path.insert(0, str(src2_path))

from models.ResNet import build_multimodal_resnet


def create_model(config):
    """
    Create a multi-modal ResNet model based on the configuration
    
    Args:
        config: Configuration dictionary containing dataset and model parameters
    
    Returns:
        model: MultiModalResNet instance
    """
    # Extract configuration parameters
    # model_config = config["model"]
    
    # Model name
    model_name = config.get("model", "resnet18")
    
    # Modality and location information
    modality_names = config.get("modality_names", [])
    location_names = config.get("location_names", [])
    modality_in_channels = config["loc_mod_in_freq_channels"]
    
    # Classification parameters
    num_classes = config.get("vehicle_classification", {}).get("num_classes", 1000)
    fc_dim = config.get("fc_dim", 512)
    dropout_ratio = config.get("dropout_ratio", 0)
    model_variant = config.get("model_variant", None)
    
    logging.info(f"Creating {model_name} model...")
    logging.info(f"  Modalities: {modality_names}")
    logging.info(f"  Locations: {location_names}")
    logging.info(f"  Input channels: {modality_in_channels}")
    logging.info(f"  Number of classes: {num_classes}")
    logging.info(f"  FC dimension: {fc_dim}")
    logging.info(f"  Dropout ratio: {dropout_ratio}")
    
    # Create model
    model = build_multimodal_resnet(
        model_name=model_variant if model_variant else model_name,
        modality_names=modality_names,
        location_names=location_names,
        modality_in_channels=modality_in_channels,
        num_classes=num_classes,
        fc_dim=fc_dim,
        dropout_ratio=dropout_ratio,
        use_standard_first_layer=True
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Model created successfully!")
    logging.info(f"  Total parameters: {total_params / 1e6:.2f}M")
    logging.info(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    return model

