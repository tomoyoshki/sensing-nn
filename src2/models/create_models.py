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
    fc_dim = config.get("fc_dim", 512) # config.get(model_name, {}).get("fc_dim", 512) is correct way to get fc_dim
    dropout_ratio = config.get("dropout_ratio", 0) # config.get(model_name,{}).get("dropout_ratio", 0) is correct way to get dropout_ratio
    model_variant = config.get("model_variant", None)
    
    logging.info(f"Creating {model_name} model...")
    logging.info(f"  Modalities: {modality_names}")
    logging.info(f"  Locations: {location_names}")
    logging.info(f"  Input channels: {modality_in_channels}")
    logging.info(f"  Number of classes: {num_classes}")
    logging.info(f"  FC dimension: {fc_dim}")
    logging.info(f"  Dropout ratio: {dropout_ratio}")
    
    # Check if quantization is enabled
    Conv = None
    BatchNorm = None
    quantization_enabled = config.get("quantization", {}).get("enable", False)
    
    if quantization_enabled:
        # Import quantization modules
        # conv_class_name = config["quantization"].get("Conv", "QuanConv")
        quantization_method_name = config.get("quantization_method",None)
        assert quantization_method_name is not None, "Quantization method name is not provided in the config"
        quantization_method_config = config["quantization"].get(quantization_method_name, {})
        # breakpoint()
        conv_class_name = quantization_method_config.get("Conv",None)
        assert conv_class_name is not None, f"Conv class name is not provided in the config for quantization method: {quantization_method_name}"

        
        logging.info(f"Quantization enabled:")
        logging.info(f"  Conv class: {conv_class_name}")
        logging.info(f"  Quantization method: {quantization_method_name}")
        
        # Dynamically import the Conv class specified in config
        import models.QuantModules as QuantModules
        try:
            Conv = getattr(QuantModules, conv_class_name)
        except AttributeError:
            raise ValueError(f"Conv class {conv_class_name} not found in QuantModules. Available classes: {list(QuantModules.__dict__.keys())}")
        logging.info(f"  Successfully loaded Conv class: {Conv.__name__}")
    else:
        logging.info("Quantization disabled - using standard Conv2d layers")
    
    # Create model
    model = build_multimodal_resnet(
        model_name=model_variant if model_variant else model_name,
        modality_names=modality_names,
        location_names=location_names,
        modality_in_channels=modality_in_channels,
        num_classes=num_classes,
        fc_dim=fc_dim,
        dropout_ratio=dropout_ratio,
        use_standard_first_layer=True,
        Conv=Conv,
        BatchNorm=BatchNorm
    )

    # model = build_vit(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Model created successfully!")
    logging.info(f"  Total parameters: {total_params / 1e6:.2f}M")
    logging.info(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    return model

