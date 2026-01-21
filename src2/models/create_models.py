import logging
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Add src2 to path for imports
src2_path = Path(__file__).parent.parent
sys.path.insert(0, str(src2_path))

from models.ResNet import build_multimodal_resnet
from models.MobileNetV2 import build_multimodal_mobilenet_v2


def create_model(config):
    """
    Create a multi-modal model based on the configuration
    
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
    
    # Classification parameters (support both activity_classification and vehicle_classification)
    task_cfg = (
        config.get("activity_classification")
        or config.get("vehicle_classification")
        or {}
    )
    num_classes = task_cfg.get("num_classes", 1000)
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
    
    # Check if quantization is enabled
    Conv = None
    BatchNorm = None
    quantization_enabled = config.get("quantization", {}).get("enable", False)
    
    if quantization_enabled:
        # Import quantization modules
        # conv_class_name = config["quantization"].get("Conv", "QuanConv")
        quantization_method_name = config.get("quantization_method", "dorefa")
        quantization_method_config = config["quantization"].get(quantization_method_name, {})
        conv_class_name = quantization_method_config.get("Conv", "QuanConv")

        
        logging.info(f"Quantization enabled:")
        logging.info(f"  Conv class: {conv_class_name}")
        logging.info(f"  Quantization method: {quantization_method_name}")
        
        # Dynamically import the Conv class specified in config
        import models.QuantModules as QuantModules
        activation_quantization = quantization_method_config.get("activation_quantization")
        if activation_quantization == "pact":
            def _pact_forward(self, inp, nbit_a, alpha, *args, **kwargs):
                inp = F.relu(inp)
                input_val = QuantModules.quantize(torch.clamp(inp, max=alpha), nbit_a, alpha)
                return input_val

            QuantModules.PACT.forward = _pact_forward
        Conv = getattr(QuantModules, conv_class_name)
        logging.info(f"  Successfully loaded Conv class: {Conv.__name__}")

        bn_type = quantization_method_config.get(
            "bn_type", config.get("quantization", {}).get("bn_type", "float")
        )
        if bn_type == "switchable":
            BatchNorm = QuantModules.SwitchableBatchNorm
        elif bn_type == "transitional":
            BatchNorm = QuantModules.TransitionalBatchNorm
        elif bn_type == "float":
            BatchNorm = torch.nn.BatchNorm2d
        else:
            raise ValueError(f"Invalid BatchNorm type: {bn_type}")

    else:
        logging.info("Quantization disabled - using standard Conv2d layers")
    
    # Create model
    selected_model_name = model_variant if model_variant else model_name
    selected_model_name_lower = str(selected_model_name).lower()

    if selected_model_name_lower in ["mobilenetv2", "mobilenet_v2", "mobilenet-v2"]:
        width_mult = config.get("MobileNetV2", {}).get("width_mult", 1.0)
        model = build_multimodal_mobilenet_v2(
            modality_names=modality_names,
            location_names=location_names,
            modality_in_channels=modality_in_channels,
            num_classes=num_classes,
            fc_dim=fc_dim,
            width_mult=width_mult,
            dropout_ratio=dropout_ratio,
            use_standard_first_layer=True,
            Conv=Conv,
            BatchNorm=BatchNorm,
        )
    else:
        model = build_multimodal_resnet(
            model_name=selected_model_name,
            modality_names=modality_names,
            location_names=location_names,
            modality_in_channels=modality_in_channels,
            num_classes=num_classes,
            fc_dim=fc_dim,
            dropout_ratio=dropout_ratio,
            use_standard_first_layer=True,
            Conv=Conv,
            BatchNorm=BatchNorm,
        )

    # Configure switchable/transitional BN after model creation
    if quantization_enabled and BatchNorm is not None:
        import models.QuantModules as QuantModules

        def _is_conv(module):
            conv_types = []
            if Conv is not None:
                conv_types.append(Conv)
            conv_types.append(torch.nn.Conv2d)
            return isinstance(module, tuple(set(conv_types)))

        if BatchNorm in (QuantModules.SwitchableBatchNorm, QuantModules.TransitionalBatchNorm):
            quant_config = quantization_method_config
            modules_in_order = list(model.modules())
            last_conv = None
            for idx, module in enumerate(modules_in_order):
                if isinstance(module, Conv):
                    last_conv = module
                    continue

                if isinstance(module, QuantModules.SwitchableBatchNorm):
                    if last_conv is not None:
                        module.set_input_conv(last_conv)
                    module.setup_quantize_funcs(quant_config)

                if isinstance(module, QuantModules.TransitionalBatchNorm):
                    input_conv = last_conv
                    output_conv = None
                    for next_module in modules_in_order[idx + 1:]:
                        if isinstance(next_module, Conv):
                            output_conv = next_module
                            break
                    module.set_convs(input_conv, output_conv)
                    module.setup_quantize_funcs(quant_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Model created successfully!")
    logging.info(f"  Total parameters: {total_params / 1e6:.2f}M")
    logging.info(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    return model

