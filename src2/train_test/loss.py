"""
Loss Functions Module

This module provides a flexible interface for different loss functions.
Currently supports CrossEntropyLoss, with easy extension for future loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

import logging

def filter_kwargs(func, kwargs):
    """
    Filter kwargs to only include valid parameters for the given function.
    
    Args:
        func: The function to filter kwargs for
        kwargs: Dictionary of keyword arguments
    
    Returns:
        Filtered dictionary containing only valid parameters
    """
    if kwargs is None:
        return {}
    sig = inspect.signature(func)
    valid_params = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in valid_params}

def get_loss_function(config=None, loss_name=None, model=None, **kwargs):
    """
    Factory function to get the appropriate loss function from config or arguments.
    
    Args:
        config (dict, optional): Configuration dictionary. If provided, extracts
            loss configuration from config['loss'] or config[model_name]['loss']
        loss_name (str, optional): Name of the loss function. Options:
            - "cross_entropy": Standard cross-entropy loss (default)
            - "label_smoothing_ce": Cross-entropy with label smoothing
            - "focal": Focal loss (can be added in the future)
        model (nn.Module, optional): PyTorch model. Some loss functions may need
            access to model weights (e.g., for regularization terms)
        **kwargs: Additional arguments to pass to the loss function
    
    Returns:
        loss_fn: The loss function
    
    Example:
        >>> # From config
        >>> config = {'loss': {'name': 'cross_entropy'}}
        >>> loss_fn = get_loss_function(config=config, model=model)
        
        >>> # Direct specification
        >>> loss_fn = get_loss_function(loss_name="cross_entropy", model=model)
        
        >>> # With label smoothing
        >>> config = {'loss': {'name': 'label_smoothing_ce', 'label_smoothing': 0.1}}
        >>> loss_fn = get_loss_function(config=config, model=model)
    """
    # Extract loss configuration from config if provided
    quantization_method_name = config.get('quantization_method', None)
    quantization_method_config = config.get('quantization', {}).get(quantization_method_name, None)

    if quantization_method_config is not None:
        # Try to get loss config from global 'loss_name' key
        loss_name = quantization_method_config.get('loss_name', 'cross_entropy')
        loss_kwargs = config.get(loss_name, None)
        if loss_name == "kurtosis":
            loss_kwargs = dict(loss_kwargs or {})
            for key in (
                "kurtosis_weight",
                "kurtosis_target",
                "kurtosis_eps",
            ):
                if key in quantization_method_config:
                    loss_kwargs[key] = quantization_method_config[key]

    # Log loss function details
    logging.info(f"Loss function: {loss_name}")
    if loss_kwargs:
        for key, value in loss_kwargs.items():
            logging.info(f"  {key}: {value}")
    
    # Create loss function based on name
    if loss_name == "cross_entropy":
        filtered_kwargs = filter_kwargs(nn.CrossEntropyLoss.__init__, loss_kwargs)
        return nn.CrossEntropyLoss(**filtered_kwargs), loss_name
    elif loss_name == "label_smoothing_ce":
        filtered_kwargs = filter_kwargs(nn.CrossEntropyLoss.__init__, loss_kwargs)
        return nn.CrossEntropyLoss(**filtered_kwargs), loss_name
    elif loss_name == "importance_vector_loss":
        # Create base task loss function (default: cross_entropy)
        base_loss = nn.CrossEntropyLoss()
        
        # Get importance vector loss parameters
        target_avg_bitwidth = loss_kwargs.get('target_avg_bitwidth', 4.0) if loss_kwargs else 4.0
        budget_coeff = loss_kwargs.get('budget_coeff', 0.05) if loss_kwargs else 0.05
        
        logging.info(f"  Using ImportanceVectorLoss with target_avg_bitwidth={target_avg_bitwidth}, budget_coeff={budget_coeff}")
        return ImportanceVectorLoss(base_loss, budget_coeff, target_avg_bitwidth), loss_name
    elif loss_name in ("kurtosis", "kurtosis_loss"):
        if model is None:
            raise ValueError("Kurtosis loss requires model to be provided")
        loss_kwargs = loss_kwargs or {}
        kurtosis_weight = loss_kwargs.get("kurtosis_weight", 0.1)
        kurtosis_target = loss_kwargs.get("kurtosis_target", 1.8)
        if isinstance(kurtosis_weight, str):
            kurtosis_weight = float(kurtosis_weight)
        if isinstance(kurtosis_target, str):
            kurtosis_target = float(kurtosis_target)
        eps = loss_kwargs.get("kurtosis_eps", 1e-8)
        if isinstance(eps, str):
            eps = float(eps)
        return KurtosisLoss(
            model=model,
            kurtosis_weight=kurtosis_weight,
            kurtosis_target=kurtosis_target,
            eps=eps
        ), loss_name
    else:
        raise ValueError(f"Unknown loss function: {loss_name}. "
                        f"Supported: 'cross_entropy', 'label_smoothing_ce', 'importance_vector_loss', 'kurtosis'")


def convert_to_one_hot(labels, num_classes):
    """
    Convert class indices to one-hot encoded labels.
    
    Args:
        labels: Tensor of shape (batch_size,) containing class indices
        num_classes: Number of classes
    
    Returns:
        one_hot: Tensor of shape (batch_size, num_classes)
    
    Note:
        PyTorch CrossEntropyLoss expects class indices, not one-hot vectors.
        This function is provided for compatibility with other loss functions.
    """
    return F.one_hot(labels, num_classes=num_classes).float()


def convert_from_one_hot(one_hot_labels):
    """
    Convert one-hot encoded labels back to class indices.
    
    Args:
        one_hot_labels: Tensor of shape (batch_size, num_classes)
    
    Returns:
        labels: Tensor of shape (batch_size,) containing class indices
    """
    return torch.argmax(one_hot_labels, dim=1)


class LossWrapper:
    """
    Wrapper class for loss functions that handles label format conversions.
    
    This is useful when your data pipeline outputs one-hot labels but
    your loss function expects class indices (or vice versa).
    """
    
    def __init__(self, loss_fn, expects_one_hot=False):
        """
        Args:
            loss_fn: The underlying loss function
            expects_one_hot: If True, converts class indices to one-hot.
                           If False (default), converts one-hot to class indices.
        """
        self.loss_fn = loss_fn
        self.expects_one_hot = expects_one_hot
    
    def __call__(self, logits, labels):
        """
        Compute loss, handling label format conversion if needed.
        
        Args:
            logits: Model output logits (batch_size, num_classes)
            labels: Labels (either class indices or one-hot)
        
        Returns:
            loss: Scalar loss value
        """
        # Check if labels are one-hot encoded (2D with more than 1 column)
        is_one_hot = len(labels.shape) == 2 and labels.shape[1] > 1
        
        if self.expects_one_hot and not is_one_hot:
            # Convert class indices to one-hot
            num_classes = logits.shape[1]
            labels = convert_to_one_hot(labels, num_classes)
        
        elif not self.expects_one_hot and is_one_hot:
            # Convert one-hot to class indices
            labels = convert_from_one_hot(labels)
        
        return self.loss_fn(logits, labels)

class ImportanceVectorLoss(nn.Module):
    """
    Simplified loss for importance vector-based adaptive quantization.
    
    Combines:
    1. Task loss (e.g., cross-entropy for classification)
    2. Bitwidth budget constraint (gentle push toward target average bitwidth)
    
    The budget component gently encourages the model to achieve a target
    average bitwidth across all layers (e.g., 4-bit average) without
    overwhelming the task loss signal.
    """
    
    def __init__(self, task_loss_fn, budget_coeff=0.001, 
                 target_avg_bitwidth=None):
        """
        Args:
            task_loss_fn: Base task loss function (e.g., nn.CrossEntropyLoss())
            budget_coeff: Coefficient for bitwidth budget constraint component
                         (default: 0.001, much smaller than before to avoid dominating)
            target_avg_bitwidth: Target average bitwidth across network (e.g., 4.0)
                                If None, budget loss is disabled
        """
        super(ImportanceVectorLoss, self).__init__()
        self.task_loss_fn = task_loss_fn
        self.budget_coeff = budget_coeff
        self.target_avg_bitwidth = target_avg_bitwidth
    
    def compute_expected_bitwidth_loss(self, model):
        """
        Compute bitwidth budget constraint loss.
        
        Encourages the network to achieve a target average bitwidth by
        computing the expected bitwidth from each layer's importance distribution.
        
        Args:
            model: PyTorch model containing QuanConvImportance layers
        
        Returns:
            budget_loss: MSE between average expected bitwidth and target
        """
        if self.target_avg_bitwidth is None:
            return torch.tensor(0.0)
        
        conv_class = model.get_conv_class()
        assert conv_class is not None, "conv_class is None"
        
        expected_bitwidths = []
        
        for module in model.modules():
            if isinstance(module, conv_class):
                # Check that this module instance has the required attributes
                # (these are set by setup_quantize_funcs on each instance)
                if not hasattr(module, 'importance_vector') or module.importance_vector is None:
                    continue
                if not hasattr(module, 'bitwidth_opts_tensor') or module.bitwidth_opts_tensor is None:
                    continue
                
                # Get softmax probabilities from importance vector
                probs = torch.softmax(module.importance_vector, dim=0)
                
                # Compute expected bitwidth: E[bitwidth] = sum(p_i * bitwidth_i)
                expected_bw = torch.sum(probs * module.bitwidth_opts_tensor)
                expected_bitwidths.append(expected_bw)
        
        if len(expected_bitwidths) == 0:
            return torch.tensor(0.0)
        
        # Average expected bitwidth across all layers
        avg_expected_bitwidth = torch.stack(expected_bitwidths).mean()
        
        # MSE loss from target
        budget_loss = (avg_expected_bitwidth - self.target_avg_bitwidth) ** 2
        
        return budget_loss
    
    def forward(self, outputs_or_list, labels_or_list, model=None):
        """
        Compute multi-component loss for K different bitwidth configurations.
        
        Handles two modes:
        1. Single forward mode: Single tensor inputs (outputs, labels) - computes task loss only
        2. Multi-config mode: Lists of K tensors - computes task loss + budget
        
        Args:
            outputs_or_list: Either a single tensor [batch_size, num_classes] (single mode)
                           or list of K output tensors (multi-config mode)
            labels_or_list: Either a single tensor [batch_size] or [batch_size, num_classes]
                          or list of K label tensors (multi-config mode)
            model: PyTorch model (required for bitwidth budget loss, optional otherwise)
        
        Returns:
            If single mode:
                total_loss: Just task loss
            If multi-config mode:
                total_loss: Combined loss (task + budget)
                loss_components: Dict with detailed breakdown
        """
        # Check if we're in single mode (single tensors) or multi-config mode (lists)
        is_single_mode = isinstance(outputs_or_list, torch.Tensor)
        
        if is_single_mode:
            # ================================================================
            # SINGLE FORWARD MODE: Just compute task loss
            # ================================================================
            outputs = outputs_or_list
            labels = labels_or_list
            
            # Handle one-hot labels if needed
            if len(labels.shape) == 2 and labels.shape[1] > 1:
                loss_labels = torch.argmax(labels, dim=1)
            else:
                loss_labels = labels
            
            # Compute and return task loss only
            return self.task_loss_fn(outputs, loss_labels)
        
        # ================================================================
        # MULTI-CONFIG MODE: Compute all loss components
        # ================================================================
        outputs_list = outputs_or_list
        labels_list = labels_or_list
        
        K = len(outputs_list)
        assert K > 1, "Need at least 2 configs to compute statistics"
        assert len(labels_list) == K, "Mismatch between outputs and labels"
        
        # ----------------------------------------------------------------
        # 1. TASK LOSS (averaged across K configurations)
        # ----------------------------------------------------------------
        task_losses = []
        accuracies = []
        
        for outputs, labels in zip(outputs_list, labels_list):
            # Handle one-hot labels if needed
            if len(labels.shape) == 2 and labels.shape[1] > 1:
                loss_labels = torch.argmax(labels, dim=1)
            else:
                loss_labels = labels
            
            # Compute loss
            loss = self.task_loss_fn(outputs, loss_labels)
            task_losses.append(loss)
            
            # Compute accuracy for logging
            predictions = torch.argmax(outputs, dim=1)
            if len(labels.shape) == 2 and labels.shape[1] > 1:
                labels_idx = torch.argmax(labels, dim=1)
            else:
                labels_idx = labels
            
            accuracy = (predictions == labels_idx).float().mean()
            accuracies.append(accuracy)
        
        # Mean task loss across configurations
        task_loss = torch.stack(task_losses).mean()
        
        # ----------------------------------------------------------------
        # 2. BITWIDTH BUDGET CONSTRAINT LOSS (gentle regularization)
        # ----------------------------------------------------------------
        if model is not None and self.target_avg_bitwidth is not None:
            budget_loss_raw = self.compute_expected_bitwidth_loss(model)
            budget_loss = self.budget_coeff * budget_loss_raw
        else:
            budget_loss_raw = torch.tensor(0.0)
            budget_loss = torch.tensor(0.0)
        
        # ----------------------------------------------------------------
        # TOTAL LOSS
        # ----------------------------------------------------------------
        total_loss = task_loss + budget_loss
        
        # ----------------------------------------------------------------
        # Return loss and components for logging
        # ----------------------------------------------------------------
        # Compute variance statistics for logging only (not used in loss)
        raw_variance = torch.stack(task_losses).var()
        
        loss_components = {
            'task_loss': task_loss.item(),
            'variance_loss': 0.0,  # Keep for backward compatibility but set to 0
            'budget_loss': budget_loss.item(),
            'total_loss': total_loss.item(),
            'raw_variance': raw_variance.item(),  # Keep for logging/monitoring
            'raw_budget_loss': budget_loss_raw.item() if isinstance(budget_loss_raw, torch.Tensor) else 0.0,
            'mean_accuracy': torch.stack(accuracies).mean().item(),
            'accuracy_std': torch.stack(accuracies).std().item()
        }
        
        return total_loss, loss_components


def compute_weight_kurtosis_regularization(
    model,
    target_kurtosis=1.8,
    eps=1e-8,
):
    """
    Compute kurtosis regularization over weight tensors.
    """
    device = next(model.parameters()).device
    kurtosis_sum = torch.tensor(0.0, device=device)
    count = 0

    for module in model.modules():
        if not hasattr(module, "weight"):
            continue
        weight = module.weight
        if not isinstance(weight, torch.Tensor) or weight.dim() < 2:
            continue

        mean = weight.mean()
        var = weight.var(unbiased=False)
        centered = weight - mean
        kurtosis = (centered ** 4).mean() / (var ** 2 + eps)
        kurtosis_sum = kurtosis_sum + (kurtosis - target_kurtosis) ** 2
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device)

    denom = float(count)
    return kurtosis_sum / denom


class KurtosisLoss(nn.Module):
    """
    Task loss + kurtosis regularization on weights.
    """

    def __init__(
        self,
        model,
        kurtosis_weight=0.1,
        kurtosis_target=1.8,
        eps=1e-8,
    ):
        super().__init__()
        self.model = model
        self.kurtosis_weight = kurtosis_weight
        self.kurtosis_target = kurtosis_target
        self.eps = eps
        self.task_loss_fn = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        if len(labels.shape) == 2 and labels.shape[1] > 1:
            loss_labels = torch.argmax(labels, dim=1)
        else:
            loss_labels = labels

        task_loss = self.task_loss_fn(outputs, loss_labels)

        kurtosis_reg = compute_weight_kurtosis_regularization(
            self.model,
            target_kurtosis=self.kurtosis_target,
            eps=self.eps,
        )

        return (
            task_loss
            + self.kurtosis_weight * kurtosis_reg
        )


