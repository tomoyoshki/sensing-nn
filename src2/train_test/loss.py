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

    # Log loss function details
    logging.info(f"Loss function: {loss_name}")
    if loss_kwargs:
        for key, value in loss_kwargs.items():
            logging.info(f"  {key}: {value}")
    
    # Create loss function based on name
    if loss_name == "cross_entropy":
        filtered_kwargs = filter_kwargs(nn.CrossEntropyLoss.__init__, loss_kwargs)
        return nn.CrossEntropyLoss(**filtered_kwargs)
    elif loss_name == "label_smoothing_ce":
        filtered_kwargs = filter_kwargs(nn.CrossEntropyLoss.__init__, loss_kwargs)
        return nn.CrossEntropyLoss(**filtered_kwargs)
    elif loss_name == "importance_vector_loss":
        # Create base task loss function (default: cross_entropy)
        base_loss = nn.CrossEntropyLoss()
        
        # Get importance vector loss parameters
        lambda_var = loss_kwargs.get('lambda_variance', 0.1) if loss_kwargs else 0.1
        variance_metric = loss_kwargs.get('variance_metric', 'loss') if loss_kwargs else 'loss'
        
        logging.info(f"  Using ImportanceVectorLoss with lambda_variance={lambda_var}, metric={variance_metric}")
        return ImportanceVectorLoss(base_loss, lambda_var, variance_metric)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}. "
                        f"Supported: 'cross_entropy', 'label_smoothing_ce', 'importance_vector_loss'")


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
    Multi-component loss for importance vector-based adaptive quantization.
    
    Combines:
    1. Task loss (e.g., cross-entropy for classification)
    2. Variance regularization (encourages consistent predictions across bitwidth configs)
    
    The variance component penalizes high variance in predictions/losses across
    different sampled bitwidth configurations, promoting robustness.
    """
    
    def __init__(self, task_loss_fn, lambda_var=0.1, variance_metric='loss'):
        """
        Args:
            task_loss_fn: Base task loss function (e.g., nn.CrossEntropyLoss())
            lambda_var: Weight for variance regularization component
            variance_metric: 'loss' or 'accuracy' - what to compute variance over
        """
        super(ImportanceVectorLoss, self).__init__()
        self.task_loss_fn = task_loss_fn
        self.lambda_var = lambda_var
        self.variance_metric = variance_metric
        
        assert variance_metric in ['loss', 'accuracy'], \
            f"variance_metric must be 'loss' or 'accuracy', got {variance_metric}"
    
    def forward(self, outputs_or_list, labels_or_list):
        """
        Compute multi-component loss for K different bitwidth configurations.
        
        Handles two modes:
        1. Soft mode: Single tensor inputs (outputs, labels) - just computes task loss
        2. Hard mode: Lists of K tensors - computes task loss + variance regularization
        
        Args:
            outputs_or_list: Either a single tensor [batch_size, num_classes] (soft mode)
                           or list of K output tensors (hard mode)
            labels_or_list: Either a single tensor [batch_size] or [batch_size, num_classes] (soft mode)
                          or list of K label tensors (hard mode)
        
        Returns:
            total_loss: Combined loss (task + variance in hard mode, just task in soft mode)
            In hard mode only: loss_components dict with 'task_loss', 'variance_loss', 'total_loss'
        """
        # Check if we're in soft mode (single tensors) or hard mode (lists)
        is_soft_mode = isinstance(outputs_or_list, torch.Tensor)
        
        if is_soft_mode:
            # Soft mode: single tensor inputs, just compute task loss
            outputs = outputs_or_list
            labels = labels_or_list
            
            # Handle one-hot labels if needed
            if len(labels.shape) == 2 and labels.shape[1] > 1:
                loss_labels = torch.argmax(labels, dim=1)
            else:
                loss_labels = labels
            
            # Compute and return task loss only
            return self.task_loss_fn(outputs, loss_labels)
        
        # Hard mode: list of tensors from multiple bitwidth configurations
        outputs_list = outputs_or_list
        labels_list = labels_or_list
        
        K = len(outputs_list)
        assert K > 1, "Need at least 2 configs to compute variance"
        assert len(labels_list) == K, "Mismatch between outputs and labels"
        
        # Compute task loss for each configuration
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
            
            # Compute accuracy for variance metric
            predictions = torch.argmax(outputs, dim=1)
            if len(labels.shape) == 2 and labels.shape[1] > 1:
                labels_idx = torch.argmax(labels, dim=1)
            else:
                labels_idx = labels
            
            accuracy = (predictions == labels_idx).float().mean()
            accuracies.append(accuracy)
        
        # Mean task loss across configurations
        task_loss = torch.stack(task_losses).mean()
        
        # Compute variance regularization
        if self.variance_metric == 'loss':
            # Variance of losses across K configurations
            variance = torch.stack(task_losses).var()
        else:  # 'accuracy'
            # Variance of accuracies across K configurations
            variance = torch.stack(accuracies).var()
        
        variance_loss = self.lambda_var * variance
        
        # Total loss
        total_loss = task_loss + variance_loss
        
        # Return loss and components for logging
        loss_components = {
            'task_loss': task_loss.item(),
            'variance_loss': variance_loss.item(),
            'total_loss': total_loss.item(),
            'raw_variance': variance.item()
        }
        
        return total_loss, loss_components

