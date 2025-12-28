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
    if config is not None:
        # Try to get loss config from global 'loss_name' key
        loss_name = config.get('loss_name', 'cross_entropy')
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
    else:
        raise ValueError(f"Unknown loss function: {loss_name}. "
                        f"Supported: 'cross_entropy', 'label_smoothing_ce'")


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

