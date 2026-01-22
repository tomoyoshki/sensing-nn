"""
Loss Functions Module

This module provides a flexible interface for different loss functions.
Supports:
- CrossEntropyLoss
- Label smoothing cross-entropy
- ImportanceVectorLoss (for importance vector quantization)
- WeightedBitCostLoss (for freqquant spectral-aware quantization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import math

import logging


# ============================================================================
# Helper Functions
# ============================================================================

def get_execution_order(model, sample_input):
    """
    Get layers in execution order using forward hooks.
    
    PyTorch's model.modules() returns modules in registration order (order defined
    in __init__), not execution order (order called in forward()). This function
    uses hooks to capture the actual execution order during a forward pass.
    
    Args:
        model: PyTorch model
        sample_input: Sample input tensor/dict to run forward pass
                     (can be dict for multi-modal models: {loc: {mod: tensor}})
    
    Returns:
        list of (name, module) tuples in execution order
    """
    execution_order = []
    hooks = []
    seen_modules = set()  # Track which modules we've already seen
    
    def make_hook(name, module):
        def hook(mod, input, output):
            # Only add each module once (first execution)
            if id(module) not in seen_modules:
                seen_modules.add(id(module))
                execution_order.append((name, module))
        return hook
    
    # Register hooks on ALL modules (not just leaf modules)
    # This is necessary to capture QuanConvSplit and similar modules that have children
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(make_hook(name, module)))
    
    # Run forward pass to collect execution order
    with torch.no_grad():
        model(sample_input)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    return execution_order

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

def get_loss_function(config=None, model=None):
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
    elif loss_name == "weighted_bitcost_loss":
        # Create base task loss function (default: cross_entropy)
        base_loss = nn.CrossEntropyLoss()
        
        # Get weighted bit cost loss parameters from config
        lambda_BC = loss_kwargs.get('lambda_BC', 0.01) if loss_kwargs else 0.01
        delta = loss_kwargs.get('delta', 1e-4) if loss_kwargs else 1e-4
        eta = loss_kwargs.get('eta', 1e-4) if loss_kwargs else 1e-4
        B_max = loss_kwargs.get('B_max', 8) if loss_kwargs else 8
        
        logging.info(f"  Using WeightedBitCostLoss (freqquant) with:")
        logging.info(f"    lambda_BC={lambda_BC}")
        logging.info(f"    delta={delta}")
        logging.info(f"    eta={eta}")
        logging.info(f"    B_max={B_max}")
        logging.info(f"  Note: Layer depth initialization will happen on first batch")
        
        return WeightedBitCostLoss(
            base_loss, model,
            lambda_BC=lambda_BC, delta=delta, eta=eta, B_max=B_max
        ), loss_name
    else:
        raise ValueError(f"Unknown loss function: {loss_name}. "
                        f"Supported: 'cross_entropy', 'label_smoothing_ce', 'importance_vector_loss', 'weighted_bitcost_loss'")


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


class WeightedBitCostLoss(nn.Module):
    """
    Weighted Bit Cost Loss for FreqQuant spectral-aware quantization.
    
    This loss function uses spectral saliency measures to guide bitwidth allocation
    across frequency bands and layers. It implements the methodology from the paper:
    
    Total Loss = L_task + lambda_BC * L_BC
    
    Where L_BC (weighted bit cost) considers:
    - Band importance from spectral analysis (energy * (1 - entropy))
    - Layer depth decay (early layers use saliency, deep layers minimize cost)
    - Variance gate (equal saliency → minimize cost)
    
    Works with QuanConvSplit layers that have beta_upper/beta_lower learnable 
    preference vectors for upper/lower frequency bands.
    """
    
    def __init__(self, task_loss_fn, model, lambda_BC=0.01, delta=1e-4, eta=1e-4, B_max=8):
        """
        Initialize WeightedBitCostLoss.
        
        Args:
            task_loss_fn: Base task loss function (e.g., nn.CrossEntropyLoss())
            model: PyTorch model containing QuanConvSplit layers
            lambda_BC: Coefficient for bit cost loss component (default: 0.01)
            delta: Threshold for soft mask computation (default: 1e-4)
            eta: Sensitivity for variance gate (default: 1e-4)
            B_max: Maximum bitwidth in bitwidth_options (default: 8)
        """
        super(WeightedBitCostLoss, self).__init__()
        self.task_loss_fn = task_loss_fn
        self.model = model
        self.lambda_BC = lambda_BC
        self.delta = delta
        self.eta = eta
        self.B_max = B_max
        
        # Layer depth information (initialized lazily on first forward pass)
        self.layer_depth_map = None  # {layer_instance: depth_index}
        self.L = None  # Total number of QuanConvSplit layers
        self._initialized = False
        
        # Cache for efficiency
        self._quanconvsplit_layers = None
        self._bitwidth_opts_tensor = None
    
    def _initialize_layer_depths(self, model, sample_input):
        """
        Initialize layer depth mapping using forward hooks.
        
        Runs a forward pass to capture the execution order of QuanConvSplit layers,
        then creates a mapping from layer instance to depth index.
        
        Args:
            model: PyTorch model
            sample_input: Sample input for forward pass
        """
        # Get execution order
        execution_order = get_execution_order(model, sample_input)
        
        # Import QuanConvSplit class
        from models.QuantModules import QuanConvSplit
        
        # Filter to only QuanConvSplit layers
        quanconvsplit_layers = []
        for name, module in execution_order:
            if isinstance(module, QuanConvSplit):
                quanconvsplit_layers.append((name, module))
        
        # Create depth mapping (1-indexed as per methodology)
        self.layer_depth_map = {}
        for idx, (name, module) in enumerate(quanconvsplit_layers):
            self.layer_depth_map[module] = idx + 1  # 1-indexed
        
        self.L = len(quanconvsplit_layers)
        self._quanconvsplit_layers = [module for _, module in quanconvsplit_layers]
        
        # Cache bitwidth options tensor (same for all layers)
        if self._quanconvsplit_layers:
            first_layer = self._quanconvsplit_layers[0]
            if hasattr(first_layer, 'bitwidth_opts') and first_layer.bitwidth_opts is not None:
                self._bitwidth_opts_tensor = torch.tensor(
                    first_layer.bitwidth_opts, 
                    dtype=torch.float32,
                    device=first_layer.beta_upper.device
                )
        
        logging.info(f"  WeightedBitCostLoss: Initialized with {self.L} QuanConvSplit layers")
        for name, module in quanconvsplit_layers:
            logging.info(f"    Layer {self.layer_depth_map[module]}/{self.L}: {name}")
    
    def compute_band_importance_single(self, importance_tensor):
        """
        Compute band importance for a single modality's importance tensor.
        
        From methodology (Eq. 26): I_g = sum_{j in G_g} i_j
        
        Args:
            importance_tensor: tensor[num_freq_bins] - importance scores for one modality
        
        Returns:
            I_upper: Total importance of upper frequency band (scalar tensor)
            I_lower: Total importance of lower frequency band (scalar tensor)
        """
        # Split into upper/lower halves
        num_bins = importance_tensor.shape[0]
        half = num_bins // 2
        
        # Sum importance within each band
        I_upper = importance_tensor[:half].sum()
        I_lower = importance_tensor[half:].sum()
        
        return I_upper, I_lower
    
    def compute_relative_importance(self, I_upper, I_lower, epsilon=1e-10):
        """
        Compute relative band importance (normalized).
        
        From methodology (Eq. 33): tilde{I}_g = I_g / (sum I_j + epsilon)
        
        Args:
            I_upper: Importance of upper band
            I_lower: Importance of lower band
            epsilon: Small constant for numerical stability
        
        Returns:
            tilde_I_upper: Relative importance of upper band (sums with lower to 1)
            tilde_I_lower: Relative importance of lower band
        """
        total = I_upper + I_lower + epsilon
        tilde_I_upper = I_upper / total
        tilde_I_lower = I_lower / total
        return tilde_I_upper, tilde_I_lower
    
    def compute_soft_mask(self, tilde_I_g):
        """
        Compute soft mask to suppress bands with near-zero saliency.
        
        From methodology (Eq. 61): m_g = tilde{I}_g / (tilde{I}_g + delta)
        
        Args:
            tilde_I_g: Relative importance of band g
        
        Returns:
            m_g: Soft mask value in [0, 1)
        """
        return tilde_I_g / (tilde_I_g + self.delta)
    
    def compute_variance_gate(self, tilde_I_upper, tilde_I_lower):
        """
        Compute variance gate to detect if saliency provides discriminative signal.
        
        From methodology (Eq. 113-114):
            sigma^2 = (1/G) * sum[(tilde_I_g - 1/G)^2]
            gamma = sigma^2 / (sigma^2 + eta)
        
        Where G=2 (number of bands).
        
        Purpose:
        - If all bands have equal saliency → sigma^2 = 0 → gamma = 0 → minimize bit cost
        - If bands have different saliency → gamma > 0 → allocate by saliency
        
        Args:
            tilde_I_upper: Relative importance of upper band
            tilde_I_lower: Relative importance of lower band
        
        Returns:
            gamma: Variance gate value in [0, 1)
        """
        G = 2  # Number of bands (upper/lower)
        uniform = 1.0 / G  # 0.5
        
        # Variance of relative importance
        sigma_sq = ((tilde_I_upper - uniform) ** 2 + (tilde_I_lower - uniform) ** 2) / G
        
        # Variance gate
        gamma = sigma_sq / (sigma_sq + self.eta)
        
        return gamma
    
    def compute_alpha_l(self, l):
        """
        Compute layer depth decay coefficient.
        
        From methodology (Eq. 67): alpha(l) = cos(π * l / (2 * L))
        
        Purpose: Early layers (l small) preserve spectral structure (alpha ≈ 1),
                 deep layers (l → L) use uniform weighting (alpha → 0).
        
        Args:
            l: Layer depth (1-indexed)
        
        Returns:
            alpha_l: Layer decay coefficient in [0, 1]
        """
        if self.L == 0:
            return 0.0
        return math.cos(math.pi * l / (2 * self.L))
    
    def compute_expected_bitwidth(self, beta_vector):
        """
        Compute expected bitwidth from beta preference vector.
        
        From methodology (Eq. 106): B_g^(l) = sum_{k in K} beta_{g,k}^(l) * k
        
        Args:
            beta_vector: Learnable preference logits tensor [num_bitwidths]
        
        Returns:
            expected_bw: Expected bitwidth (scalar tensor)
        """
        # Apply softmax to get probabilities
        probs = torch.softmax(beta_vector, dim=0)
        
        # Compute expected bitwidth
        expected_bw = torch.sum(probs * self._bitwidth_opts_tensor)
        
        return expected_bw
    
    def compute_bit_allocation_coefficient(self, tilde_I_g, alpha_l, gamma, m_g, max_importance):
        """
        Compute bit allocation coefficient for a band at a layer.
        
        From methodology (Eq. 120): 
            c_g(l) = alpha(l) * gamma * m_g * (tilde{I}_g / max_j{tilde{I}_j})
        
        Purpose: Determines whether to minimize or maximize bitwidth
        - c_g → 0: minimize bit cost (low saliency or deep layer or no discriminative signal)
        - c_g → 1: maximize bitwidth (high saliency band at early layer)
        
        Args:
            tilde_I_g: Relative importance of band g
            alpha_l: Layer depth decay coefficient
            gamma: Variance gate
            m_g: Soft mask for band g
            max_importance: Maximum relative importance across bands
        
        Returns:
            c_g: Bit allocation coefficient in [0, 1]
        """
        epsilon = 1e-10
        normalized_importance = tilde_I_g / (max_importance + epsilon)
        c_g = alpha_l * gamma * m_g * normalized_importance
        return c_g
    
    def compute_bitcost_for_modality(self, importance_tensor, device):
        """
        Compute weighted bit cost loss for a single modality.
        
        This computes the full L_BC for one modality's importance scores,
        iterating over all QuanConvSplit layers.
        
        Args:
            importance_tensor: tensor[num_freq_bins] - importance scores for one modality
            device: torch device
        
        Returns:
            modality_loss: Weighted bit cost loss for this modality (scalar tensor)
        """
        # 1. Compute band importance for this modality
        I_upper, I_lower = self.compute_band_importance_single(importance_tensor)
        
        # 2. Compute relative importance
        tilde_I_upper, tilde_I_lower = self.compute_relative_importance(I_upper, I_lower)
        
        # 3. Compute variance gate
        gamma = self.compute_variance_gate(tilde_I_upper, tilde_I_lower)
        
        # 4. Compute soft masks
        m_upper = self.compute_soft_mask(tilde_I_upper)
        m_lower = self.compute_soft_mask(tilde_I_lower)
        
        # 5. Get max importance for normalization
        max_importance = torch.max(tilde_I_upper, tilde_I_lower)
        
        # 6. Iterate over all QuanConvSplit layers and compute weighted bit cost
        modality_loss = torch.tensor(0.0, device=device)
        
        for layer in self._quanconvsplit_layers:
            # Get layer depth
            l = self.layer_depth_map[layer]
            
            # Compute alpha for this layer depth
            alpha_l = self.compute_alpha_l(l)
            
            # --- Upper band ---
            # Compute expected bitwidth
            B_upper = self.compute_expected_bitwidth(layer.beta_upper)
            
            # Compute bit allocation coefficient
            c_upper = self.compute_bit_allocation_coefficient(
                tilde_I_upper, alpha_l, gamma, m_upper, max_importance
            )
            
            # Compute loss contribution: (1 - c) * B + c * (B_max - B)
            loss_upper = (1 - c_upper) * B_upper + c_upper * (self.B_max - B_upper)
            
            # --- Lower band ---
            # Compute expected bitwidth
            B_lower = self.compute_expected_bitwidth(layer.beta_lower)
            
            # Compute bit allocation coefficient
            c_lower = self.compute_bit_allocation_coefficient(
                tilde_I_lower, alpha_l, gamma, m_lower, max_importance
            )
            
            # Compute loss contribution
            loss_lower = (1 - c_lower) * B_lower + c_lower * (self.B_max - B_lower)
            
            # Add to modality total
            modality_loss = modality_loss + loss_upper + loss_lower
        
        return modality_loss
    
    def compute_weighted_bitcost_loss(self, model, importance_scores):
        """
        Compute weighted bit cost loss across all QuanConvSplit layers and all modalities.
        
        From methodology (Eq. 128):
            L_BC = sum_{l=1}^{L} sum_{g=1}^{G} [(1 - c_g(l)) * B_g^(l) + c_g(l) * (B_max - B_g^(l))]
        
        Since different modalities have different numbers of frequency bins, we compute
        L_BC separately for each modality and sum them:
            L_BC_total = sum_{modality} L_BC(modality)
        
        Where:
        - When c_g → 0: loss = B_g → optimizer minimizes bitwidth
        - When c_g → 1: loss = B_max - B_g → optimizer maximizes bitwidth
        
        Args:
            model: PyTorch model with QuanConvSplit layers
            importance_scores: dict {loc: {mod: tensor[num_freq_bins]}}
                              Note: Each modality can have different num_freq_bins
        
        Returns:
            L_BC: Weighted bit cost loss summed across all modalities (scalar tensor)
        """
        # Handle case where no importance scores provided
        if importance_scores is None:
            raise ValueError("No importance scores found in compute_weighted_bitcost_loss")
        
        device = next(model.parameters()).device
        total_loss = torch.tensor(0.0, device=device)
        num_modalities = 0
        
        # Iterate over each location and modality
        for loc in importance_scores:
            for mod in importance_scores[loc]:
                importance_tensor = importance_scores[loc][mod]
                
                # Compute L_BC for this modality
                modality_loss = self.compute_bitcost_for_modality(importance_tensor, device)
                total_loss = total_loss + modality_loss
                num_modalities += 1
        
        if num_modalities == 0:
            logging.warning("No importance scores found in compute_weighted_bitcost_loss")
            return torch.tensor(0.0, device=device, requires_grad=False)
        
        # Optionally normalize by number of modalities (to make loss scale consistent)
        # Uncomment the line below if you want to average instead of sum
        # total_loss = total_loss / num_modalities
        
        return total_loss
    
    def forward(self, outputs, labels, model=None, importance_scores=None, data=None):
        """
        Compute total loss combining task loss and weighted bit cost.
        
        Args:
            outputs: Model output logits [batch_size, num_classes]
            labels: Ground truth labels [batch_size] or [batch_size, num_classes]
            model: PyTorch model (uses self.model if not provided)
            importance_scores: dict {loc: {mod: tensor[num_freq_bins]}}
            data: Optional input data for lazy initialization
        
        Returns:
            total_loss: Combined loss value
            loss_components: Dict with detailed breakdown for logging
        """
        model = model if model is not None else self.model
        
        # Validate required inputs - fail fast if not provided
        if importance_scores is None:
            raise ValueError("WeightedBitCostLoss requires importance_scores. "
                           "Make sure augmenter is configured with augmentation_mode='with_energy_and_entropy'")
        
        if data is None:
            raise ValueError("WeightedBitCostLoss requires data for layer depth initialization. "
                           "Pass data to the forward() method.")
        
        # Lazy initialization of layer depths (on first forward pass)
        if not self._initialized:
            self._initialize_layer_depths(model, data)
            self._initialized = True
        
        # Handle one-hot labels
        if len(labels.shape) == 2 and labels.shape[1] > 1:
            loss_labels = torch.argmax(labels, dim=1)
        else:
            loss_labels = labels
        
        # 1. Compute task loss
        task_loss = self.task_loss_fn(outputs, loss_labels)
        
        # 2. Compute weighted bit cost loss
        bitcost_loss = self.compute_weighted_bitcost_loss(model, importance_scores)
        
        # 3. Combine losses
        total_loss = task_loss + self.lambda_BC * bitcost_loss
        
        # 4. Compute accuracy and F1 score for logging
        predictions = torch.argmax(outputs, dim=1)
        if len(labels.shape) == 2 and labels.shape[1] > 1:
            labels_idx = torch.argmax(labels, dim=1)
        else:
            labels_idx = labels
        accuracy = (predictions == labels_idx).float().mean()
        
        # Compute precision, recall, and F1 score (macro-averaged across classes)
        num_classes = outputs.shape[1]
        precision_sum = 0.0
        recall_sum = 0.0
        valid_classes = 0
        
        for c in range(num_classes):
            # True positives: predicted c and actual c
            tp = ((predictions == c) & (labels_idx == c)).sum().float()
            # False positives: predicted c but actual not c
            fp = ((predictions == c) & (labels_idx != c)).sum().float()
            # False negatives: predicted not c but actual c
            fn = ((predictions != c) & (labels_idx == c)).sum().float()
            
            # Precision = TP / (TP + FP)
            if (tp + fp) > 0:
                precision_c = tp / (tp + fp)
                precision_sum += precision_c
            
            # Recall = TP / (TP + FN)
            if (tp + fn) > 0:
                recall_c = tp / (tp + fn)
                recall_sum += recall_c
                valid_classes += 1
        
        # Macro-averaged precision and recall
        if valid_classes > 0:
            precision = precision_sum / valid_classes
            recall = recall_sum / valid_classes
            # F1 = 2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = torch.tensor(0.0)
        else:
            precision = torch.tensor(0.0)
            recall = torch.tensor(0.0)
            f1_score = torch.tensor(0.0)
        
        # 5. Return with components for logging
        loss_components = {
            'task_loss': task_loss.item(),
            'bitcost_loss': bitcost_loss.item() if isinstance(bitcost_loss, torch.Tensor) else bitcost_loss,
            'scaled_bitcost_loss': (self.lambda_BC * bitcost_loss).item() if isinstance(bitcost_loss, torch.Tensor) else 0.0,
            'total_loss': total_loss.item(),
            'accuracy': accuracy.item(),
            'precision': precision.item() if isinstance(precision, torch.Tensor) else precision,
            'recall': recall.item() if isinstance(recall, torch.Tensor) else recall,
            'f1_score': f1_score.item() if isinstance(f1_score, torch.Tensor) else f1_score,
            'lambda_BC': self.lambda_BC
        }
        
        return total_loss, loss_components