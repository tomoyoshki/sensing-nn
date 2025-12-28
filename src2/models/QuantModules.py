import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd


"""
Quantization Modules for Neural Networks
=========================================

This module provides quantization support for convolutional layers with the following features:
- Multiple quantization methods: DoReFa, PACT, LSQ, LSQPlus
- Adaptive bitwidth support
- Base class architecture for extensibility

Architecture Overview
--------------------

BaseQuanConv: Abstract base class providing common functionality
├── Constructor: Initializes conv parameters, weights, bias
├── setup_quantize_funcs(): Setup from config
├── _setup_quantizers(): Initialize quantizers (DoReFa/PACT/LSQ/LSQPlus)
├── alpha_setup(): Configure PACT alpha parameters
├── Bitwidth management: set/get/random bitwidth methods
├── Helper methods:
│   ├── _quantize_weight(): Quantize weights
│   ├── _quantize_activation(): Quantize activations
│   └── _apply_convolution(): Apply conv2d operation
└── _custom_init(): Hook for subclass-specific initialization

QuanConv: Standard implementation (original behavior)
└── forward(): Weight quant -> Weight norm -> Activation quant -> Conv

QuanConvActivationFirst: Alternative implementation
└── forward(): Activation quant -> Weight quant -> Weight norm -> Conv

QuanConvNoWeightNorm: Simplified implementation
└── forward(): Weight quant -> Activation quant -> Conv (no normalization)

Creating Custom Implementations
-------------------------------

To create a new quantized convolution variant:

1. Inherit from BaseQuanConv
2. Override forward() to implement your quantization strategy
3. Optionally override _custom_init() for additional setup
4. Use the provided helper methods:
   - self._quantize_weight()
   - self._quantize_activation(inp)
   - self._apply_convolution(inp, weight, bias)

Example:
    class MyCustomQuanConv(BaseQuanConv):
        def _custom_init(self):
            # Custom initialization
            self.my_custom_param = nn.Parameter(torch.ones(1))
        
        def forward(self, inp):
            if self.float_mode or self.curr_bitwidth == 32:
                return self._apply_convolution(inp, self.weight)
            
            # Your custom quantization logic here
            w = self._quantize_weight()
            x = self._quantize_activation(inp)
            # ... custom processing ...
            return self._apply_convolution(x, w)
"""


def quantize(x, nbit, alpha=None):
    """Quantization function used by DoReFa and PACT"""
    class DynamicQuantizer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, nbit, alpha=None):
            if nbit == 32:
                # Full precision mode: don't save alpha, no quantization happens
                # Alpha should not affect gradients in this mode
                ctx.save_for_backward(x)
                ctx.is_full_precision = True
                return x
            else:
                # Quantized mode: save alpha only if provided, compute gradients for it
                ctx.is_full_precision = False
                if alpha is not None:
                    ctx.save_for_backward(x, alpha)
                    ctx.has_alpha = True
                else:
                    ctx.save_for_backward(x)
                    ctx.has_alpha = False
                scale = (
                    (2**nbit - 1) if alpha is None else (2**nbit - 1) / alpha
                )
                return torch.round(scale * x) / scale

        @staticmethod
        def backward(ctx, grad_output):
            saved = ctx.saved_tensors
            x = saved[0]
            
            # In full precision mode, alpha should not receive gradients
            if ctx.is_full_precision:
                grad_input = grad_output.clone()
                return grad_input, None, None
            
            # In quantized mode, check if alpha was provided
            if not ctx.has_alpha:
                # No alpha provided (e.g., DoReFa without PACT)
                grad_input = grad_output.clone()
                return grad_input, None, None
            else:
                # Alpha was provided (PACT quantization), compute gradients
                alpha = saved[1]
                lower_bound = x < 0
                upper_bound = x > alpha
                x_range = ~(lower_bound | upper_bound)
                # Fix alpha gradient: alpha is tensor shape [len(bitwidths), 1, 1, 1]
                # Correct STE for PACT: y = clamp(x, 0, α), so α gets gradient only where x > α
                # Sum over batch, channels, height, width dimensions, keep bitwidth dimension
                grad_alpha = (grad_output * (x > alpha).float()).sum(dim=(0, 1, 2, 3), keepdim=True)
                return grad_output * x_range.float(), None, grad_alpha
    
    return DynamicQuantizer.apply(x, nbit, alpha)


class LSQQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, step_size, nbit, is_activation=False):
        qn = 0 if is_activation else -(2 ** (nbit - 1))
        qp = 2 ** (nbit - 1) - 1 if not is_activation else 2 ** nbit - 1

        ctx.save_for_backward(x, step_size)
        ctx.other = (qn, qp, is_activation)

        step_size = step_size.to(x.device)
        x = x / step_size
        x_clipped = torch.clamp(x.round(), qn, qp)
        return x_clipped * step_size

    @staticmethod
    def backward(ctx, grad_output):
        x, step_size = ctx.saved_tensors
        qn, qp, is_activation = ctx.other

        x_div = x / step_size
        x_clipped = torch.clamp(x_div.round(), qn, qp)
        mask = ((x_div >= qn) & (x_div <= qp)).float()

        grad_x = grad_output * mask
        grad_s = ((grad_output * (x_div - x_clipped)).sum()).view_as(step_size)
        grad_s /= ((qp * x.numel()) ** 0.5)

        return grad_x, grad_s, None, None


class LSQ(nn.Module):
    def __init__(self, is_activation=False, bitwidth=8, shape=1):
        super().__init__()
        self.is_activation = is_activation
        self.bitwidth = bitwidth
        self.step_size = nn.Parameter(torch.ones(shape))
        self.initialized = False

    def update_bitwidth(self, bw):
        """Update bitwidth dynamically"""
        self.bitwidth = bw

    def forward(self, x):
        if not self.initialized:
            if self.is_activation:
                # Activations: scale = E(|x|) / (2^(b-1) - 1)
                scale = x.abs().mean() / (2 ** (self.bitwidth - 1) - 1)
            else:
                # Weights: scale = 2·E(|w|) / (2^(b-1) - 1)
                scale = 2 * x.abs().mean() / (2 ** (self.bitwidth - 1) - 1)
            self.step_size.data.copy_(scale.clamp(min=1e-6))
            self.initialized = True
        return LSQQuantizer.apply(x, self.step_size, self.bitwidth, self.is_activation)


class LSQPlus(LSQ):
    def __init__(self, is_activation=False, bitwidth=8, shape=1, zero_point=True):
        super().__init__(is_activation, bitwidth, shape)
        self.zero_point = zero_point
        self.zp = nn.Parameter(torch.zeros(shape), requires_grad=zero_point)

    def update_bitwidth(self, bw):
        """Update bitwidth dynamically"""
        self.bitwidth = bw

    def forward(self, x):
        if not self.initialized:
            if self.is_activation:
                # Activations: scale = E(|x|) / (2^(b-1) - 1)
                scale = x.abs().mean() / (2 ** (self.bitwidth - 1) - 1)
            else:
                # Weights: scale = 2·E(|w|) / (2^(b-1) - 1)
                scale = 2 * x.abs().mean() / (2 ** (self.bitwidth - 1) - 1)
            self.step_size.data.copy_(scale.clamp(min=1e-6))
            self.initialized = True
        # LSQPlus with zero-point: compute quantized integer first, then clamp, then dequantize
        qn = 0
        qp = 2 ** self.bitwidth - 1
        x_int = torch.clamp((x / self.step_size + self.zp).round(), qn, qp)
        return (x_int - self.zp) * self.step_size


class DoReFaW(nn.Module):
    def __init__(self):
        super(DoReFaW, self).__init__()

    def forward(self, inp, nbit_w, *args, **kwargs):
        """forward pass"""
        w = torch.tanh(inp)
        maxv = torch.abs(w).max() + 1e-8  # Add epsilon to prevent division by zero
        w = w / (2 * maxv) + 0.5  # This is the quantization step in DoreFa
        w = 2 * quantize(w, nbit_w) - 1  # This is the dequantization step in DoreFa
        return w


class DoReFaA(nn.Module):
    def __init__(self):
        super(DoReFaA, self).__init__()

    def forward(self, inp, nbit_a, *args, **kwargs):
        """forward pass"""
        return quantize(torch.clamp(inp, 0, 1), nbit_a, *args, **kwargs)


class PACT(nn.Module):
    def __init__(self):
        super(PACT, self).__init__()

    def forward(self, inp, nbit_a, alpha, *args, **kwargs):
        """forward pass"""
        input = torch.clamp(inp, min=0, max=alpha)  # Use tensor alpha directly
        input_val = quantize(input, nbit_a, alpha)
        return input_val


class BaseQuanConv(nn.Module):
    """
    Base class for quantized convolutional layers.
    
    This class provides common functionality for quantized convolutions:
    - Convolution parameter setup (channels, kernel size, stride, etc.)
    - Weight and bias initialization
    - Quantizer setup (DoReFa, PACT, LSQ, LSQPlus)
    - Alpha parameter management for PACT
    - Bitwidth management (adaptive quantization)
    - Float mode support
    - Config-based setup
    
    Subclasses should override the forward() method to implement
    different quantization strategies.
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(BaseQuanConv, self).__init__()

        
        # Basic conv parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Weight and bias
        # For grouped convolutions, weight shape is (out_channels, in_channels // groups, kernel_h, kernel_w)
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        nn.init.kaiming_uniform_(
            self.weight, mode="fan_out", nonlinearity="relu"
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        
        # Current bitwidth (adaptive quantization)
        self.curr_bitwidth = None  # Default to full precision
        
        # Initialize quantizer placeholders (will be set by setup_quantize_funcs)
        self.quantize_w = None
        self.quantize_a = None
        self.quantization_enabled = False
        self.alpha_setup_flag = False
        self.weight_quantization = None
        self.activation_quantization = None
        self.bitwidth_opts = None
        self.switchable_clipping = None
        self.sat_weight_normalization = None
        
        # Setup quantization functions if parameters provided directly
        self.quantization_config = None
        # Call custom initialization hook for subclasses
        self._custom_init()
    
    def _custom_init(self):
        """
        Custom initialization hook for subclasses.
        Override this method to add custom initialization logic.
        """
        pass
    
    def setup_quantize_funcs(self, quantization_config):
        """
        Setup quantization functions from quantization configuration dictionary
        
        Args:
            quantization_config: Quantization configuration dictionary
        """
        self.quantization_config = quantization_config
        
        # Set quantization_enabled from config (defaults to True if not specified)
        self.quantization_enabled = quantization_config.get('quantization_enabled', True)
        
        self.weight_quantization = quantization_config["weight_quantization"]
        self.activation_quantization = quantization_config["activation_quantization"]
        self.bitwidth_opts = quantization_config["bitwidth_options"]
        self.switchable_clipping = quantization_config["switchable_clipping"]
        self.sat_weight_normalization = quantization_config["sat_weight_normalization"]
        
        # Store for _setup_quantizers
        
        # Setup quantizers
        self._setup_quantizers()
        
        # Alpha setup for PACT
        if self.activation_quantization == "pact":
            self.alpha_setup_flag = True
            if self.switchable_clipping:
                # Alpha shape: [len(bitwidth_opts), 1, 1, 1]
                self.alpha = nn.Parameter(torch.ones(len(self.bitwidth_opts), 1, 1, 1), requires_grad=True)
            else:
                # Single alpha with shape [1, 1, 1, 1]
                self.alpha = nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=True)
            nn.init.constant_(self.alpha, 10.0)
        elif self.activation_quantization in ["lsq", "lsqplus"]:
            self.alpha_setup_flag = False  # LSQ doesn't use alpha the same way
    
    def _setup_quantizers(self):
        """Setup weight and activation quantizers"""
        # Weight quantization
        if self.weight_quantization == "dorefa":
            self.quantize_w = DoReFaW()
        elif self.weight_quantization == "lsq":
            self.quantize_w = LSQ(is_activation=False, bitwidth=self.bitwidth_opts[0])
        elif self.weight_quantization == "lsqplus":
            self.quantize_w = LSQPlus(is_activation=False, bitwidth=self.bitwidth_opts[0])
        else:
            raise NotImplementedError(
                f"Weight Quantization '{self.weight_quantization}' not supported. "
                "Use 'dorefa', 'lsq', or 'lsqplus'"
            )
        
        # Activation quantization
        if self.activation_quantization == "dorefa":
            self.quantize_a = DoReFaA()
        elif self.activation_quantization == "pact":
            self.quantize_a = PACT()
        elif self.activation_quantization == "lsq":
            self.quantize_a = LSQ(is_activation=True, bitwidth=self.bitwidth_opts[0])
        elif self.activation_quantization == "lsqplus":
            self.quantize_a = LSQPlus(is_activation=True, bitwidth=self.bitwidth_opts[0])
        else:
            raise NotImplementedError(
                f"Activation Quantization '{self.activation_quantization}' not supported. "
                "Use 'dorefa', 'pact', 'lsq', or 'lsqplus'"
            )
    
    def get_alpha(self, bitwidth=None):
        """
        Get alpha value for PACT quantization
        
        Args:
            bitwidth: Specific bitwidth to get alpha for. If None, uses current bitwidth.
        
        Returns:
            Alpha parameter value
        """
        assert self.alpha_setup_flag, "Alpha not setup. Call alpha_setup() first."
        bitwidth = self.curr_bitwidth if bitwidth is None else bitwidth
        if self.switchable_clipping:
            if bitwidth in self.bitwidth_opts:
                idx = self.bitwidth_opts.index(bitwidth)
                return self.alpha[idx]
            else:
                # If bitwidth not in options, return first alpha
                return self.alpha[0]
        else:
            return self.alpha[0]
    
    def set_bitwidth(self, bitwidth):
        """
        Set the current bitwidth for adaptive quantization
        
        Args:
            bitwidth: Target bitwidth (must be in bitwidth_opts or 32 for full precision)
        """
        assert bitwidth <= 32 and bitwidth > 1, "bitwidth should be between 2 and 32"
        if bitwidth != 32 and self.bitwidth_opts is not None:
            assert bitwidth in self.bitwidth_opts, (
                f"bitwidth {bitwidth} not in bitwidth_options {self.bitwidth_opts}"
            )
        self.curr_bitwidth = bitwidth
        
        # Update LSQ/LSQPlus bitwidth dynamically (only for quantized modes)
        if bitwidth != 32 and isinstance(self.quantize_w, (LSQ, LSQPlus)):
            self.quantize_w.update_bitwidth(bitwidth)
        if bitwidth != 32 and isinstance(self.quantize_a, (LSQ, LSQPlus)):
            self.quantize_a.update_bitwidth(bitwidth)
    
    def get_bitwidth(self):
        """Get the current bitwidth"""
        return self.curr_bitwidth
    
    def set_random_bitwidth(self):
        """Randomly select a bitwidth from available options"""
        assert self.bitwidth_opts is not None, "bitwidth options not set"
        bw = self.bitwidth_opts[torch.randint(0, len(self.bitwidth_opts), (1,)).item()]
        self.set_bitwidth(bw)
    
    def set_highest_bitwidth(self):
        """Set bitwidth to the maximum available option"""
        assert self.bitwidth_opts is not None, "bitwidth options not set"
        self.curr_bitwidth = max(self.bitwidth_opts)
    
    def set_lowest_bitwidth(self):
        """Set bitwidth to the minimum available option"""
        assert self.bitwidth_opts is not None, "bitwidth options not set"
        self.curr_bitwidth = min(self.bitwidth_opts)
    
    def forward(self, inp):
        """
        Forward pass - to be implemented by subclasses.
        
        This is the main method that subclasses should override to implement
        different quantization strategies.
        
        Args:
            inp: Input tensor of shape [batch_size, in_channels, height, width]
        
        Returns:
            Output tensor of shape [batch_size, out_channels, out_height, out_width]
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def _quantize_weight(self):
        """
        Helper method to quantize weights according to current settings.
        
        Returns:
            Quantized weight tensor
        """
        if isinstance(self.quantize_w, (LSQ, LSQPlus)):
            return self.quantize_w(self.weight)
        else:
            return self.quantize_w(self.weight, self.curr_bitwidth)
    
    def _quantize_activation(self, inp):
        """
        Helper method to quantize activations according to current settings.
        
        Args:
            inp: Input activation tensor
            
        Returns:
            Quantized activation tensor
        """
        if isinstance(self.quantize_a, PACT):
            alpha = self.get_alpha()
            return self.quantize_a(inp, self.curr_bitwidth, alpha)
        elif isinstance(self.quantize_a, DoReFaA):
            return self.quantize_a(inp, self.curr_bitwidth)
        elif isinstance(self.quantize_a, (LSQ, LSQPlus)):
            return self.quantize_a(inp)
        else:
            raise ValueError(
                f"Unsupported activation quantization type: {type(self.quantize_a)}"
            )
    
    def _apply_convolution(self, inp, weight, bias=None):
        """
        Helper method to apply the convolution operation.
        
        Args:
            inp: Input tensor
            weight: Weight tensor
            bias: Optional bias tensor (defaults to self.bias if None)
            
        Returns:
            Output tensor after convolution
        """
        if bias is None:
            bias = self.bias
        return nn.functional.conv2d(
            inp,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class QuanConv(BaseQuanConv):
    """
    Standard quantized convolution implementation.
    """
    
    def forward(self, inp):
        """
        Forward pass with adaptive quantization
        """
        if not self.quantization_enabled:
            return self._apply_convolution(inp, self.weight)
        
        # Validate setup
        if isinstance(self.quantize_a, PACT):
            assert self.alpha_setup_flag, "Alpha not setup for PACT quantization"
        assert self.curr_bitwidth is not None, "bitwidth is None"
        assert self.quantize_w is not None, "quantize_w is None"
        assert self.quantize_a is not None, "quantize_a is None"
        
        # Quantize weights
        w = self._quantize_weight()
        
        # Weight normalization (optional)
        if self.sat_weight_normalization:
            std = w.detach().std()
            if std > 0:
                w = w / std
        
        # Quantize activations
        x = self._quantize_activation(inp)
        
        # Convolution
        output = self._apply_convolution(x, w)
        
        return output



