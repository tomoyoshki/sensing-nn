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
        assert self.curr_bitwidth is not None, "curr_bitwidth is None in _quantize_weight"
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


class QuanConvSplit(BaseQuanConv):
    """
    Quantized convolution with split-precision for upper/lower frequency bands.
    
    This class splits the input along the frequency dimension (last dimension),
    applies different quantization bitwidths to each half, then combines them
    using the mask+add approach for seamless boundary handling.
    
    Architecture:
        1. Split input at F/2: upper [B,C,H,F/2], lower [B,C,H,F/2]
        2. Quantize each half with different bitwidths
        3. Pad each quantized half with zeros to restore [B,C,H,F] shape
        4. Convolve each padded tensor with shared weight
        5. Add the outputs element-wise
    
    This approach preserves full spatial information at boundaries - the 
    convolution kernel can see across the split point, and the addition 
    reconstructs the complete information.
    
    Attributes:
        curr_bitwidth_upper (int): Bitwidth for upper half (frequencies 0 to F/2)
        curr_bitwidth_lower (int): Bitwidth for lower half (frequencies F/2 to F)
    
    Example:
        >>> conv = QuanConvSplit(in_channels=64, out_channels=128, kernel_size=3)
        >>> conv.setup_quantize_funcs(quant_config)
        >>> conv.set_bitwidth(bitwidth_upper=8, bitwidth_lower=4)
        >>> output = conv(input)  # input shape: [B, 64, H, F]
    """
    
    def _custom_init(self):
        """
        Custom initialization for split-precision quantization.
        
        Initializes dual bitwidth attributes for upper and lower frequency bands.
        
        Note: self.curr_bitwidth is kept as None to make it explicit that this
        class does not use a single bitwidth. Only curr_bitwidth_upper and 
        curr_bitwidth_lower should be used.
        """
        # Initialize dual bitwidths (None means not set yet)
        self.curr_bitwidth_upper = None
        self.curr_bitwidth_lower = None
        
        # Explicitly keep base class curr_bitwidth as None
        # This prevents unintended use of single-bitwidth code paths
        self.curr_bitwidth = None
        
        # Separate quantizers for upper and lower halves (initialized in _setup_quantizers)
        self.quantize_w_upper = None
        self.quantize_w_lower = None
        self.quantize_a_upper = None
        self.quantize_a_lower = None
    
    def _setup_quantizers(self):
        """
        Override to setup separate quantizers for upper and lower frequency bands.
        
        Creates four quantizer instances:
        - quantize_w_upper, quantize_w_lower: Weight quantizers
        - quantize_a_upper, quantize_a_lower: Activation quantizers
        
        Also calls parent to setup base quantizers (for potential compatibility).
        """
        # Call parent setup (sets self.quantize_w and self.quantize_a)
        super()._setup_quantizers()
        
        # Create separate quantizers for upper half
        if self.weight_quantization == "dorefa":
            self.quantize_w_upper = DoReFaW()
        elif self.weight_quantization == "lsq":
            self.quantize_w_upper = LSQ(is_activation=False, bitwidth=self.bitwidth_opts[0])
        elif self.weight_quantization == "lsqplus":
            self.quantize_w_upper = LSQPlus(is_activation=False, bitwidth=self.bitwidth_opts[0])
        
        if self.activation_quantization == "dorefa":
            self.quantize_a_upper = DoReFaA()
        elif self.activation_quantization == "pact":
            self.quantize_a_upper = PACT()
        elif self.activation_quantization == "lsq":
            self.quantize_a_upper = LSQ(is_activation=True, bitwidth=self.bitwidth_opts[0])
        elif self.activation_quantization == "lsqplus":
            self.quantize_a_upper = LSQPlus(is_activation=True, bitwidth=self.bitwidth_opts[0])
        
        # Create separate quantizers for lower half
        if self.weight_quantization == "dorefa":
            self.quantize_w_lower = DoReFaW()
        elif self.weight_quantization == "lsq":
            self.quantize_w_lower = LSQ(is_activation=False, bitwidth=self.bitwidth_opts[0])
        elif self.weight_quantization == "lsqplus":
            self.quantize_w_lower = LSQPlus(is_activation=False, bitwidth=self.bitwidth_opts[0])
        
        if self.activation_quantization == "dorefa":
            self.quantize_a_lower = DoReFaA()
        elif self.activation_quantization == "pact":
            self.quantize_a_lower = PACT()
        elif self.activation_quantization == "lsq":
            self.quantize_a_lower = LSQ(is_activation=True, bitwidth=self.bitwidth_opts[0])
        elif self.activation_quantization == "lsqplus":
            self.quantize_a_lower = LSQPlus(is_activation=True, bitwidth=self.bitwidth_opts[0])
    
    def set_bitwidth(self, bitwidth_upper, bitwidth_lower=None):
        """
        Set bitwidths for both upper and lower frequency bands.
        
        Updates the bitwidths in the separate upper/lower quantizers.
        
        Args:
            bitwidth_upper (int): Bitwidth for upper half (0 to F/2)
            bitwidth_lower (int, optional): Bitwidth for lower half (F/2 to F).
                                          If None, uses same as upper.
        
        Raises:
            AssertionError: If bitwidths are invalid or not in bitwidth_opts
        """
        # If lower not provided, use same as upper
        if bitwidth_lower is None:
            bitwidth_lower = bitwidth_upper
        
        # Validate bitwidths
        assert bitwidth_upper <= 32 and bitwidth_upper > 1, \
            "bitwidth_upper should be between 2 and 32"
        assert bitwidth_lower <= 32 and bitwidth_lower > 1, \
            "bitwidth_lower should be between 2 and 32"
        
        if bitwidth_upper != 32 and self.bitwidth_opts is not None:
            assert bitwidth_upper in self.bitwidth_opts, \
                f"bitwidth_upper {bitwidth_upper} not in bitwidth_options {self.bitwidth_opts}"
        if bitwidth_lower != 32 and self.bitwidth_opts is not None:
            assert bitwidth_lower in self.bitwidth_opts, \
                f"bitwidth_lower {bitwidth_lower} not in bitwidth_options {self.bitwidth_opts}"
        
        self.curr_bitwidth_upper = bitwidth_upper
        self.curr_bitwidth_lower = bitwidth_lower
        
        # Update LSQ/LSQPlus bitwidths for upper quantizers
        if bitwidth_upper != 32:
            if isinstance(self.quantize_w_upper, (LSQ, LSQPlus)):
                self.quantize_w_upper.update_bitwidth(bitwidth_upper)
            if isinstance(self.quantize_a_upper, (LSQ, LSQPlus)):
                self.quantize_a_upper.update_bitwidth(bitwidth_upper)
        
        # Update LSQ/LSQPlus bitwidths for lower quantizers
        if bitwidth_lower != 32:
            if isinstance(self.quantize_w_lower, (LSQ, LSQPlus)):
                self.quantize_w_lower.update_bitwidth(bitwidth_lower)
            if isinstance(self.quantize_a_lower, (LSQ, LSQPlus)):
                self.quantize_a_lower.update_bitwidth(bitwidth_lower)
        
        # NOTE: We intentionally do NOT set self.curr_bitwidth
        # It remains None to prevent silent invocation of single-bitwidth code paths
    
    def set_bitwidth_upper(self, bitwidth):
        """
        Set bitwidth for upper frequency band only.
        
        Args:
            bitwidth (int): Bitwidth for upper half
        """
        assert bitwidth <= 32 and bitwidth > 1, "bitwidth should be between 2 and 32"
        if bitwidth != 32 and self.bitwidth_opts is not None:
            assert bitwidth in self.bitwidth_opts, \
                f"bitwidth {bitwidth} not in bitwidth_options {self.bitwidth_opts}"
        
        self.curr_bitwidth_upper = bitwidth
        
        # Update LSQ/LSQPlus bitwidths for upper quantizers
        if bitwidth != 32:
            if isinstance(self.quantize_w_upper, (LSQ, LSQPlus)):
                self.quantize_w_upper.update_bitwidth(bitwidth)
            if isinstance(self.quantize_a_upper, (LSQ, LSQPlus)):
                self.quantize_a_upper.update_bitwidth(bitwidth)
        
        # NOTE: self.curr_bitwidth remains None intentionally
    
    def set_bitwidth_lower(self, bitwidth):
        """
        Set bitwidth for lower frequency band only.
        
        Args:
            bitwidth (int): Bitwidth for lower half
        """
        assert bitwidth <= 32 and bitwidth > 1, "bitwidth should be between 2 and 32"
        if bitwidth != 32 and self.bitwidth_opts is not None:
            assert bitwidth in self.bitwidth_opts, \
                f"bitwidth {bitwidth} not in bitwidth_options {self.bitwidth_opts}"
        
        self.curr_bitwidth_lower = bitwidth
        
        # Update LSQ/LSQPlus bitwidths for lower quantizers
        if bitwidth != 32:
            if isinstance(self.quantize_w_lower, (LSQ, LSQPlus)):
                self.quantize_w_lower.update_bitwidth(bitwidth)
            if isinstance(self.quantize_a_lower, (LSQ, LSQPlus)):
                self.quantize_a_lower.update_bitwidth(bitwidth)
    
    def get_bitwidth(self):
        """
        Get current bitwidths for both frequency bands.
        
        Returns:
            tuple: (upper_bitwidth, lower_bitwidth)
        """
        return (self.curr_bitwidth_upper, self.curr_bitwidth_lower)
    
    def set_uniform_bitwidth(self, bitwidth):
        """
        Set the same bitwidth for both upper and lower bands.
        
        Args:
            bitwidth (int): Bitwidth to use for both halves
        """
        self.set_bitwidth(bitwidth, bitwidth)
    
    def set_random_bitwidth(self):
        """
        Randomly select bitwidths for both bands from available options.
        
        Each band independently samples from bitwidth_opts.
        """
        assert self.bitwidth_opts is not None, "bitwidth options not set"
        bw_upper = self.bitwidth_opts[torch.randint(0, len(self.bitwidth_opts), (1,)).item()]
        bw_lower = self.bitwidth_opts[torch.randint(0, len(self.bitwidth_opts), (1,)).item()]
        self.set_bitwidth(bw_upper, bw_lower)
    
    def forward(self, inp):
        """
        Forward pass with split-precision quantization using mask+add approach.
        
        Splits input along frequency dimension, quantizes each half (both weights 
        AND activations) with different bitwidths, performs two separate convolutions,
        and adds the outputs.
        
        Key difference from standard QuanConv:
            - Weights are quantized TWICE at different bitwidths:
              * w_upper quantized at curr_bitwidth_upper
              * w_lower quantized at curr_bitwidth_lower
            - Each convolution uses differently-quantized weights
        
        Args:
            inp (torch.Tensor): Input tensor of shape [B, C, T, F] where:
                - B: batch size
                - C: input channels
                - T: time segments
                - F: frequency bins
        
        Returns:
            torch.Tensor: Output tensor of shape [B, C', T', F'] where:
                - C': out_channels
                - T', F': time segments and frequency bins after convolution
        
        Mathematical Justification:
            At boundaries, the convolution kernel sees contributions from both halves.
            By adding outputs:
            - output_upper contains contributions from upper (lower was zeros)
            - output_lower contains contributions from lower (upper was zeros)
            - sum = full convolution with mixed quantization
        """
        # Full precision mode (no quantization)
        if not self.quantization_enabled:
            return self._apply_convolution(inp, self.weight)
        
        # Validate setup
        if isinstance(self.quantize_a, PACT):
            assert self.alpha_setup_flag, "Alpha not setup for PACT quantization"
        assert self.curr_bitwidth_upper is not None, "bitwidth_upper is None"
        assert self.curr_bitwidth_lower is not None, "bitwidth_lower is None"
        assert self.quantize_w is not None, "quantize_w is None"
        assert self.quantize_a is not None, "quantize_a is None"
        
        # Split along frequency dimension (last dim)
        F = inp.shape[-1]
        F_half = F // 2
        inp_upper = inp[..., :F_half]      # Upper half [B, C, T, F/2]
        inp_lower = inp[..., F_half:]      # Lower half [B, C, T, F/2 or F/2+1 if odd]
        
        # ===================================================================
        # UPPER HALF: Quantize with upper_bitwidth using upper quantizers
        # ===================================================================
        # Quantize weight with upper quantizer
        if isinstance(self.quantize_w_upper, (LSQ, LSQPlus)):
            w_upper = self.quantize_w_upper(self.weight)
        else:
            w_upper = self.quantize_w_upper(self.weight, self.curr_bitwidth_upper)
        
        # Weight normalization if enabled
        if self.sat_weight_normalization:
            std = w_upper.detach().std()
            if std > 0:
                w_upper = w_upper / std
        
        # Quantize activation with upper quantizer
        if isinstance(self.quantize_a_upper, PACT):
            alpha_upper = self.get_alpha(self.curr_bitwidth_upper)
            x_upper_q = self.quantize_a_upper(inp_upper, self.curr_bitwidth_upper, alpha_upper)
        elif isinstance(self.quantize_a_upper, DoReFaA):
            x_upper_q = self.quantize_a_upper(inp_upper, self.curr_bitwidth_upper)
        elif isinstance(self.quantize_a_upper, (LSQ, LSQPlus)):
            x_upper_q = self.quantize_a_upper(inp_upper)
        else:
            raise ValueError(f"Unsupported activation quantization: {type(self.quantize_a_upper)}")
        
        # Pad lower half with zeros: [B, C, T, F/2] -> [B, C, T, F]
        zeros_lower = torch.zeros_like(inp_lower)
        x_upper_padded = torch.cat([x_upper_q, zeros_lower], dim=-1)  # [B, C, T, F]
        
        # Convolve upper padded with upper-bitwidth quantized weight
        output_upper = self._apply_convolution(x_upper_padded, w_upper)
        
        # ===================================================================
        # LOWER HALF: Quantize with lower_bitwidth using lower quantizers
        # ===================================================================
        # Quantize weight with lower quantizer
        if isinstance(self.quantize_w_lower, (LSQ, LSQPlus)):
            w_lower = self.quantize_w_lower(self.weight)
        else:
            w_lower = self.quantize_w_lower(self.weight, self.curr_bitwidth_lower)
        
        # Weight normalization if enabled
        if self.sat_weight_normalization:
            std = w_lower.detach().std()
            if std > 0:
                w_lower = w_lower / std
        
        # Quantize activation with lower quantizer
        if isinstance(self.quantize_a_lower, PACT):
            alpha_lower = self.get_alpha(self.curr_bitwidth_lower)
            x_lower_q = self.quantize_a_lower(inp_lower, self.curr_bitwidth_lower, alpha_lower)
        elif isinstance(self.quantize_a_lower, DoReFaA):
            x_lower_q = self.quantize_a_lower(inp_lower, self.curr_bitwidth_lower)
        elif isinstance(self.quantize_a_lower, (LSQ, LSQPlus)):
            x_lower_q = self.quantize_a_lower(inp_lower)
        else:
            raise ValueError(f"Unsupported activation quantization: {type(self.quantize_a_lower)}")
        
        # Pad upper half with zeros: [B, C, T, F/2] -> [B, C, T, F]
        zeros_upper = torch.zeros_like(inp_upper)
        x_lower_padded = torch.cat([zeros_upper, x_lower_q], dim=-1)  # [B, C, T, F]
        # breakpoint() # Check if the padding is correct
        
        # Convolve lower padded with lower-bitwidth quantized weight
        output_lower = self._apply_convolution(x_lower_padded, w_lower)
        
        # NOTE: self.curr_bitwidth remains None throughout (never modified)
        
        # ===================================================================
        # ADD outputs (preserves full information at boundaries)
        # ===================================================================
        output = output_upper + output_lower
        
        # ===================================================================
        # VALIDATION: Compare with floating point convolution (comment out later)
        # ===================================================================
        # Perform full precision convolution for shape validation
        # output_fp = self._apply_convolution(inp, self.weight)
        
        # # Ensure shapes match
        # assert output.shape == output_fp.shape, \
        #     f"Shape mismatch! Quantized split: {output.shape}, Floating point: {output_fp.shape}"
        
        # print(f"✓ Shape validation passed: {output.shape} == {output_fp.shape}")
        # ===================================================================
        
        return output


# class QuanConvImportance(BaseQuanConv):
#     """
#     Quantized convolution with learnable per-layer bit-width importance vectors.
    
#     Uses Gumbel-Softmax to sample bit-widths from learned importance distributions,
#     enabling automatic discovery of optimal per-layer quantization precision.
    
#     Features:
#     - Learnable importance vector for bit-width selection
#     - Gumbel-Softmax sampling with temperature annealing
#     - Soft mode (training): Differentiable weighted combination of all bitwidth outputs
#     - Hard mode (inference): Argmax selection of single best bitwidth
#     - Unique layer identification for tracking and visualization
    
#     Training vs Inference:
#     - During training, use 'soft' strategy: outputs from all bitwidths are weighted
#       by importance probabilities, enabling gradient flow to importance vectors
#     - During inference, use 'hard' strategy: argmax selects single best bitwidth
#     """
    
#     # Class variable for unique layer identification
#     _layer_counter = 0
    
#     def _custom_init(self):
#         """
#         Custom initialization for importance vector quantization.
        
#         Sets up:
#         - Unique layer ID and name
#         - Learnable importance vector (initialized uniformly)
#         """
#         # Assign unique layer ID
#         QuanConvImportance._layer_counter += 1
#         self.layer_id = QuanConvImportance._layer_counter
#         self.layer_name = f"conv_{self.layer_id}"
        
#         # Importance vector will be initialized after setup_quantize_funcs is called
#         # because we need to know the number of bitwidth options
#         self.importance_vector = None
#         self.sampling_strategy = None  # Default to soft for training (differentiable)
#         self.current_temperature = 1.0  # Default temperature (will be updated during training)
    
#     def setup_quantize_funcs(self, quantization_config):
#         """
#         Override to initialize importance vector after bitwidth options are known.
#         Also extracts sampling strategy from config.
#         """
#         super().setup_quantize_funcs(quantization_config)
        
#         # Extract sampling strategy from config (default to 'soft' for training)
#         iv_config = quantization_config.get('importance_vector', {})
#         self.sampling_strategy = iv_config.get('sampling_strategy', 'soft')
        
#         # Initialize importance vector uniformly: [1/N, 1/N, ..., 1/N]
#         num_bitwidths = len(self.bitwidth_opts)
#         # Use logits (unnormalized), which will be passed through softmax
#         # Initialize to zeros so softmax gives uniform distribution initially
#         self.importance_vector = nn.Parameter(
#             torch.zeros(num_bitwidths),
#             requires_grad=True
#         )

#         # Register as buffer so it moves with model.to(device) but doesn't require gradients
#         self.register_buffer(
#             'bitwidth_opts_tensor',
#             torch.tensor(self.bitwidth_opts, dtype=torch.float32)
#         )
        

    
    
#     def set_temperature(self, temperature):
#         """
#         Set the current temperature for Gumbel-Softmax sampling.
        
#         This is typically called by the TemperatureScheduler at the end of each epoch.
#         See models/Temperature_Scheduler.py for scheduler implementations.
        
#         Args:
#             temperature: Temperature value (higher = more exploration, lower = more exploitation)
#         """
#         self.current_temperature = temperature

#     def get_temperature(self):
#         """
#         Get the current temperature for Gumbel-Softmax sampling.
#         """
#         return self.current_temperature
    
#     def get_best_bitwidth(self):
#         """
#         Get the bitwidth with highest importance (argmax).
#         Used during inference/hard mode.
        
#         Returns:
#             int: Best bitwidth according to learned importance
#         """
#         assert self.importance_vector is not None, "Importance vector not initialized"
#         best_idx = torch.argmax(self.importance_vector).item()
#         return self.bitwidth_opts[best_idx]
    
#     def _forward_with_bitwidth(self, inp, bitwidth):
#         """
#         Compute forward pass with a specific bitwidth.
        
#         Args:
#             inp: Input tensor [batch_size, in_channels, height, width]
#             bitwidth: Specific bitwidth to use for quantization
        
#         Returns:
#             Output tensor [batch_size, out_channels, out_height, out_width]
#         """
#         # Set bitwidth for quantization functions
#         self.set_bitwidth(bitwidth)
        
#         # Quantize weights
#         w = self._quantize_weight()
        
#         # Weight normalization (optional)
#         if self.sat_weight_normalization:
#             std = w.detach().std()
#             if std > 0:
#                 w = w / std
        
#         # Quantize activations
#         x = self._quantize_activation(inp)
        
#         # Convolution
#         output = self._apply_convolution(x, w)
        
#         return output
    
#     def get_importance_distribution(self):
#         """
#         Get the current importance distribution (softmax of importance vector).
        
#         Returns:
#             dict: {'layer_name': str, 'distribution': tensor, 'bitwidth_options': list}
#         """
#         if self.importance_vector is None:
#             return None
        
#         distribution = torch.softmax(self.importance_vector, dim=0)
#         return {
#             'layer_name': self.layer_name,
#             'distribution': distribution.detach().cpu(),
#             'bitwidth_options': self.bitwidth_opts
#         }
    
#     @classmethod
#     def set_all_layers_mode(cls, model, mode):
#         """
#         Set the sampling strategy for all QuanConvImportance layers in a model.
        
#         Args:
#             model: PyTorch model containing QuanConvImportance layers
#             mode: Sampling strategy:
#                 - 'best_bitwidth': Use argmax of importance vector (inference/validation)
#                 - 'soft_gumbel_softmax': Sample from importance distribution (training)
#                 - 'uniform_sampling': Sample uniformly at random (validation baseline)
#         """
#         valid_modes = ['best_bitwidth', 'soft_gumbel_softmax', 'uniform_sampling']
#         assert mode in valid_modes, f"Mode must be one of {valid_modes}, got {mode}"
        
#         for module in model.modules():
#             if isinstance(module, cls):
#                 module.sampling_strategy = mode
    
#     @classmethod
#     def get_all_best_bitwidths(cls, model):
#         """
#         Get the best bitwidth (argmax of importance) for all layers.
        
#         Useful for logging and visualization during inference.
        
#         Args:
#             model: PyTorch model containing QuanConvImportance layers
        
#         Returns:
#             dict: {layer_name: best_bitwidth}
#         """
#         best_bitwidths = {}
#         for module in model.modules():
#             if isinstance(module, cls):
#                 best_bitwidths[module.layer_name] = module.get_best_bitwidth()
#         return best_bitwidths
    
#     @classmethod
#     def get_all_importance_distributions(cls, model):
#         """
#         Get importance distributions for all QuanConvImportance layers in a model.
        
#         This is a class method that collects softmax distributions from all instances
#         of QuanConvImportance in the given model, useful for visualization and analysis.
        
#         Args:
#             model: PyTorch model containing QuanConvImportance layers
        
#         Returns:
#             dict: {layer_name: {'distribution': tensor, 'bitwidth_options': list}}
#         """
#         distributions = {}
#         for module in model.modules():
#             if isinstance(module, cls):
#                 dist_info = module.get_importance_distribution()
#                 if dist_info is not None:
#                     distributions[dist_info['layer_name']] = {
#                         'distribution': dist_info['distribution'],
#                         'bitwidth_options': dist_info['bitwidth_options']
#                     }
        
#         return distributions
    
#     @classmethod
#     def collect_all_importance_vectors(cls, model):
#         """
#         Collect all importance vectors from QuanConvImportance layers in a model.
        
#         Args:
#             model: PyTorch model containing QuanConvImportance layers
        
#         Returns:
#             dict: {layer_name: importance_vector_tensor}
#         """
#         importance_vectors = {}
#         for module in model.modules():
#             if isinstance(module, cls):
#                 if hasattr(module, 'importance_vector') and module.importance_vector is not None:
#                     importance_vectors[module.layer_name] = module.importance_vector.detach().cpu()
        
#         return importance_vectors
    
    
#     @staticmethod
#     def plot_importance_distributions(importance_dists, bitwidth_options, epoch):
#         """
#         Create a bar plot showing learned importance distributions for each layer.
        
#         Args:
#             importance_dists: Dict of {layer_name: {'distribution': tensor, 'bitwidth_options': list}}
#             bitwidth_options: List of bitwidth options (for reference)
#             epoch: Current epoch number (for title)
        
#         Returns:
#             matplotlib.figure.Figure: The created figure
#         """
#         import matplotlib.pyplot as plt
#         import numpy as np
        
#         num_layers = len(importance_dists)
#         if num_layers == 0:
#             # Create empty figure if no importance vectors
#             fig, ax = plt.subplots(figsize=(8, 6))
#             ax.text(0.5, 0.5, 'No importance vectors found', 
#                    ha='center', va='center', fontsize=14)
#             ax.set_title(f'Importance Distributions (Epoch {epoch + 1})')
#             return fig
        
#         # Create subplots: arrange in grid
#         cols = min(3, num_layers)
#         rows = (num_layers + cols - 1) // cols
        
#         fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        
#         # Flatten axes for easier indexing
#         if num_layers == 1:
#             axes = [axes]
#         elif rows == 1 or cols == 1:
#             axes = axes.flatten()
#         else:
#             axes = axes.flatten()
        
#         # Plot each layer's distribution
#         for idx, (layer_name, dist_info) in enumerate(sorted(importance_dists.items())):
#             ax = axes[idx]
            
#             distribution = dist_info['distribution'].numpy()
#             bw_opts = dist_info['bitwidth_options']
            
#             # Create bar plot
#             x_pos = np.arange(len(bw_opts))
#             bars = ax.bar(x_pos, distribution, alpha=0.7, color='steelblue', edgecolor='navy')
            
#             # Highlight the maximum probability
#             max_idx = np.argmax(distribution)
#             bars[max_idx].set_color('orange')
#             bars[max_idx].set_alpha(0.9)
            
#             # Labels and formatting
#             ax.set_xlabel('Bitwidth', fontsize=10, fontweight='bold')
#             ax.set_ylabel('Probability', fontsize=10, fontweight='bold')
#             ax.set_title(f'{layer_name}', fontsize=11, fontweight='bold')
#             ax.set_xticks(x_pos)
#             ax.set_xticklabels([f'{bw}' for bw in bw_opts])
#             ax.set_ylim([0, 1.0])
#             ax.grid(True, alpha=0.3, axis='y')
            
#             # Add text annotation for max probability
#             ax.text(max_idx, distribution[max_idx] + 0.05, 
#                    f'{distribution[max_idx]:.2f}',
#                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
#         # Hide unused subplots
#         for idx in range(num_layers, len(axes)):
#             axes[idx].axis('off')
        
#         fig.suptitle(f'Learned Importance Distributions (Epoch {epoch + 1})', 
#                     fontsize=14, fontweight='bold', y=0.995)
#         plt.tight_layout()
        
#         return fig

    
#     def sample_bitwidth_uniform(self):
#         """
#         Sample a bitwidth uniformly at random from bitwidth options.
#         """
#         idx = torch.randint(0, len(self.bitwidth_opts), (1,)).item()
#         return self.bitwidth_opts[idx]
    
#     def forward(self, inp):
#         """
#         Forward pass with importance-based bitwidth selection.
        
#         Two modes:
#         - 'hard' (inference): Use argmax to select single best bitwidth.
        
#         Args:
#             inp: Input tensor [batch_size, in_channels, height, width]
        
#         Returns:
#             Output tensor [batch_size, out_channels, out_height, out_width]
#         """
#         if not self.quantization_enabled:
#             return self._apply_convolution(inp, self.weight)
        
#         # Validate setup
#         if isinstance(self.quantize_a, PACT):
#             assert self.alpha_setup_flag, "Alpha not setup for PACT quantization"
#         assert self.importance_vector is not None, "Importance vector not initialized"
#         assert self.quantize_w is not None, "quantize_w is None"
#         assert self.quantize_a is not None, "quantize_a is None"
        
#         # Validate mode vs model.training state
#         assert self.sampling_strategy is not None, "sampling_strategy is None"
#         if self.sampling_strategy == 'best_bitwidth':
#             # ================================================================
#             # BEST BITWIDTH MODE: Use argmax to select single best bitwidth
#             # ================================================================
#             best_bitwidth = self.get_best_bitwidth()
#             output = self._forward_with_bitwidth(inp, best_bitwidth)
#         elif self.sampling_strategy == 'soft_gumbel_softmax':
#             # ================================================================
#             # HARD GUMBEL SOFTMAX MODE: Sample from importance vector
#             # Temperature is managed by TemperatureScheduler.step()
#             # ================================================================
#             soft_Weights = F.gumbel_softmax(self.importance_vector,
#              tau=self.current_temperature, hard=False)

#             outputs = []
#             for bitwidth in self.bitwidth_opts:
#                 output = self._forward_with_bitwidth(inp, bitwidth)
#                 outputs.append(output)

#             stacked_outputs = torch.stack(outputs, dim=0)
#             probs_expanded = soft_Weights.view(-1, 1, 1, 1,1)
#             output = (stacked_outputs * probs_expanded).sum(dim=0)

#         elif self.sampling_strategy == 'uniform_sampling':
#             # ================================================================
#             # UNIFORM SAMPLING MODE (Validation): Sample uniformly at random
#             # ================================================================
#             selected_bitwidth = self.sample_bitwidth_uniform()
#             output = self._forward_with_bitwidth(inp, selected_bitwidth)
#         else:
#             raise ValueError(f"Invalid sampling strategy: {self.sampling_strategy}, \
#              must be one of: best_bitwidth, hard_gumbel_softmax, uniform_sampling")
#         return output


