import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd


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
        if nbit_w == 32:
            return inp
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
        if nbit_a == 32:
            return inp
        input = torch.clamp(inp, min=0, max=alpha)  # Use tensor alpha directly
        input_val = quantize(input, nbit_a, alpha)
        return input_val


class QuanConv(nn.Module):
    """
    Clean, simplified version of QuanConv with only essential features:
    - DoReFa, PACT, LSQ quantization methods
    - Alpha setup for PACT
    - Adaptive quantization (dynamic bitwidth)
    - Simple bitwidth setting mechanism
    - Float mode support
    - Config-based setup
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
        weight_quantization=None,  # Will be set from config if None
        activation_quantization=None,  # Will be set from config if None
        bitwidth_options=None,  # Will be set from config if None
        switchable_clipping=None,  # Will be set from config if None
        sat_weight_normalization=None,  # Will be set from config if None
    ):
        super(QuanConv, self).__init__()
        
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
        
        # Quantization setup (can be set via config later)
        self.weight_quantization = weight_quantization
        self.activation_quantization = activation_quantization
        self.bitwidth_opts = bitwidth_options
        self.switchable_clipping = switchable_clipping
        self.sat_weight_normalization = sat_weight_normalization
        
        # Current bitwidth (adaptive quantization)
        self.curr_bitwidth = 32  # Default to full precision
        
        # Float mode flag
        self.float_mode = False
        
        # Initialize quantizers (will be set up later via setup_quantize_funcs or _setup_quantizers)
        self.quantize_w = None
        self.quantize_a = None
        self.alpha_setup_flag = False
        self.alpha = None
        self.quantization_config = None
        
        # Setup quantization functions if parameters provided directly
        if weight_quantization is not None and activation_quantization is not None:
            self._setup_quantizers()
            # Alpha setup for PACT
            if activation_quantization == "pact" and bitwidth_options is not None:
                self.alpha_setup(bitwidth_options, switchable=switchable_clipping if switchable_clipping is not None else True)
    
    def setup_quantize_funcs(self, args):
        """
        Setup quantization functions from config (like original QuanConv)
        
        Args:
            args: Object with dataset_config attribute containing quantization config
        """
        quantization_config = args.dataset_config["quantization"]
        self.quantization_config = quantization_config
        
        weight_quantization = quantization_config["weight_quantization"]
        activation_quantization = quantization_config["activation_quantization"]
        self.bitwidth_opts = quantization_config["bitwidth_options"]
        self.switchable_clipping = quantization_config["switchable_clipping"]
        self.sat_weight_normalization = quantization_config["sat_weight_normalization"]
        
        # Store for _setup_quantizers
        self.weight_quantization = weight_quantization
        self.activation_quantization = activation_quantization
        
        # Setup quantizers
        self._setup_quantizers()
        
        # Alpha setup for PACT
        if activation_quantization == "pact":
            self.alpha_setup_flag = True
            if self.switchable_clipping:
                # Alpha shape: [len(bitwidth_opts), 1, 1, 1]
                self.alpha = nn.Parameter(torch.ones(len(self.bitwidth_opts), 1, 1, 1), requires_grad=True)
            else:
                # Single alpha with shape [1, 1, 1, 1]
                self.alpha = nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=True)
            nn.init.constant_(self.alpha, 10.0)
        elif activation_quantization in ["lsq", "lsqplus"]:
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
    
    def alpha_setup(self, bitwidth_opts, switchable=True):
        """
        Setup alpha parameter for PACT activation quantization
        
        Args:
            bitwidth_opts: List of bitwidth options [4, 6, 8, ...]
            switchable: If True, use different alpha per bitwidth; else use single alpha
        """
        self.bitwidth_opts = bitwidth_opts
        self.alpha_setup_flag = True
        if switchable:
            # Alpha shape: [len(bitwidth_opts), 1, 1, 1]
            self.alpha = nn.Parameter(
                torch.ones(len(bitwidth_opts), 1, 1, 1), requires_grad=True
            )
            nn.init.constant_(self.alpha, 10.0)
        else:
            # Single alpha with shape [1, 1, 1, 1]
            self.alpha = nn.Parameter(
                torch.ones(1, 1, 1, 1), requires_grad=True
            )
            nn.init.constant_(self.alpha, 10.0)
    
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
        if bitwidth != 32:
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
    
    def set_float_mode(self, value=True):
        """Set float mode - when True, uses full precision without quantization"""
        self.float_mode = value
    
    def forward(self, inp):
        """
        Forward pass with adaptive quantization
        
        Args:
            inp: Input tensor of shape [batch_size, in_channels, height, width]
        
        Returns:
            Output tensor of shape [batch_size, out_channels, out_height, out_width]
        """
        # Float mode: bypass quantization entirely
        if self.float_mode:
            return nn.functional.conv2d(
                inp,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        
        # Validate setup
        if isinstance(self.quantize_a, PACT):
            assert self.alpha_setup_flag, "Alpha not setup for PACT quantization"
        assert self.curr_bitwidth is not None, "bitwidth is None"
        assert self.quantize_w is not None, "quantize_w is None"
        assert self.quantize_a is not None, "quantize_a is None"
        
        # Full precision mode (bitwidth == 32 but not float_mode)
        if self.curr_bitwidth == 32:
            return nn.functional.conv2d(
                inp,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        
        # Quantized mode
        # Quantize weights
        if isinstance(self.quantize_w, (LSQ, LSQPlus)):
            w = self.quantize_w(self.weight)
        else:
            w = self.quantize_w(self.weight, self.curr_bitwidth)
        
        bias = self.bias
        
        # Weight normalization (optional)
        if self.sat_weight_normalization:
            std = w.detach().std()
            if std > 0:
                w = w / std
        
        # Quantize activations
        if isinstance(self.quantize_a, PACT):
            alpha = self.get_alpha()
            x = self.quantize_a(inp, self.curr_bitwidth, alpha)
        elif isinstance(self.quantize_a, DoReFaA):
            x = self.quantize_a(inp, self.curr_bitwidth)
        elif isinstance(self.quantize_a, (LSQ, LSQPlus)):
            x = self.quantize_a(inp)
        else:
            raise ValueError(
                f"Unsupported activation quantization type: {type(self.quantize_a)}"
            )
        
        # Convolution
        output = nn.functional.conv2d(
            x,
            w,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        
        return output


class DynamicQuanConv(nn.Module):
    """
    Clean dynamic quantization convolution layer.
    Supports:
      - DoReFa / LSQ / LSQPlus for weight quantization
      - DoReFa / PACT / LSQ / LSQPlus for activation quantization
      - Dynamic per-forward bitwidth selection
      - Switchable PACT α per-bitwidth
      - Optional weight normalization
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
        weight_quantization="dorefa",     # "dorefa", "lsq", "lsqplus"
        activation_quantization="pact",    # "dorefa", "pact", "lsq", "lsqplus"
        bitwidth_options=[4, 6, 8],
        switchable_clipping=True,
        sat_weight_normalization=True,
    ):
        super(DynamicQuanConv, self).__init__()

        # ------------------------------
        # convolution parameters
        # ------------------------------
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        # raw conv parameters
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, mode="fan_out", nonlinearity="relu")

        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # ------------------------------
        # quantization settings
        # ------------------------------
        self.weight_quantization = weight_quantization
        self.activation_quantization = activation_quantization
        self.bitwidth_opts = bitwidth_options
        self.switchable_clipping = switchable_clipping
        self.sat_weight_normalization = sat_weight_normalization

        # dynamic bitwidth
        self.curr_bitwidth = 32    # default full-precision

        # quantizers
        self.quantize_w = None
        self.quantize_a = None

        # PACT α
        self.alpha_setup_flag = False
        self.alpha = None

        # setup quantizers
        self._setup_quantizers()

        # setup α for PACT
        if self.activation_quantization == "pact":
            self._setup_alpha(self.bitwidth_opts, self.switchable_clipping)

    # ------------------------------------------------------------------
    # Setup quantizers
    # ------------------------------------------------------------------
    def _setup_quantizers(self):

        # weight quantizers
        if self.weight_quantization == "dorefa":
            self.quantize_w = DoReFaW()
        elif self.weight_quantization == "lsq":
            self.quantize_w = LSQ(is_activation=False, bitwidth=self.bitwidth_opts[0])
        elif self.weight_quantization == "lsqplus":
            self.quantize_w = LSQPlus(is_activation=False, bitwidth=self.bitwidth_opts[0])
        else:
            raise ValueError(f"Unsupported weight quantization: {self.weight_quantization}")

        # activation quantizers
        if self.activation_quantization == "dorefa":
            self.quantize_a = DoReFaA()
        elif self.activation_quantization == "pact":
            self.quantize_a = PACT()
        elif self.activation_quantization == "lsq":
            self.quantize_a = LSQ(is_activation=True, bitwidth=self.bitwidth_opts[0])
        elif self.activation_quantization == "lsqplus":
            self.quantize_a = LSQPlus(is_activation=True, bitwidth=self.bitwidth_opts[0])
        else:
            raise ValueError(f"Unsupported activation quantization: {self.activation_quantization}")

    # ------------------------------------------------------------------
    # Setup PACT α
    # ------------------------------------------------------------------
    def _setup_alpha(self, bitwidth_opts, switchable):
        self.alpha_setup_flag = True

        if switchable:
            # α per-bitwidth (shape = [num_bw, 1,1,1])
            self.alpha = nn.Parameter(torch.ones(len(bitwidth_opts), 1, 1, 1), requires_grad=True)
        else:
            # single α (shape = [1,1,1,1])
            self.alpha = nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=True)

        nn.init.constant_(self.alpha, 10.0)

    # ------------------------------------------------------------------
    # PACT alpha getter
    # ------------------------------------------------------------------
    def get_alpha(self, bitwidth):

        if not self.alpha_setup_flag:
            raise RuntimeError("PACT alpha not initialized!")

        if self.switchable_clipping:
            if bitwidth in self.bitwidth_opts:
                idx = self.bitwidth_opts.index(bitwidth)
                return self.alpha[idx]
            return self.alpha[0]  # fallback
        else:
            return self.alpha[0]

    # ------------------------------------------------------------------
    # Bitwidth control
    # ------------------------------------------------------------------
    def set_bitwidth(self, bitwidth):
        assert 2 <= bitwidth <= 32
        if bitwidth != 32:
            assert bitwidth in self.bitwidth_opts

        self.curr_bitwidth = bitwidth

        # update LSQ/LSQPlus quantizers (only for quantized modes)
        if bitwidth != 32:
            if isinstance(self.quantize_w, (LSQ, LSQPlus)):
                self.quantize_w.update_bitwidth(bitwidth)

            if isinstance(self.quantize_a, (LSQ, LSQPlus)):
                self.quantize_a.update_bitwidth(bitwidth)

    def set_random_bitwidth(self):
        bw = self.bitwidth_opts[torch.randint(0, len(self.bitwidth_opts), (1,)).item()]
        self.set_bitwidth(bw)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, inp, bitwidth=None):

        # dynamic bitwidth override
        if bitwidth is not None:
            self.set_bitwidth(bitwidth)

        bw = self.curr_bitwidth

        # -----------------------------------------
        # FP32 mode
        # -----------------------------------------
        if bw == 32:
            return nn.functional.conv2d(
                inp, self.weight, self.bias, self.stride,
                self.padding, self.dilation, self.groups
            )

        # -----------------------------------------
        # Quantized mode
        # -----------------------------------------

        # ---- quantize weights
        if isinstance(self.quantize_w, (LSQ, LSQPlus)):
            w = self.quantize_w(self.weight)
        else:
            # DoReFa: needs (tensor, nbit)
            w = self.quantize_w(self.weight, bw)

        # ---- optional weight normalization
        if self.sat_weight_normalization:
            std = w.detach().std()
            if std > 0:
                w = w / std

        # ---- quantize activations
        if isinstance(self.quantize_a, PACT):
            alpha = self.get_alpha(bw)
            x = self.quantize_a(inp, bw, alpha)
        elif isinstance(self.quantize_a, DoReFaA):
            x = self.quantize_a(inp, bw)
        else:
            x = self.quantize_a(inp)

        # ---- conv
        return nn.functional.conv2d(
            x, w, self.bias, self.stride,
            self.padding, self.dilation, self.groups
        )
