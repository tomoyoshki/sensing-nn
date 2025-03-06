import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

# Fix this
class DynamicQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, nbit, alpha=None):
        if nbit == 32:
            ctx.save_for_backward(x, alpha)
            return x
        else:
            ctx.save_for_backward(x, alpha)
            scale = (
                (2**nbit - 1) if alpha is None else (2**nbit - 1) / alpha
            )
            # scale = (2**nbit -1)
            return torch.round(scale * x) / scale

    @staticmethod
    def backward(ctx, grad_output):
        # breakpoint()
        x, alpha = ctx.saved_tensors
        if alpha is None:
            grad_input = grad_output.clone()
            return grad_input, None, None
        else:
            lower_bound = x < 0
            upper_bound = x > alpha

            x_range = ~(lower_bound | upper_bound)
            grad_alpha = torch.sum(
                grad_output * torch.ge(x, alpha).float()
            ).view(-1)

            return grad_output * x_range.float(), None, grad_alpha
            # x, = ctx.saved_tensors
            # print("grad_output: ", grad_output)
            # grad_input = grad_output.clone()
            return grad_output, None, None
        


def quantize(x, nbit, alpha=None):
    return DynamicQuantizer.apply(x, nbit, alpha)


class DoReFaW(nn.Module):
    def __init__(self):
        super(DoReFaW, self).__init__()
        # self.quantize = DynmQuantizerAdabit.apply

    def forward(self, inp, nbit_w, *args, **kwargs):
        """forward pass"""
        w = torch.tanh(inp)
        maxv = torch.abs(w).max()
        w = w / (2 * maxv) + 0.5
        w = 2 * quantize(w, nbit_w) - 1
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
        input = torch.clamp(inp, min=0, max=alpha.item())
        input_val = quantize(input, nbit_a, alpha)
        return input_val
    

class QuanConv(nn.Module):
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
        has_offset=False,
        fix=False,
        batch_norm=False,
    ):
        super(QuanConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *kernel_size)
        )

        nn.init.kaiming_uniform_(
            self.weight, mode="fan_out", nonlinearity="relu"
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.quantize_w = None
        self.quantize_a = None
        self.curr_bitwidth = None
        self.alpha_setup_flag = False
        self.alpha_common = nn.Parameter(torch.ones(1), requires_grad=True)
        nn.init.constant_(self.alpha_common, 10)

    def alpha_setup(self, bitwidth_opts):
        self.bitwidth_opts = bitwidth_opts
        self.alpha_setup_flag = True
        self.alpha = nn.Parameter(
            torch.ones(len(bitwidth_opts)), requires_grad=True
        )

    def setup_quantize_funcs(self, quan_name_w="dorefa", quan_name_a="dorefa"):
        if quan_name_w == "dorefa":
            self.quantize_w = DoReFaW()
        else:
            raise NotImplementedError("Weight Quantization only with Dorefa")
        if quan_name_a == "dorefa":
            self.quantize_a = DoReFaA()
        elif quan_name_a == "pact":
            self.quantize_a = PACT()

    def forward(self, inp):
        assert self.alpha_setup_flag, "alpha not setup"
        assert self.curr_bitwidth is not None, "bitwidth is None"
        assert self.quantize_w is not None, "quantize_w is None"
        assert self.quantize_a is not None, "quantize_a is None"

        # idx_alpha = self.bitwidth_opts.index(self.curr_bitwidth)
        # alpha = torch.abs(self.alpha[idx_alpha])

        if self.curr_bitwidth < 32:
            w = self.quantize_w(self.weight, self.curr_bitwidth)
            weight_scale = (
                1.0
                / (
                    self.out_channels
                    * self.kernel_size[0]
                    * self.kernel_size[1]
                )
                ** 0.5
            )
            weight_scale = weight_scale / torch.std(w.detach())
            w = w * weight_scale

            bias = self.bias
            if bias is not None:
                bias = bias * weight_scale

            alpha = self.alpha[self.bitwidth_opts.index(self.curr_bitwidth)]
            if isinstance(self.quantize_a, PACT):
                x = self.quantize_a(inp, self.curr_bitwidth, alpha)
            elif isinstance(self.quantize_a, DoReFaA):
                x = self.quantize_a(inp, self.curr_bitwidth)
            else:
                raise ValueError(
                    "Only PACT and DoReFaA implmented for activation"
                )

            x = nn.functional.conv2d(
                x,
                w,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        else:
            assert self.curr_bitwidth == 32, "bitwidth is not 32"
            x = nn.functional.conv2d(
                inp,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        return x