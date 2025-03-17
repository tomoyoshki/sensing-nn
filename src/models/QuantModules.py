import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd


def generate_random_bitwidth(bitwidth_options):
    return bitwidth_options[torch.randint(0, len(bitwidth_options), (1,))]

# Fix this
class DynamicQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, nbit, alpha=None):
        if nbit == 32:
            ctx.save_for_backward(x, alpha)
            return x
        else:
            ctx.save_for_backward(x, alpha)
            # update this according to PACT implementation: https://github.com/KwangHoonAn/PACT/blob/master/module.py - slight difference right now
            # this is asymmetric quantization, we can also do symmetric quantization if needed
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
        w = w / (2 * maxv) + 0.5 # This is the quantization step in DoreFa
        w = 2 * quantize(w, nbit_w) - 1 # This is the dequantization step in DoreFa
        return w


class DoReFaA(nn.Module):
    def __init__(self):
        super(DoReFaA, self).__init__()

    def forward(self, inp, nbit_a, *args, **kwargs):
        """forward pass"""
        # Ideally there is no need to dequantize here, but try it out if accuracy is not good.
        return quantize(torch.clamp(inp, 0, 1), nbit_a, *args, **kwargs) #


class PACT(nn.Module):
    def __init__(self):
        super(PACT, self).__init__()

    def forward(self, inp, nbit_a, alpha, *args, **kwargs):
        """forward pass"""
        input = torch.clamp(inp, min=0, max=alpha.item()) #
        input_val = quantize(input, nbit_a, alpha)
        return input_val
    
# TODO: Make QuanConv a geenral class which access all different kind of Quantizations - Maybe its a cleaner way of doing it.
class QuanConv(nn.Module):

    layer_registry = {} # "Conv_1,2,3": QuanConv()
    layer_counter = 0

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        sat_weight_normalization = True,
        bias=True,
        has_offset=False,
        fix=False,
        batch_norm=False,
        layer_name=None,
    ):
        super(QuanConv, self).__init__()
        self.sat_weight_normalization = sat_weight_normalization
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
        self.curr_bitwidth = 32
        self.alpha_setup_flag = False
        self.alpha_common = nn.Parameter(torch.ones(1), requires_grad=True)
        nn.init.constant_(self.alpha_common, 10)

        # Put entry in layer_registry for each instance of QuanConv created
        self.layer_name = f"layer_{QuanConv.layer_counter}"
        QuanConv.layer_counter += 1
        QuanConv.layer_registry[self.layer_name] = self
        self.float_mode = False
        self.args = None

    def set_bitwidth(self, bitwidth):
        assert bitwidth <=32 and bitwidth > 1, "bitwidth should be between 2 and 32"
        self.curr_bitwidth = bitwidth

    def get_bitwidth(self):
        assert self.curr_bitwidth is not None, "bitwidth is None"
        return self.curr_bitwidth

    def alpha_setup(self, bitwidth_opts, switchable = True): #bitwidth_opts a list of different bit-width options [4,6,8]
        self.bitwidth_opts = bitwidth_opts
        self.alpha_setup_flag = True
        if switchable:
            self.alpha = nn.Parameter(
                torch.ones(len(bitwidth_opts)), requires_grad=True
            )
            # nn.init.constant_(self.alpha, 10)
        else:
            self.alpha = nn.Parameter(
                torch.ones(1), requires_grad=True
            )
            # nn.init.constant_(self.alpha, 10)

    def set_random_bitwidth(self):
        assert self.bitwidth_opts is not None, "bitwidth options not set"
        self.curr_bitwidth = self.bitwidth_opts[torch.randint(0, len(self.bitwidth_opts), (1,))]
    
    @classmethod
    def set_random_bitwidth_all(cls):
        """Set random bitwidth for all registered QuanConv layers"""
        for layer_name, layer in cls.layer_registry.items():
            layer.set_random_bitwidth()


    @classmethod
    def get_average_bitwidth(cls):
        total = len(cls.layer_registry)
        total_sum = 0
        for layer_name, layer in cls.layer_registry.items():
            total_sum += layer.set_random_bitwidth()
        return total_sum / total

        
    def setup_quantize_funcs(self, args):
        # classifier_config = args.dataset_config[args.model]
        quantization_config = args.dataset_config["quantization"]
        # quantization_config = args["quantization_config"]
        weight_quantization = quantization_config["weight_quantization"]
        activation_quantization = quantization_config["activation_quantization"]
        self.bitwidth_opts = quantization_config["bitwidth_options"]
        self.switchable_clipping = quantization_config["switchable_clipping"]
        self.sat_weight_normalization = quantization_config["sat_weight_normalization"]
        if weight_quantization == "dorefa":
            self.quantize_w = DoReFaW()
        else:
            raise NotImplementedError("Weight Quantization only with Dorefa")
        if activation_quantization == "dorefa":
            self.quantize_a = DoReFaA()
        elif activation_quantization == "pact":
            self.quantize_a = PACT()
            # self.bitwidth_opts = bitwidth_opts
            self.alpha_setup_flag = True
            if self.switchable_clipping:
                self.alpha = nn.Parameter(
                    torch.ones(len(self.bitwidth_opts)), requires_grad=True
                )
                # nn.init.constant_(self.alpha, 10)
            else:
                self.alpha = nn.Parameter(
                    torch.ones(1), requires_grad=True
                )
            # self.alpha_setup(bitwidth_opts)

    def get_alpha(self):
        # assert self.args is not None, "args not set in QuanConv"
        assert self.alpha_setup_flag, "alpha not setup, run setup_quantize_funcs before running model inference"

        if self.switchable_clipping:
            return self.alpha[self.bitwidth_opts.index(self.curr_bitwidth)]
        else:
            return self.alpha[0]

        

    def forward(self, inp):
        if not self.float_mode:
            assert self.alpha_setup_flag, "alpha not setup"
            assert self.curr_bitwidth is not None, "bitwidth is None"
            assert self.quantize_w is not None, "quantize_w is None"
            assert self.quantize_a is not None, "quantize_a is None"

            # idx_alpha = self.bitwidth_opts.index(self.curr_bitwidth)
            # alpha = torch.abs(self.alpha[idx_alpha])

            if self.curr_bitwidth <= 32:
                w = self.quantize_w(self.weight, self.curr_bitwidth)
                bias = self.bias
                # Adabits method of scaling weights I think may or maynot use it.
                if self.sat_weight_normalization:
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

                    bias = bias * weight_scale if bias is not None else None
                
                if isinstance(self.quantize_a, PACT):
                    # alpha = self.alpha[self.bitwidth_opts.index(self.curr_bitwidth)]
                    alpha = self.get_alpha()
                    x = self.quantize_a(inp, self.curr_bitwidth, alpha)
                elif isinstance(self.quantize_a, DoReFaA):
                    x = self.quantize_a(inp, self.curr_bitwidth)
                else:
                    raise ValueError(
                        "Only PACT and DoReFaA implmented for activation quantization"
                    )

                x = nn.functional.conv2d(
                    x,
                    w,
                    bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )
                return x
            else:
                raise ValueError(f"Invalid bitwidth {self.curr_bitwidth}")
        elif self.float_mode:
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
        else:
            raise ValueError(f"Invalid mode, self.float_mode is set to {self.float_mode} - should be either True or False")
            # else:
            #     assert self.curr_bitwidth == 32, "bitwidth is not 32"
            #     x = nn.functional.conv2d(
            #         inp,
            #         self.weight,
            #         self.bias,
            #         self.stride,
            #         self.padding,
            #         self.dilation,
            #         self.groups,
            #     )
            # return x
    

class CustomBatchNorm(nn.Module):
    layer_registry = {}
    layer_counter = 0
    def __init__(self, out_channels): #type can be float, switchable, or transitional
        super(CustomBatchNorm, self).__init__()
        self.out_channels = out_channels
        self.bn_float = nn.BatchNorm2d(out_channels)
        self.bns = None
        self.bitwidth_options = None # List of bitwidths
        self.switchable = False
        self.transitional = False

        self.layer_name = f"layer_{CustomBatchNorm.layer_counter}"
        CustomBatchNorm.layer_counter += 1
        CustomBatchNorm.layer_registry[self.layer_name] = self

        # Used for transitional BatchNorm
        self.prev_bitwidth = 32
        self.succ_bitwidth = 32
        self.floating_point = None
        # self.bn_type = None
        self.input_conv = None
        self.output_conv = None

    def set_corresponding_input_conv(self, input_conv):
        self.input_conv = input_conv

    def set_corresponding_input_output_convs(self, input_conv, output_conv):
        self.input_conv = input_conv
        self.output_conv = output_conv

    def set_corresponding_output_conv(self, output_conv):
        self.output_conv = output_conv

    def set_prev_bitwidth(self, prev_bitwidth):
        self.prev_bitwidth = prev_bitwidth

    def set_succ_bitwidth(self, succ_bitwidth):
        self.succ_bitwidth = succ_bitwidth
    
    def set_to_switchable(self, bitwidth_options): # bitwidth options is a list of bitwidths
        self.bitwidth_options = bitwidth_options
        if 32 not in self.bitwidth_options:
            self.bitwidth_options.append(32)
        self.bns = nn.ModuleDict({str(bitwidth): nn.BatchNorm2d(self.out_channels) for bitwidth in bitwidth_options})
        # for bitwidth in bitwidth_options:
        #     self.bns[bitwidth] = nn.BatchNorm2d(self.out_channels)
        self.switchable = True
        self.transitional = False
        self.floating_point = False

    def set_to_transitional(self, bitwidth_options):
        self.switchable = False
        self.transitional = True
        self.bitwidth_options = bitwidth_options
        if 32 not in self.bitwidth_options:
            self.bitwidth_options.append(32)
        self.bns = nn.ModuleDict()
        for bitwidth_pred in bitwidth_options:
            for bitwidth_succ in bitwidth_options:
                # self.bns[bitwidth_pred][bitwidth_succ] = nn.BatchNorm2d(self.out_channels)
                key = f"{bitwidth_pred}_{bitwidth_succ}"
                self.bns[key] = nn.BatchNorm2d(self.out_channels)
        self.floating_point = False

    def set_to_float(self):
        self.switchable = False
        self.transitional = False
        self.floating_point = True

    def forward(self, x):
        if self.switchable:
            # Switchable BatchNorm (Used in CoQuant, and AnyPrecision)
            assert self.bitwidth_options is not None, "bitwidth options not set"
            # assert self.input_conv is not None, "input conv not set"
            input_bitwidth = self.input_conv.curr_bitwidth if self.input_conv != None else 32
            # idx_alpha = self.bitwidth_options.index(self.input_conv.curr_bitwidth)
            # print(next(self.bns[str(input_bitwidth)].parameters()).device)
            # print(x.device)
            # breakpoint()
            return self.bns[str(input_bitwidth)](x)
        elif self.floating_point:
            # Traditional BatchNorm
            return self.bn_float(x)
        elif self.transitional:
            # Transitional BatchNorm (Used in Bit-Mixer)
            assert self.input_conv is not None, "input conv not set for self.transitional == True"
            input_bitwidth = self.input_conv.curr_bitwidth if self.input_conv != None else 32
            output_bitwidth = self.output_conv.curr_bitwidth if self.output_conv!= None else 32
            key = f"{input_bitwidth}_{output_bitwidth}"
            return self.bns[key](x)
        else:
            raise ValueError(f"Invalid BatchNorm type, valid options are switchable, "\
            "transitional, and floating_point. Make sure you have called either,\
            set_switchable, set_to_transitional, or set_to_float before running model inference. Currently "
            "self.floating_point is set to {self.floating_point},\
            self.switchable is set to {self.switchable}, and self.transitional is set to {self.transitional}")