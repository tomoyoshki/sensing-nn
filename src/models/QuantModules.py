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

class LSQQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, step_size, nbit, is_activation=False):
        qn = 0 if is_activation else -(2 ** (nbit - 1))
        qp = 2 ** nbit - 1 if is_activation else 2 ** (nbit - 1) - 1

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

    def forward(self, x):
        if not self.initialized:
            self.step_size.data.copy_(
                (2 * x.abs().mean() / (2 ** self.bitwidth - 1)).clamp(min=1e-6)
            )
            self.initialized = True

        return LSQQuantizer.apply(x, self.step_size, self.bitwidth, self.is_activation)

class LSQPlus(LSQ):
    def __init__(self, is_activation=False, bitwidth=8, shape=1, zero_point=True):
        super().__init__(is_activation, bitwidth, shape)
        self.zero_point = zero_point
        self.zp = nn.Parameter(torch.zeros(shape), requires_grad=zero_point)

    def forward(self, x):
        if not self.initialized:
            self.step_size.data.copy_(
                (2 * x.abs().mean() / (2 ** self.bitwidth - 1)).clamp(min=1e-6)
            )
            self.initialized = True
        x_q = x / self.step_size + self.zp
        qn = 0 if self.is_activation else -(2 ** (self.bitwidth - 1))
        qp = 2 ** self.bitwidth - 1 if self.is_activation else 2 ** (self.bitwidth - 1) - 1
        x_clipped = torch.clamp(x_q.round(), qn, qp)
        return (x_clipped - self.zp) * self.step_size


    
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
        teacher_selection_strategy = "weighted_activation_min", # "weighted_activation_min", "weight_min", "activation_min", "random", "weighted_activation_max", "weight_max", "activation_max"
        has_offset=False,
        fix=False,
        batch_norm=False,
        layer_name=None,
    ):
        super(QuanConv, self).__init__()
        self.teacher_selection_strategy = teacher_selection_strategy
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
        self.quantization_config = None
        self.si_weights = None
        self.best_teacher_bitwidth = None
        self.best_teacher_activation = None
        self.recent_student_activation = None
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
    quantization_config = args.dataset_config["quantization"]
    self.quantization_config = quantization_config

    weight_quantization = quantization_config["weight_quantization"]
    activation_quantization = quantization_config["activation_quantization"]
    self.bitwidth_opts = quantization_config["bitwidth_options"]
    self.switchable_clipping = quantization_config["switchable_clipping"]
    self.sat_weight_normalization = quantization_config["sat_weight_normalization"]

    # Weight quantization
    if weight_quantization == "dorefa":
        self.quantize_w = DoReFaW()
    elif weight_quantization == "lsq":
        self.quantize_w = LSQ(is_activation=False, bitwidth=self.bitwidth_opts[0])
    elif weight_quantization == "lsqplus":
        self.quantize_w = LSQPlus(is_activation=False, bitwidth=self.bitwidth_opts[0])
    else:
        raise NotImplementedError(f"Weight Quantization '{weight_quantization}' not supported yet")

    # Activation quantization
    if activation_quantization == "dorefa":
        self.quantize_a = DoReFaA()
    elif activation_quantization == "pact":
        self.quantize_a = PACT()
        self.alpha_setup_flag = True
        if self.switchable_clipping:
            self.alpha = nn.Parameter(torch.ones(len(self.bitwidth_opts)), requires_grad=True)
        else:
            self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
    elif activation_quantization == "lsq":
        self.quantize_a = LSQ(is_activation=True, bitwidth=self.bitwidth_opts[0])
        self.alpha_setup_flag = False
    elif activation_quantization == "lsqplus":
        self.quantize_a = LSQPlus(is_activation=True, bitwidth=self.bitwidth_opts[0])
        self.alpha_setup_flag = False
    else:
        raise NotImplementedError(f"Activation Quantization '{activation_quantization}' not supported yet")


    def get_alpha(self, bitwidth = None):
        # assert self.args is not None, "args not set in QuanConv"
        assert self.alpha_setup_flag, "alpha not setup, run setup_quantize_funcs before running model inference"
        bitwidth = self.curr_bitwidth if bitwidth is None else bitwidth
        if self.switchable_clipping:
            return self.alpha[self.bitwidth_opts.index(bitwidth)]
        else:
            return self.alpha[0]

    def get_weights_for_all_teachers(self,weight_scale=None):
        assert self.bitwidth_opts is not None, "bitwidth options not set"
        teacher_bitwidths = [bitwidth for bitwidth in self.bitwidth_opts if bitwidth > self.curr_bitwidth]
        teacher_weights = {}
        # breakpoint()
        for teacher_bitwidth in teacher_bitwidths:
            teacher_weights[teacher_bitwidth] = {}
            teacher_weight = self.quantize_w(self.weight, teacher_bitwidth)
            teacher_weights[teacher_bitwidth]['weight'] = teacher_weight * weight_scale if weight_scale is not None else teacher_weight
            if self.bias is not None:
                teacher_weights[teacher_bitwidth]['bias'] = self.bias * weight_scale if weight_scale is not None else self.bias
            else:
                teacher_weights[teacher_bitwidth]['bias'] = None
        return teacher_weights
    def _activation_min_sequential(self, x_student, input_teacher, teacher_weights, teacher_bitwidths):
        """Sequential implementation of activation_min strategy"""
        min_diff = float("inf")
        best_teacher_bitwidth = None
        best_activation = None
        
        for teacher_bitwidth in teacher_bitwidths:
            weights = teacher_weights[teacher_bitwidth]
            teacher_weight = weights['weight']
            teacher_bias = weights['bias']

            # Get teacher activation
            if isinstance(self.quantize_a, PACT):
                alpha = self.get_alpha(teacher_bitwidth)
                x_teacher = self.quantize_a(input_teacher, teacher_bitwidth, alpha)
            elif isinstance(self.quantize_a, DoReFaA):
                x_teacher = self.quantize_a(input_teacher, teacher_bitwidth)
            else:
                raise ValueError("Only PACT and DoReFaA implemented for activation quantization")

            # Get teacher output
            x_teacher = nn.functional.conv2d(
                x_teacher,
                teacher_weight,
                teacher_bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            
            activation_diff = x_student - x_teacher
            msroot = torch.sqrt(torch.mean(activation_diff**2))
            
            if msroot < min_diff:
                min_diff = msroot
                best_teacher_bitwidth = teacher_bitwidth
                best_activation = x_teacher
                
        return best_activation, best_teacher_bitwidth
    
    def _activation_min_parallel(self, x_student, input_teacher, teacher_weights, teacher_bitwidths):
        """
        Efficient batched implementation of activation_min strategy using a single convolution operation.
        
        This implementation handles grouped convolutions correctly by ensuring channel dimensions match.
        """
        batch_size, in_channels, height, width = input_teacher.shape
        num_bitwidths = len(teacher_bitwidths)
        
        # Step 1: Quantize activations for each bitwidth (this remains sequential as it's not the bottleneck)
        x_teacher_all = {}
        for teacher_bitwidth in teacher_bitwidths:
            if isinstance(self.quantize_a, PACT):
                alpha = self.get_alpha(teacher_bitwidth)
                x_teacher_all[teacher_bitwidth] = self.quantize_a(input_teacher, teacher_bitwidth, alpha)
            elif isinstance(self.quantize_a, DoReFaA):
                x_teacher_all[teacher_bitwidth] = self.quantize_a(input_teacher, teacher_bitwidth)
            else:
                raise ValueError("Only PACT and DoReFaA implemented for activation quantization")
        
        # Get sample weight to understand dimensions
        sample_weight = teacher_weights[teacher_bitwidths[0]]['weight']
        out_channels, weight_in_channels, kernel_h, kernel_w = sample_weight.shape
        
        # Check if we're using grouped convolution originally (self.groups > 1)
        original_groups = self.groups
        
        # IMPORTANT: We need a different approach when the original convolution uses groups
        if original_groups > 1:
            # We'll perform the convolutions separately and combine results at the end
            all_outputs = []
            for bitwidth in teacher_bitwidths:
                weight = teacher_weights[bitwidth]['weight']
                bias = teacher_weights[bitwidth]['bias']
                activation = x_teacher_all[bitwidth]
                
                # Perform individual convolution with correct grouping
                output = nn.functional.conv2d(
                    activation,
                    weight,
                    bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups
                )
                all_outputs.append(output)
            
            # Find the minimum difference
            min_diff = float("inf")
            best_teacher_bitwidth = None
            best_activation = None
            
            for i, bitwidth in enumerate(teacher_bitwidths):
                activation_diff = x_student - all_outputs[i]
                msroot = torch.sqrt(torch.mean(activation_diff**2))
                
                if msroot < min_diff:
                    min_diff = msroot
                    best_teacher_bitwidth = bitwidth
                    best_activation = all_outputs[i]
            
            return best_activation, best_teacher_bitwidth
        
        # The approach below is for standard convolutions (groups=1)
        # For each bitwidth, run a separate convolution
        all_outputs = []
        for bitwidth in teacher_bitwidths:
            weight = teacher_weights[bitwidth]['weight']
            bias = teacher_weights[bitwidth]['bias']
            activation = x_teacher_all[bitwidth]
            
            # Perform standard convolution
            output = nn.functional.conv2d(
                activation,
                weight,
                bias,
                self.stride, 
                self.padding,
                self.dilation,
                self.groups
            )
            all_outputs.append(output)
            
        # Find the minimum difference
        min_diff = float("inf")
        best_teacher_bitwidth = None
        best_activation = None
        
        for i, bitwidth in enumerate(teacher_bitwidths):
            activation_diff = x_student - all_outputs[i]
            msroot = torch.sqrt(torch.mean(activation_diff**2))
            
            if msroot < min_diff:
                min_diff = msroot
                best_teacher_bitwidth = bitwidth
                best_activation = all_outputs[i]
        
        return best_activation, best_teacher_bitwidth
    
    def weight_min(self, student_weight, input_teacher, teacher_weights, teacher_bitwidths, weight_scale=None):
        """
        Batched implementation of weight_min strategy to find the teacher weight with minimum difference.
        
        Args:
            student_weight: Quantized student weight
            input_teacher: Teacher input tensor
            teacher_weights: Dictionary of teacher weights for different bitwidths
            teacher_bitwidths: List of teacher bitwidths to consider
            weight_scale: Optional scaling factor for student weight
            
        Returns:
            best_activation: Teacher activation using the best weight
            best_teacher_bitwidth: Bitwidth with minimum weight difference
        """
        # Apply weight scale if provided
        if weight_scale is not None:
            student_weight = student_weight * weight_scale
        
        # Step 1: Get all teacher weights at once and compute differences in batch
        all_teacher_weights = []
        for bitwidth in teacher_bitwidths:
            all_teacher_weights.append(teacher_weights[bitwidth]['weight'])
        
        # Step 2: Stack student weight for easier comparison
        stacked_student_weight = student_weight.unsqueeze(0).expand(len(teacher_bitwidths), -1, -1, -1, -1)
        
        # Step 3: Stack all teacher weights
        stacked_teacher_weights = torch.stack(all_teacher_weights, dim=0)
        
        # Step 4: Compute all weight differences in parallel
        weight_diffs = stacked_student_weight - stacked_teacher_weights
        
        # Step 5: Compute RMSE for all weight differences at once
        msroots = torch.sqrt(torch.mean(weight_diffs**2, dim=(1, 2, 3, 4)))
        
        # Step 6: Find minimum difference
        min_diff_idx = torch.argmin(msroots)
        min_diff = msroots[min_diff_idx]
        best_teacher_bitwidth = teacher_bitwidths[min_diff_idx]
        
        # Step 7: Calculate activation for the best weight teacher
        if isinstance(self.quantize_a, PACT):
            alpha = self.get_alpha(best_teacher_bitwidth)
            x_teacher = self.quantize_a(input_teacher, best_teacher_bitwidth, alpha)
        elif isinstance(self.quantize_a, DoReFaA):
            x_teacher = self.quantize_a(input_teacher, best_teacher_bitwidth)
        else:
            raise ValueError("Only PACT and DoReFaA implemented for activation quantization")
        
        # Step 8: Compute final teacher activation using the best weight
        best_activation = nn.functional.conv2d(
            x_teacher,
            teacher_weights[best_teacher_bitwidth]['weight'],
            teacher_weights[best_teacher_bitwidth]['bias'],
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        
        return best_activation, best_teacher_bitwidth

    def get_best_teacher(self, x_student, input_teacher, teacher_weights, weight_scale, strategy="activation_min", student_weight = None):
        """
        Get the best teacher based on specified strategy.
        
        Args:
            x_student: Student activation output
            input_teacher: Teacher input
            teacher_weights: Dictionary of teacher weights for different bitwidths
            weight_scale: Weight normalization scale factor
            strategy: Strategy to select best teacher ("activation_min" or "weight_min")
            
        Returns:
            tuple: (best_teacher_activation, best_teacher_bitwidth)
        """
        activations = {}
        min_diff = float("inf")
        best_teacher_bitwidth = None
        best_activation = None

        if len(teacher_weights) == 0:
            # No teachers available, return student activation
            best_teacher_bitwidth = self.curr_bitwidth
            best_activation = x_student
        elif strategy == "activation_min":

            # best_activation, best_teacher_bitwidth = self._activation_min_sequential(x_student, input_teacher, teacher_weights, teacher_bitwidths = list(teacher_weights.keys()))

            best_activation, best_teacher_bitwidth = self._activation_min_parallel(x_student, input_teacher, teacher_weights, teacher_bitwidths = list(teacher_weights.keys()))

            # try:
            #     assert best_teacher_bitwidth == parallel_best_teacher_bitwidth, "Best teacher bitwidths do not match between sequential and parallel implementations"
            #     assert torch.allclose(best_activation, parallel_best_activation, atol=1e-6), "Best activations do not match between sequential and parallel implementations"
            #     assert parallel_best_activation.shape == best_activation.shape, "Best activations do not match in shape between sequential and parallel implementations"
            # except AssertionError as e:
            #     print(f"AssertionError: {e}")
            #     breakpoint()
            #     raise

        elif strategy == "weight_min":

            # Get student weight
            best_activation, best_teacher_bitwidth = self.weight_min(
                student_weight,
                input_teacher,
                teacher_weights,
                list(teacher_weights.keys()),
                weight_scale
            )


            # Find teacher with minimum weight difference
            # student_weight = self.quantize_w(self.weight, self.curr_bitwidth)
            # assert student_weight is not None, "student_weight is None, in QuanConv, get_best_teacher(), and teacher selection strategy is weight_min - you need to pass student_weight"
            # if weight_scale is not None:
            #     student_weight = student_weight * weight_scale
                
            # for teacher_bitwidth, weights in teacher_weights.items():
            #     teacher_weight = weights['weight']
            #     teacher_bias = weights['bias']
                
            #     weight_diff = student_weight - teacher_weight
            #     msroot = torch.sqrt(torch.mean(weight_diff**2))
                
            #     if msroot < min_diff:
            #         min_diff = msroot
            #         best_teacher_bitwidth = teacher_bitwidth
            
            # # Calculate activation for best weight teacher
            # if isinstance(self.quantize_a, PACT):
            #     alpha = self.get_alpha(best_teacher_bitwidth)
            #     x_teacher = self.quantize_a(input_teacher, best_teacher_bitwidth, alpha)
            # elif isinstance(self.quantize_a, DoReFaA):
            #     x_teacher = self.quantize_a(input_teacher, best_teacher_bitwidth)
                
            # x_teacher = nn.functional.conv2d(
            #     x_teacher,
            #     teacher_weights[best_teacher_bitwidth]['weight'],
            #     teacher_weights[best_teacher_bitwidth]['bias'],
            #     self.stride,
            #     self.padding,
            #     self.dilation,
            #     self.groups,
            # )
            # best_activation = x_teacher

        else:
            raise ValueError(f"Unknown teacher selection strategy: {strategy}")

        # Save best teacher info
        self.best_teacher_bitwidth = best_teacher_bitwidth
        self.best_teacher_activation = best_activation
        try:
            assert self.best_teacher_bitwidth is not None, "best_teacher_bitwidth is None in QuanConv, get_best_teacher()"
            assert self.best_teacher_activation is not None, "best_teacher_activation is None in QuanConv, get_best_teacher()"
        except AssertionError as e:
            print(f"AssertionError: {e}")
            breakpoint()
            raise

        return best_activation, best_teacher_bitwidth


    def forward(self, inp, teacher_input = None):
        if not self.float_mode:
            assert self.alpha_setup_flag, "alpha not setup"
            assert self.curr_bitwidth is not None, "bitwidth is None"
            assert self.quantize_w is not None, "quantize_w is None"
            assert self.quantize_a is not None, "quantize_a is None"

            # idx_alpha = self.bitwidth_opts.index(self.curr_bitwidth)
            # alpha = torch.abs(self.alpha[idx_alpha])
            weight_scale = None

            if self.curr_bitwidth <= 32:
                if isinstance(self.quantize_w, (LSQ, LSQPlus)):
                    self.quantize_w.bitwidth = self.curr_bitwidth
                    w = self.quantize_w(self.weight)
                else:
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
                    alpha = self.get_alpha()
                    x = self.quantize_a(inp, self.curr_bitwidth, alpha)
                elif isinstance(self.quantize_a, DoReFaA):
                    x = self.quantize_a(inp, self.curr_bitwidth)
                elif isinstance(self.quantize_a, (LSQ, LSQPlus)):
                    self.quantize_a.bitwidth = self.curr_bitwidth
                    x = self.quantize_a(inp)
                else:
                    raise ValueError("Unsupported activation quantization type. Use PACT, DoReFaA, LSQ, or LSQPlus.")

                # try:
                    # print(f"Input shape: {x.shape}")
                    # print(f"Weight shape: {w.shape}")
                x_student = nn.functional.conv2d(
                        x,
                        w,
                        bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                    )
                self.recent_student_activation = x_student
                if self.quantization_config["teacher_student"] and self.training:
                    assert teacher_input is not None, "teacher_input is None in QuanConv, forward()"
                    try:
                        teacher_weights = self.get_weights_for_all_teachers(weight_scale)
                        x_teacher, _ = self.get_best_teacher(
                                            x_student,
                                            teacher_input,
                                            teacher_weights,
                                            weight_scale,
                                            strategy=self.quantization_config["teacher_selection_strategy"],
                                            student_weight = w
                                        )
                        # breakpoint()
                        return x_student, x_teacher
                    except Exception as e:
                        print(f"Error in QuanConv forward pass (teacher_student section): {e}")
                        breakpoint()
                        raise
                else:
                    return x_student
                # except Exception as e:
                #     print(f"Error in QuanConv forward pass: {e}")
                #     breakpoint()
                #     raise
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

    def setup_quantize_funcs(self, args):
        quantization_config = args.dataset_config["quantization"]
        self.quantization_config = quantization_config
        self.bitwidth_options = quantization_config["bitwidth_options"]
        if quantization_config["bn_type"] == "float":
            self.set_to_float()
        elif quantization_config["bn_type"] == "transitional":
            self.set_to_transitional(quantization_config["bitwidth_options"])
        elif quantization_config["bn_type"] == "switch":
            self.set_to_switchable(quantization_config["bitwidth_options"])
        else:
            raise ValueError("\
                quantization_config['bn_type'] is not set as needed in the dataset config file. Valid options are float, transitional, switch.\
            ")
        
        if self.quantization_config["teacher_student"]:
            assert self.quantization_config["teacher_bn_type"] is not None, "teacher_bn_type not set in the quantization config, \
            please update the yaml file with options teacher_bn_type: switchable or float"
            if self.quantization_config["teacher_bn_type"] == "switchable":
                self.bns = nn.ModuleDict({str(bitwidth): nn.BatchNorm2d(self.out_channels) for bitwidth in self.bitwidth_options})
            elif self.quantization_config["teacher_bn_type"] == "float":
                self.bn_float = nn.BatchNorm2d(self.out_channels)
            else:
                raise ValueError(f"Invalid teacher_bn_type: {self.quantization_config['teacher_bn_type']}")

    def forward(self, x, teacher_input = None):
        x_student = x
        if self.switchable:
            # Switchable BatchNorm (Used in CoQuant, and AnyPrecision)
            assert self.bitwidth_options is not None, "bitwidth options not set"
            input_bitwidth = self.input_conv.curr_bitwidth if self.input_conv != None else 32
            x_student = self.bns[str(input_bitwidth)](x)
        elif self.floating_point:
            # Traditional BatchNorm
            x_student = self.bn_float(x)
        elif self.transitional:
            # Transitional BatchNorm (Used in Bit-Mixer)
            assert self.input_conv is not None, "input conv not set for self.transitional == True"
            input_bitwidth = self.input_conv.curr_bitwidth if self.input_conv != None else 32
            output_bitwidth = self.output_conv.curr_bitwidth if self.output_conv!= None else 32
            key = f"{input_bitwidth}_{output_bitwidth}"
            x_student = self.bns[key](x)
        else:
            raise ValueError(f"Invalid BatchNorm type, valid options are switchable, "\
            "transitional, and floating_point. Make sure you have called either,\
            set_switchable, set_to_transitional, or set_to_float before running model inference. Currently "
            "self.floating_point is set to {self.floating_point},\
            self.switchable is set to {self.switchable}, and self.transitional is set to {self.transitional}")
        
        if teacher_input is not None:
            # Training mode
            # Two cases student teacher enabled and disabled
            if self.quantization_config["teacher_bn_type"] == "float":
                # If no teacher bitwidth is set, use floating point
                x_teacher = self.bn_float(teacher_input)
            elif self.quantization_config["teacher_bn_type"] == "switchable":
                teacher_bitwidth = self.input_conv.best_teacher_bitwidth if self.input_conv is not None else 32
                x_teacher = self.bns[str(teacher_bitwidth)](teacher_input)
            else:
                raise ValueError(f"Invalid teacher_bn_type: {self.quantization_config['teacher_bn_type']} \
                                 - valid options are switchable, and float")
            return x_student, x_teacher
        else:
            # Inference mode
            return x_student

            