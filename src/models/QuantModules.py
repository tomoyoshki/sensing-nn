import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import logging
import random

class StatTracker:
    def __init__(self):
        pass

class TileImportanceVector:
    """Manages importance vectors for tile-based quantization of a convolutional layer"""
    
    def __init__(self, num_tiles, bitwidth_options, temperature=1.0, enable_grad_scaling=True):
        """
        Args:
            num_tiles (int): Number of tiles in the conv layer
            bitwidth_options (list): Available bitwidth options
            temperature (float): Temperature parameter for Gumbel Softmax
        """
        self.num_tiles = num_tiles
        self.bitwidth_options = bitwidth_options
        self.num_options = len(bitwidth_options)
        self.temperature = temperature
                # Create importance vectors for each tile
        self.importance_logits = nn.Parameter(
            torch.zeros(num_tiles, self.num_options),
            requires_grad=True
        )
        self.enable_grad_scaling = enable_grad_scaling
        
        # Initialize with uniform probabilities
        nn.init.uniform_(self.importance_logits, -0.1, 0.1)
        
        # Keep track of sampled bitwidths for each tile
        self.current_samples = None

        if self.enable_grad_scaling:
            self.importance_logits.register_hook(self._gradient_scaling_hook)

        self.grad_scaling_vector = None
        self.batch_grad_scaling = None
        self.target_device = None

    def set_batch_gradient_scaling(self, scaling_vector):
        """
        Set gradient scaling vector for current batch
        
        Args:
            scaling_vector (Tensor): Gradient scaling vector for current batch
                Shape can be [batch_size, num_tiles, num_options]
        """
        self.batch_grad_scaling = scaling_vector
        
    def _gradient_scaling_hook(self, grad):
        """
        Hook function to scale gradients during backward pass
        Now handles batch-specific scaling vectors
        
        Args:
            grad (Tensor): Original gradient [num_tiles, num_options]
            
        Returns:
            Tensor: Scaled gradient
        """
        if not self.enable_grad_scaling or self.batch_grad_scaling is None:
            return grad
            
        # Calculate mean gradient scaling across batch dimension
        # breakpoint()
        # grad = grad.to(self.target_device)
        mean_scaling = self.batch_grad_scaling.mean(dim=0).cpu()  # [num_tiles, num_options]
        
        # Scale the gradients using the mean scaling vector
        # breakpoint()
        scaled_grad = grad * mean_scaling
        return scaled_grad
        
    def sample_bitwidths(self, scaling_vector=None, hard=True):
        """
        Sample new bitwidths for all tiles using Gumbel Softmax
        
        Args:
            scaling_vector (Tensor, optional): Vector to scale importance logits before sampling
            hard (bool): If True, use hard sampling (one-hot), else soft sampling
            
        Returns:
            Tensor: Sampled probabilities for each tile's bitwidth
        """
        if scaling_vector is not None:
            # breakpoint()
            # Ensure scaling_vector is on the same device as importance_logits
            scaling_vector = scaling_vector.to(self.importance_logits.device)
            
            # Handle both per-option and per-tile-per-option scaling
            if scaling_vector.dim() == 1:
                # Broadcast scaling_vector to match importance_logits shape
                scaling_vector = scaling_vector.expand(self.num_tiles, -1)
            
            # Apply scaling
            # breakpoint()
            scaled_logits = self.importance_logits * scaling_vector
        else:
            scaled_logits = self.importance_logits

        samples = F.gumbel_softmax(
            scaled_logits,
            tau=self.temperature,
            hard=hard,
            dim=-1
        )
        
        return samples
    
    def to(self, device):
        """
        Explicitly move importance_logits to the specified device
        """
        self.target_device = device
        self.importance_logits = self.importance_logits.to(device)
        return self

    # def get_tile_bitwidths(self, scaling_vector=None):
    #     """
    #     Sample new bitwidth values for each tile
        
    #     Args:
    #         scaling_vector (Tensor, optional): Vector to scale importance logits before sampling
            
    #     Returns:
    #         Tensor: Sampled bitwidth values for each tile
    #     """
    #     # Always sample new bitwidths
    #     samples = self.sample_bitwidths(scaling_vector)
    #     # breakpoint()
    #     # Convert one-hot samples to actual bitwidth values
    #     bitwidth_tensor = torch.tensor(self.bitwidth_options).to(samples.device).to(samples.dtype)
    #     tile_bitwidths = torch.matmul(samples, bitwidth_tensor)
        
    #     return tile_bitwidths
    
    def get_tile_bitwidths(self, scaling_vector=None):
        """
        Sample new bitwidth values for each tile for each sample in the batch
        
        Args:
            scaling_vector: Tensor of shape [batch_size, num_tiles, num_options] to scale importance logits
                
        Returns:
            Tensor: Sampled bitwidth values for each tile for each sample [batch_size, num_tiles]
        """
        # Check if we have batch-specific scaling vectors
        # breakpoint()
        if scaling_vector is not None and scaling_vector.dim() == 3:
            batch_size = scaling_vector.shape[0]
            
            # Process each sample in the batch separately
            all_tile_bitwidths = []
            
            for sample_idx in range(batch_size):
                # Get the scaling vector for this sample
                sample_scaling = scaling_vector[sample_idx]
                
                # Sample bitwidths using Gumbel softmax
                samples = self.sample_bitwidths(sample_scaling)
                
                # Convert one-hot samples to actual bitwidth values
                # bitwidth_tensor = torch.tensor(self.bitwidth_options).to(samples.device).to(samples.dtype)
                # sample_tile_bitwidths = torch.matmul(samples, bitwidth_tensor)
                
                all_tile_bitwidths.append(samples)
            
            # Stack the results for all samples
            # Shape: [batch_size, num_tiles]
            tile_bitwidths = torch.stack(all_tile_bitwidths, dim=0)
            # breakpoint()
            
        else:
            # Without batch-specific scaling, use the original implementation
            tile_bitwidths = self.sample_bitwidths(scaling_vector)
            # breakpoint()
            # Convert one-hot samples to actual bitwidth values
            # bitwidth_tensor = torch.tensor(self.bitwidth_options).to(samples.device).to(samples.dtype)
            # tile_bitwidths = torch.matmul(samples, bitwidth_tensor)
        
        return tile_bitwidths
    
    def update_temperature(self, epoch, max_epochs, min_temp=0.1):
        """
        Anneal the temperature parameter
        
        Args:
            epoch (int): Current epoch
            max_epochs (int): Maximum number of epochs
            min_temp (float): Minimum temperature value
        """
        self.temperature = max(
            min_temp,
            1.0 * (1 - epoch / max_epochs)
        )

    def get_probabilities(self):
        """
        Get the current probabilities for each tile's bitwidth
        
        Returns:
            Tensor: Current probabilities
        """
        return F.softmax(self.importance_logits, dim=-1)


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

        if nbit_w == 32:
            return inp

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
        if nbit_a == 32:
            return inp
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
        self.sequence_length = None

        self.tile_bitwidths = None
        self.fixed_tiled_bitwidths = False
        self.tile_mode = False


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

    def set_bitwidth_with_importance_vector(cls):
        logging.info(f"Setting bitwidth using tile importance vector")
        for layer_name, layer in cls.layer_registry.items():
            if layer.tile_mode:
                # Get the tile importance vector
                tile_importance_vector = layer.tile_importance.get_tile_bitwidths(scaling_vector = None)
                
                # Set the bitwidth for each tile
                layer.tile_bitwidths = tile_importance_vector
            else:
                # If not using tiles, set a single bitwidth
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
        self.quantization_config = quantization_config
        self.sequence_length = args.dataset_config["seq_len"]
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

        self.number_of_tiles = quantization_config["number_of_tiles"]
        assert self.number_of_tiles%2 == 0, "number_of_tiles should be even"
        self.tile_importance = TileImportanceVector(
            num_tiles=self.number_of_tiles,
            bitwidth_options=self.bitwidth_opts,
            enable_grad_scaling=quantization_config["enable_grad_scaling"]
        )

        self.tile_mode = quantization_config["tile_mode"]
        if self.tile_mode:
            self.current_bitwidths_tile = []
        

    @classmethod
    def move_tile_importance_to_device(cls, device):
        """
        Move tile importance vector to the specified device
        """
        for layer_name, layer in cls.layer_registry.items():
            assert layer.tile_importance is not None, f"Tile importance vector is None for layer {layer_name}"
            layer.tile_importance.to(device)


        





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


    def forward_vanilla_and_distillation(self, inp, teacher_input = None):
        assert self.alpha_setup_flag, "alpha not setup"
        assert self.curr_bitwidth is not None, "bitwidth is None"
        assert self.quantize_w is not None, "quantize_w is None"
        assert self.quantize_a is not None, "quantize_a is None"

        # idx_alpha = self.bitwidth_opts.index(self.curr_bitwidth)
        # alpha = torch.abs(self.alpha[idx_alpha])
        weight_scale = None

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
                return x_student, None
            # except Exception as e:
            #     print(f"Error in QuanConv forward pass: {e}")
            #     breakpoint()
            #     raise
        else:
            raise ValueError(f"Invalid bitwidth {self.curr_bitwidth}")


    def get_tile_dimensions(self, inp):
        num_tiles = self.number_of_tiles
        assert num_tiles % 2 == 0, "Number of tiles should be even"
        
        batch_size, in_channels, height, width = inp.shape
        
        # Handle cases where dimensions are smaller than requested tiles
        height_tiles = min(2 if num_tiles > 1 else 1, height)
        width_tiles = min(num_tiles // height_tiles, width)
        
        # Calculate tile dimensions
        tile_height = height // height_tiles
        tile_width = width // width_tiles
        
        # Create a dictionary containing all necessary information for tiling
        tile_info = {
            'tile_height': tile_height,
            'tile_width': tile_width,
            'height_tiles': height_tiles,
            'width_tiles': width_tiles,
            # Add slicing indices for easy tensor operations
            'height_indices': torch.arange(height_tiles) * tile_height,
            'width_indices': torch.arange(width_tiles) * tile_width
        }
        
        return tile_info



    def get_scaling_vectors(self, input_tiles, tile_info=None):
        """
        Calculate scaling vectors for tile importance based on tile characteristics
        
        Args:
            input_tiles: List of input tiles, each with shape [batch_size, channels, height, tile_width]
            tile_info: Optional dictionary with tile information
            
        Returns:
            Scaling vectors of shape [batch_size, num_tiles, num_bitwidth_options]
        """
        function_type = self.quantization_config["importance_vector_scaling_function"]
        if function_type == "none":
            return None
        
        # Get batch size from first tile
        batch_size = input_tiles[0].shape[0]
        num_tiles = len(input_tiles)
        
        if function_type == "energy":
            # Calculate energy of each tile and normalize
            gamma = 1.0  # Hyperparameter
            
            # Get min and max bitwidths from config
            min_bits = min(self.quantization_config["bitwidth_options"])
            max_bits = max(self.quantization_config["bitwidth_options"])
            bit_options = torch.tensor(self.quantization_config["bitwidth_options"], 
                                    device=input_tiles[0].device)
            
            # Calculate energy for each tile for each sample in the batch
            tile_energies = []
            total_energy = 0
            
            for tile in input_tiles:
                # Calculate energy (sum of squares)
                # Shape: [batch_size]
                energy = torch.sum(tile**2, dim=[1, 2, 3])
                tile_energies.append(energy)
                total_energy += energy
            
            # Stack tile energies
            # Shape: [batch_size, num_tiles]
            tile_energies = torch.stack(tile_energies, dim=1)
            
            # Shape: [batch_size, 1]
            total_energy = total_energy.unsqueeze(1)
            
            # Normalize tile energies
            # Shape: [batch_size, num_tiles]
            normalized_energies = tile_energies / (total_energy + 1e-8)  # Add small epsilon for numerical stability
            
            # Prepare bitwidth scaling term
            # Shape: [num_bitwidth_options]
            bitwidth_scale = 2 * (bit_options - min_bits) / (max_bits - min_bits + 1e-8) - 1
            
            # Calculate scaling vectors
            # Shape: [batch_size, num_tiles, num_bitwidth_options]
            energy_term = 2 * normalized_energies - 1
            
            # Expand dimensions for broadcasting
            # Shape: [batch_size, num_tiles, 1]
            energy_term = energy_term.unsqueeze(-1)
            
            # Shape: [1, 1, num_bitwidth_options]
            bitwidth_scale = bitwidth_scale.view(1, 1, -1)
            
            # Final scaling vectors
            # Shape: [batch_size, num_tiles, num_bitwidth_options]
            scaling_vectors = torch.exp(gamma * energy_term * bitwidth_scale)
            
            return scaling_vectors
        else:
            raise NotImplementedError(f"Scaling function {function_type} not implemented")

    # def get_scaling_vectors(self, input_tiles, tile_info=None):
    #     """
    #     Calculate scaling vectors for tile importance based on input characteristics
        
    #     Args:
    #         inp: Input tensor of shape [batch_size, channels, height, width]
    #         tile_info: Optional dictionary with tile information
            
    #     Returns:
    #         Scaling vectors of shape [batch_size, num_tiles, num_bitwidth_options]
    #     """
    #     function_type = self.quantization_config["importance_vector_scaling_function"]
    #     if function_type  == "none":
    #         return None
        
    #     # batch_size, channels, height, width = inp.shape
    #     num_tiles = self.number_of_tiles
    #     target_device = input_tiles[0].device
        
    #     # If tile_info is not provided, calculate it
    #     # if tile_info is None:
    #     #     _, tile_info = self.split_input_into_tiles(inp)
        
    #     if function_type == "energy":
    #         # Calculate energy of each tile and normalize
    #         gamma = 1.0  # Hyperparameter
            
    #         # Get min and max bitwidths from config
    #         min_bits = min(self.quantization_config["bitwidth_options"])
    #         max_bits = max(self.quantization_config["bitwidth_options"])
    #         bit_options = torch.tensor(self.quantization_config["bitwidth_options"], 
    #                                 device=target_device)
            
    #         # Calculate energy for each tile for each sample in the batch
    #         tile_energies = []
            
    #         for tile in input_tiles:
    #             # Extract the current tile boundaries
    #             # start_idx = tile_info['start_indices'][tile_idx]
    #             # end_idx = tile_info['end_indices'][tile_idx]
                    
    #             # # Extract the current tile
    #             # tile = inp[:, :, :, start_idx:end_idx]
                
    #             # Calculate energy (sum of squares)
    #             # Shape: [batch_size]
    #             energy = torch.sum(tile**2, dim=[1, 2, 3])
    #             tile_energies.append(energy)
            
    #         # Stack tile energies
    #         # Shape: [batch_size, num_tiles]
    #         tile_energies = torch.stack(tile_energies, dim=1)
            
    #         # Calculate total energy per sample
    #         # Shape: [batch_size, 1]
    #         total_energy = torch.sum(inp**2, dim=[1, 2, 3]).unsqueeze(1)
            
    #         # Normalize tile energies
    #         # Shape: [batch_size, num_tiles]
    #         normalized_energies = tile_energies / (total_energy + 1e-8)  # Add small epsilon for numerical stability
            
    #         # Prepare bitwidth scaling term
    #         # Shape: [num_bitwidth_options]
    #         bitwidth_scale = 2 * (bit_options - min_bits) / (max_bits - min_bits + 1e-8) - 1
            
    #         # Calculate scaling vectors
    #         # Shape: [batch_size, num_tiles, num_bitwidth_options]
    #         energy_term = 2 * normalized_energies - 1
            
    #         # Expand dimensions for broadcasting
    #         # Shape: [batch_size, num_tiles, 1]
    #         energy_term = energy_term.unsqueeze(-1)
            
    #         # Shape: [1, 1, num_bitwidth_options]
    #         bitwidth_scale = bitwidth_scale.view(1, 1, -1)
            
    #         # Final scaling vectors
    #         # Shape: [batch_size, num_tiles, num_bitwidth_options]
    #         scaling_vectors = torch.exp(gamma * energy_term * bitwidth_scale)
            
    #         return scaling_vectors
    #     else:
    #         raise NotImplementedError(f"Scaling function {function_type} not implemented")



    def get_scaling_vectors_grad(self, input_tiles, tile_info=None):
        """
        Calculate scaling vectors for tile importance based on tile characteristics
        
        Args:
            input_tiles: List of input tiles, each with shape [batch_size, channels, height, tile_width]
            tile_info: Optional dictionary with tile information
            
        Returns:
            Scaling vectors of shape [batch_size, num_tiles, num_bitwidth_options]
        """
        function_type = self.quantization_config["importance_vector_scaling_function"]
        if function_type == "none":
            return None
        
        # Get batch size from first tile
        batch_size = input_tiles[0].shape[0]
        num_tiles = len(input_tiles)
        
        if function_type == "energy":
            # Calculate energy of each tile and normalize
            gamma = 1.0  # Hyperparameter
            
            # Get min and max bitwidths from config
            min_bits = min(self.quantization_config["bitwidth_options"])
            max_bits = max(self.quantization_config["bitwidth_options"])
            bit_options = torch.tensor(self.quantization_config["bitwidth_options"], 
                                    device=input_tiles[0].device)
            
            # Calculate energy for each tile for each sample in the batch
            tile_energies = []
            total_energy = 0
            
            for tile in input_tiles:
                # Calculate energy (sum of squares)
                # Shape: [batch_size]
                energy = torch.sum(tile**2, dim=[1, 2, 3])
                tile_energies.append(energy)
                total_energy += energy
            
            # Stack tile energies
            # Shape: [batch_size, num_tiles]
            tile_energies = torch.stack(tile_energies, dim=1)
            
            # Shape: [batch_size, 1]
            total_energy = total_energy.unsqueeze(1)
            
            # Normalize tile energies
            # Shape: [batch_size, num_tiles]
            normalized_energies = tile_energies / (total_energy + 1e-8)  # Add small epsilon for numerical stability
            
            # Prepare bitwidth scaling term
            # Shape: [num_bitwidth_options]
            bitwidth_scale = 2 * (bit_options - min_bits) / (max_bits - min_bits + 1e-8) - 1
            
            # Calculate scaling vectors
            # Shape: [batch_size, num_tiles, num_bitwidth_options]
            energy_term = 2 * normalized_energies - 1
            
            # Expand dimensions for broadcasting
            # Shape: [batch_size, num_tiles, 1]
            energy_term = energy_term.unsqueeze(-1)
            
            # Shape: [1, 1, num_bitwidth_options]
            bitwidth_scale = bitwidth_scale.view(1, 1, -1)
            
            # Final scaling vectors
            # Shape: [batch_size, num_tiles, num_bitwidth_options]
            scaling_vectors = torch.exp(gamma * energy_term * bitwidth_scale)
            
            return scaling_vectors
        else:
            raise NotImplementedError(f"Scaling function {function_type} not implemented")


    def split_input_into_tiles(self, inp):
        """
        Split input tensor into tiles along the width dimension
        
        Args:
            inp: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            List of tiles, each with shape [batch_size, channels, height, tile_width]
            Dictionary with tile information
        """
        batch_size, in_channels, height, width = inp.shape
        num_tiles = self.number_of_tiles
        
        # Calculate base tile width and remainder
        tile_width = width // num_tiles
        remainder = width % num_tiles
        
        # List to store tiles
        tiles = []
        
        # Dictionary to store tile info
        tile_info = {
            'tile_widths': [],
            'start_indices': [],
            'end_indices': []
        }
        
        # Split input into tiles
        start_idx = 0
        for tile_idx in range(num_tiles):
            # Add 1 extra column to tiles that handle remainder
            current_tile_width = tile_width + (1 if tile_idx < remainder else 0)
            
            # Extract current tile
            end_idx = start_idx + current_tile_width
            tile = inp[:, :, :, start_idx:end_idx]
            
            # Store tile and information
            tiles.append(tile)
            tile_info['tile_widths'].append(current_tile_width)
            tile_info['start_indices'].append(start_idx)
            tile_info['end_indices'].append(end_idx)
            
            # Update starting index for next tile
            start_idx = end_idx
        
        return tiles, tile_info
    

    def sample_bitwidth_allocations(self, input_tiles, tile_info):
        """
        Sample bitwidth allocations for each tile in each sample
        
        Args:
            inp: Input tensor of shape [batch_size, channels, height, width]
            tile_info: Dictionary with tile information
            
        Returns:
            Tensor of bitwidth allocations with shape [batch_size, num_tiles]
        """
    
        # batch_size = inp.shape[0]
        num_tiles = self.number_of_tiles
        
        # Calculate scaling vectors based on tile importance

        scaling_vectors = self.get_scaling_vectors(input_tiles, tile_info)

        if self.quantization_config["enable_grad_scaling"]:
            scaling_vectors_grad = self.get_scaling_vectors_grad(input_tiles, tile_info)
            # Combine scaling vectors and gradients
            self.tile_importance.set_batch_gradient_scaling(scaling_vector=scaling_vectors_grad)
        
        # Get per-sample, per-tile bitwidth allocations
        # Shape: [batch_size, num_tiles]
        tile_bitwidths = self.tile_importance.get_tile_bitwidths(scaling_vector=scaling_vectors)
        
        return tile_bitwidths

    def get_quantized_weights(self, bitwidth):
        """
        Get quantized weights for a specific bitwidth
        
        Args:
            bitwidth: Bitwidth to quantize weights
            
        Returns:
            Quantized weights
        """


        w = self.quantize_w(self.weight, bitwidth)
        if self.sat_weight_normalization:
            weight_scale = (1.0 / (self.out_channels * self.kernel_size[0] * self.kernel_size[1])) ** 0.5
            weight_scale = weight_scale / torch.std(w.detach())
            w = w * weight_scale
            bias = self.bias * weight_scale if self.bias is not None else None
        else:
            bias = self.bias
            
        return w, bias
    

    def check_if_reshaping_is_correct(self, stacked_activations, stacked_activation_dict, batch_size, num_tiles, num_bitwidths, in_channels, height, tile_width):
        """
        Verify that the reshaping of stacked activations maintains the correct order.
        
        Args:
            stacked_activations: Tensor of shape [batch_size, num_tiles*num_bitwidths*in_channels, height, tile_width]
            stacked_activation_dict: Dictionary of tensors {tile_idx: {bitwidth: tensor}}
            batch_size: Batch size
            num_tiles: Number of tiles
            num_bitwidths: Number of bitwidths
            in_channels: Number of input channels
            height: Height of each tile
            tile_width: Width of each tile
            
        Returns:
            bool: True if reshaping is correct, False otherwise
        """

        bitwidth_options = self.bitwidth_opts

        for tile_idx in range(num_tiles):
            for bitwidth_idx, bitwidth in enumerate(bitwidth_options):
                # Get the tensor from the dictionary
                tensor = stacked_activation_dict[tile_idx][bitwidth]
                # Calculate the expected index in the reshaped tensor
                expected_index = (
                    tile_idx * num_bitwidths * in_channels
                    + bitwidth_idx * in_channels
                )

                # Check if the reshaped tensor matches the original tensor
                reshaped_tensor = stacked_activations[:, expected_index:expected_index + in_channels, :, :]
                if not torch.allclose(tensor, reshaped_tensor):
                    print(f"Mismatch at tile {tile_idx}, bitwidth {bitwidth}, expected index {expected_index}")
                    return False
        
        # print("Reshaping is correct!")
        return True


    @classmethod
    def set_fixed_tiled_bitwidths(cls):
        """
        Set fixed bitwidths for all registered QuanConv layers.
        
        This method is called to set the bitwidths for all layers in the model.
        """
        logging.info("Setting fixed bitwidths for all registered QuanConv layers.")
        for layer_name, layer in cls.layer_registry.items():
            if layer.tile_mode:
                layer.tile_bitwidths = layer.tile_importance.get_tile_bitwidths(scaling_vector=None)
                layer.fixed_tiled_bitwidths = True
                # print(f"Layer {layer_name} fixed bitwidths set to {layer.current_bitwidths_tile}")
            else:
                # pick random bitwidth from layer.bitwidth_opts
                layer.curr_bitwidth = random.choice(layer.bitwidth_opts)

    def forward_tiled_parallel(self, inp):
        batch_size, in_channels, height, width = inp.shape

        # this can be optimized if number of tiles is always divisible to widht
        input_tiles, tile_info = self.split_input_into_tiles(inp)

        tile_widths = tile_info['tile_widths']
        # all tile_width need to be same
        assert len(set(tile_widths)) == 1, "All tile widths should be the same"
        tile_width = tile_widths[0]

        num_tiles = len(input_tiles)
        # breakpoint()
        # Step 2: Sample bitwidth allocations for each tile in each sample
        if self.training:
            tile_bitwidth_allocation = self.sample_bitwidth_allocations(input_tiles, tile_info) #TODO: We should just be directly passing tiles not the whole input
            self.tile_bitwidths = tile_bitwidth_allocation
        else:
            if not self.fixed_tiled_bitwidths:
                tile_bitwidth_allocation = self.tile_importance.get_tile_bitwidths(scaling_vector=None)
        # Save for reference
                self.tile_bitwidths = tile_bitwidth_allocation
            else:
                assert self.tile_bitwidths is not None, "tile_bitwidths is None, in QuanConv, forward_tiled_parallel()"
        # breakpoint()
        # Step 3: Process all tiles in parallel with their bitwidth allocations
        batch_size, in_channels, height, _ = inp.shape

        unique_bitwidths = torch.tensor(self.bitwidth_opts, device=inp.device)
        num_bitwidths = len(self.bitwidth_opts)

        total_number_of_groups = self.groups * num_bitwidths * self.number_of_tiles

        x_quantized = {}
        for i, tile in enumerate(input_tiles):
            tile_quanitized = {}
            for bitwidth in self.bitwidth_opts:
                if isinstance(self.quantize_a, PACT):
                    alpha = self.get_alpha(bitwidth)
                    tile_quanitized[bitwidth] = self.quantize_a(tile, bitwidth, alpha)
                elif isinstance(self.quantize_a, DoReFaA):
                    tile_quanitized[bitwidth] = self.quantize_a(tile, bitwidth)
                else:
                    raise ValueError("Only PACT and DoReFaA implemented for activation quantization")
            x_quantized[i] = tile_quanitized

        w_quantized = {}
        b_quantized = {}
        for bitwidth in self.bitwidth_opts:
            w, bias = self.get_quantized_weights(bitwidth)
            w_quantized[bitwidth] = w
            if bias is not None:
                b_quantized[bitwidth] = bias

            

        channels_per_group = in_channels // self.groups

        out_channels_per_group = self.out_channels // self.groups

        stacked_activations = []
        stacked_activation_dict = {}
        for tile_idx in range(len(input_tiles)):
            stacked_activation_dict[tile_idx] = {}
            for bitwidth in self.bitwidth_opts:
                stacked_activations.append(x_quantized[tile_idx][bitwidth])
                stacked_activation_dict[tile_idx][bitwidth] = x_quantized[tile_idx][bitwidth]

        # shape: [batch_size, num_tiles*num_bitwidths*in_channels, height, width]
        stacked_activations = torch.stack(stacked_activations, dim=1)
        # breakpoint()
        stacked_activations = stacked_activations.reshape(
            batch_size,
            num_tiles * num_bitwidths * in_channels,
            height,
            tile_width,
        )
        # self.check_if_reshaping_is_correct(stacked_activations, stacked_activation_dict, batch_size, \
        #                               num_tiles, num_bitwidths, in_channels, height, tile_width)
        
        # breakpoint()
        stacked_weights = []

        # Stack in the same order as activations: tile 0 bitwidth 0, tile 0 bitwidth 1, etc.
        for tile_idx in range(num_tiles):
            for bitwidth in self.bitwidth_opts:
                stacked_weights.append(w_quantized[bitwidth])

        # Concat all weights
        # Shape: [num_tiles*num_bitwidths*out_channels, in_channels/self.groups, kernel_h, kernel_w]
        stacked_weights = torch.cat(stacked_weights, dim=0)

        # Reshape weights to maintain group correspondence
        stacked_weights = stacked_weights.reshape(
            num_tiles * num_bitwidths * self.groups,  # New number of groups
            self.out_channels // self.groups,         # Original out channels per group
            in_channels // self.groups,               # Original in channels per group
            self.kernel_size[0],                      # kernel height
            self.kernel_size[1]                       # kernel width
        ).reshape(
            -1,                                       # Combined groups and out channels
            in_channels // self.groups,               # In channels per group
            self.kernel_size[0],                      # kernel height
            self.kernel_size[1]                       # kernel width
        )
        # breakpoint()

        # Stack all biases in the same order
        if self.bias is not None:
            stacked_biases = []
            for tile_idx in range(num_tiles):
                for bitwidth in self.bitwidth_opts:
                    stacked_biases.append(b_quantized[bitwidth])

        # Concat all biases
        # Shape: [num_tiles*num_bitwidths*out_channels]
            stacked_biases = torch.cat(stacked_biases, dim=0)

        output = F.conv2d(
            stacked_activations,
            stacked_weights,
            bias=stacked_biases if self.bias is not None else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=total_number_of_groups
        )
        
        output_batch_size, out_channels, output_height, output_tile_width = output.shape

        output = output.reshape(
            batch_size,
            num_tiles,
            num_bitwidths,
            self.out_channels // self.groups,
            output_height,
            output_tile_width
        )

        if self.tile_bitwidths.ndim == 2:
            mask = self.tile_bitwidths.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(inp.device)
        elif self.tile_bitwidths.ndim == 3:
            mask = self.tile_bitwidths.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(inp.device)

        # Apply mask to zero out unwanted precision outputs
        masked_output = output * mask
        # Sum along the bitwidth dimension (dim=2)
        summed_output = masked_output.sum(dim=2)

        final_output = summed_output.permute(0, 2, 3, 1, 4).reshape(batch_size, self.out_channels, output_height, output_tile_width*num_tiles)

        return final_output


    def forward(self, inp, teacher_input = None):
        if not self.float_mode:
            if not self.tile_mode:
                x_student, x_teacher = self.forward_vanilla_and_distillation(inp, teacher_input)
                if x_teacher is not None:
                    return x_student, x_teacher
                else:
                    return x_student
            elif self.tile_mode:
                x_student = self.forward_tiled_parallel(inp)
                return x_student
        
        elif self.float_mode:
            # print("Shape of input tensor: ", inp.shape) 
            x = nn.functional.conv2d(
                inp,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            # print("Shape of output tensor: ", x.shape)
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

            