import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

import logging

from models.FusionModules import MeanFusionBlock


class PatchTSMixerGatedAttention(nn.Module):
    """
    Module that applies gated attention to input data.

    Args:
        in_size (`int`): The input size.
        out_size (`int`): The output size.
    """

    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.attn_layer = nn.Linear(in_size, out_size)
        self.attn_softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        atten_inputs = self.attn_layer(inputs)
        attn_weight = self.attn_softmax(atten_inputs)
        inputs = inputs * attn_weight
        return inputs

def get_num_patches(args, loc, mod):
    sequence_length = args.dataset_config["num_segments"] * args.dataset_config["loc_mod_spectrum_len"][loc][mod]
    patch_length = args.dataset_config["TSMixer"]["patch_length"][mod]
    patch_stride = args.dataset_config["TSMixer"]["patch_stride"][mod]
    

    if sequence_length <= patch_length:
        raise ValueError(
            f"Sequence length ({sequence_length}) has to be greater than the patch length ({patch_length})"
        )

    # get the number of patches
    num_patches = (max(sequence_length, patch_length) - patch_length) // patch_stride + 1
    return num_patches

# Copied from transformers.models.patchtst.modeling_patchtst.PatchTSTBatchNorm with PatchTST->PatchTSMixer
class PatchTSMixerBatchNorm(nn.Module):
    """
    Compute batch normalization over the sequence length (time) dimension.
    """

    def __init__(self, args):
        super().__init__()
        self.config = args.dataset_config["TSMixer"]
        self.batchnorm = nn.BatchNorm1d(self.config["dim"], eps=self.config["norm_eps"])

    def forward(self, inputs: torch.Tensor):
        """
        Parameters:
            inputs (`torch.Tensor` of shape `(batch_size, sequence_length, d_model)`):
                input for Batch norm calculation
        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, d_model)`
        """
        output = inputs.transpose(1, 2)  # output: (batch_size, d_model, sequence_length)
        output = self.batchnorm(output)
        return output.transpose(1, 2)


class PatchTSMixerPositionalEncoding(nn.Module):
    """
    Class for positional encoding
    """

    def __init__(self, args, loc, mod):
        super().__init__()
        self.config = args.dataset_config["TSMixer"]
        # positional encoding: [num_patches x d_model]
        self.num_patches = get_num_patches(args, loc, mod)
        self.dim = self.config["dim"]
        if self.config["use_positional_encoding"] == True:
            self.position_enc = self._init_pe(args)
        else:
            self.position_enc = nn.Parameter(torch.zeros(self.num_patches, self.dim))

    def _init_pe(self, args) -> nn.Parameter:
        # Positional encoding
        if self.config["positional_encoding_type"]== "random":
            position_enc = nn.Parameter(torch.randn(self.num_pathces, self.dim), requires_grad=True)
        elif self.config["positional_encoding_type"] == "sincos":
            position_enc = torch.zeros(self.num_patches, self.dim)
            position = torch.arange(0, self.num_patches).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.dim, 2) * -(math.log(10000.0) / self.dim))
            position_enc[:, 0::2] = torch.sin(position * div_term)
            position_enc[:, 1::2] = torch.cos(position * div_term)
            position_enc = position_enc - position_enc.mean()
            position_enc = position_enc / (position_enc.std() * 10)
            position_enc = nn.Parameter(position_enc, requires_grad=False)
        else:
            raise ValueError(
                f"{self.config['use_positional_encoding']} is not a valid positional encoder. Available types are 'random' and 'sincos'."
            )
        return position_enc

    def forward(self, patch_input):
        # hidden_state: [bs x num_channels x num_patches x d_model]
        hidden_state = patch_input + self.position_enc
        return hidden_state


class PatchTSMixerNormLayer(nn.Module):
    """Normalization block

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, args):
        super().__init__()

        self.config = args.dataset_config["TSMixer"]
        self.norm_mlp = self.config["norm_mlp"]

        if "batch" in self.norm_mlp.lower():
            self.norm = PatchTSMixerBatchNorm(args)
        else:
            self.norm = nn.LayerNorm(self.config["dim"], self.config["norm_eps"])

    def forward(self, inputs):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the normalization layer.
        Returns:
            `torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`
        """
        if "batch" in self.norm_mlp.lower():
            # reshape the data
            inputs_reshaped = torch.reshape(
                inputs,
                (
                    inputs.shape[0] * inputs.shape[1],
                    inputs.shape[2],
                    inputs.shape[3],
                ),
            )  # inputs_reshaped: [batch_size*num_channels, num_patches, d_model]
            # inputs_reshaped: [batch_size*num_channels, num_patches, d_model]
            inputs_reshaped = self.norm(inputs_reshaped)

            # put back data to the original shape
            inputs = torch.reshape(inputs_reshaped, inputs.shape)

        else:
            inputs = self.norm(inputs)

        return inputs


class PatchTSMixerMLP(nn.Module):
    def __init__(self, in_features, out_features, args):
        super().__init__()
        self.config = args.dataset_config["TSMixer"]
        
        num_hidden = in_features * self.config["expansion_factor"]
        self.fc1 = nn.Linear(in_features, num_hidden)
        self.dropout1 = nn.Dropout(self.config["dropout"])
        self.fc2 = nn.Linear(num_hidden, out_features)
        self.dropout2 = nn.Dropout(self.config["dropout"])

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the MLP layer.
        Returns:
            `torch.Tensor` of the same shape as `inputs`
        """
        inputs = self.dropout1(nn.functional.gelu(self.fc1(inputs)))
        inputs = self.fc2(inputs)
        inputs = self.dropout2(inputs)
        return inputs


class PatchTSMixerChannelFeatureMixerBlock(nn.Module):
    """This module mixes the features in the channel dimension.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """
    def __init__(self, args, loc, mod):
        super().__init__()
        
        self.args = args
        self.config = args.dataset_config["TSMixer"]

        self.norm = PatchTSMixerNormLayer(args)
        self.gated_attn = self.config["gated_attn"]
        
        self.num_input_channels = self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod] #TODO: Change to num_input_channels

        self.mlp = PatchTSMixerMLP(self.num_input_channels, self.num_input_channels, args)

        if self.gated_attn:
            self.gating_block = PatchTSMixerGatedAttention(
                in_size=self.num_input_channels, out_size=self.num_input_channels
            )

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                input to the MLP layer
        Returns:
            `torch.Tensor` of the same shape as `inputs`
        """
        
        residual = inputs
        inputs = self.norm(inputs)
        
        inputs = inputs.permute(0, 3, 2, 1)

        if self.gated_attn:
            inputs = self.gating_block(inputs)

        inputs = self.mlp(inputs)

        inputs = inputs.permute(0, 3, 2, 1)

        out = inputs + residual
        return out


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->PatchTSMixer
class PatchTSMixerAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        args,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.args = args

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class PatchMixerBlock(nn.Module):
    """This module mixes the patch dimension.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, args, loc, mod):
        super().__init__()

        self.args = args
        self.config = args.dataset_config["TSMixer"]

        self.norm = PatchTSMixerNormLayer(args)

        self.self_attn = self.config["self_attn"]
        self.gated_attn = self.config["gated_attn"]
        
        self.num_patches = get_num_patches(args, loc, mod)
        self.dim = self.config["dim"]

        self.mlp = PatchTSMixerMLP(
            in_features=self.num_patches,
            out_features=self.num_patches,
            args=args,
        )

        if self.config["gated_attn"]:
            self.gating_block = PatchTSMixerGatedAttention(in_size=self.num_patches, out_size=self.num_patches)

        if self.self_attn:
            self.self_attn_layer = PatchTSMixerAttention(
                args,
                embed_dim=self.dim,
                num_heads=self.config["self_attn_heads"],
                dropout=self.config["dropout"],
            )
            self.norm_attn = PatchTSMixerNormLayer(args)

    def forward(self, hidden_state):
        residual = hidden_state

        hidden_state = self.norm(hidden_state)

        if self.self_attn:
            batch_size, n_vars, num_patches, d_model = hidden_state.shape
            hidden_state_reshaped = hidden_state.reshape(batch_size * n_vars, num_patches, d_model)

            x_attn, _, _ = self.self_attn_layer(hidden_state_reshaped, output_attentions=False)
            x_attn = x_attn.reshape(batch_size, n_vars, num_patches, d_model)

        # Transpose so that num_patches is the last dimension
        hidden_state = hidden_state.transpose(2, 3)
        hidden_state = self.mlp(hidden_state)

        if self.gated_attn:
            hidden_state = self.gating_block(hidden_state)

        # Transpose back
        hidden_state = hidden_state.transpose(2, 3)

        if self.self_attn:
            hidden_state = self.norm_attn(hidden_state + x_attn)

        out = hidden_state + residual
        return out


class FeatureMixerBlock(nn.Module):
    """This module mixes the hidden feature dimension.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    """

    def __init__(self, args, loc, mod):
        super().__init__()
        self.args = args
        self.config = args.dataset_config["TSMixer"]
        self.norm = PatchTSMixerNormLayer(args)

        self.gated_attn = self.config["gated_attn"]
        self.dim = self.config["dim"]

        self.mlp = PatchTSMixerMLP(
            in_features=self.dim,
            out_features=self.dim,
            args=args,
        )

        if self.config["gated_attn"]:
            self.gating_block = PatchTSMixerGatedAttention(in_size=self.dim, out_size=self.dim)

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, num_patches, d_model)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        residual = hidden
        hidden = self.norm(hidden)
        hidden = self.mlp(hidden)

        if self.gated_attn:
            hidden = self.gating_block(hidden)

        out = hidden + residual
        return out


class PatchTSMixerLayer(nn.Module):
    """
    The `PatchTSMixer` layer that does all three kinds of mixing.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    """

    def __init__(self, args, loc, mod):
        super().__init__()

        self.config = args.dataset_config["TSMixer"]
        self.patch_mixer = PatchMixerBlock(args, loc, mod)
        self.feature_mixer = FeatureMixerBlock(args, loc, mod)

        self.mode = self.config["mode"]

        if self.mode == "mix_channel":
            self.channel_feature_mixer = PatchTSMixerChannelFeatureMixerBlock(args, loc, mod)

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, num_patches, d_model)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        if self.mode == "mix_channel":
            hidden = self.channel_feature_mixer(hidden)
        
        hidden = self.patch_mixer(hidden)
        hidden = self.feature_mixer(hidden)  # hidden: (batch_size x num_patches x d_model)
        return hidden


class PatchTSMixerBlock(nn.Module):
    """The main computing framework of the `PatchTSMixer` model.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, args, loc, mod):
        super().__init__()

        self.config = args.dataset_config["TSMixer"]
        num_layers = self.config["num_layers"]

        logging.info(f"=\t[PatchTSMixerBlock]: Initializing {loc} {mod} PatchTSMixerBlock with {num_layers} layers.")
        self.mixers = nn.ModuleList([PatchTSMixerLayer(args, loc, mod) for _ in range(num_layers)])
        logging.info(f"=\t[PatchTSMixerBlock]: Initialized {loc} {mod} PatchTSMixerBlock")

    def forward(self, hidden_state, output_hidden_states: bool = False):
        """
        Args:
            hidden_state (`torch.Tensor`): The input tensor.
            output_hidden_states (`bool`, *optional*, defaults to False.):
                Whether to output the hidden states as well.

        Returns:
            `torch.Tensor`: The embedding. `list`: List of all hidden states if `output_hidden_states` is set to
            `True`.
        """
        all_hidden_states = []

        embedding = hidden_state

        for mod in self.mixers:
            embedding = mod(embedding)
            if output_hidden_states:
                all_hidden_states.append(embedding)

        if output_hidden_states:
            return embedding, all_hidden_states
        else:
            return embedding, None


class PatchTSMixerLinearHead(nn.Module):
    """Linear head for Classification and Regression.

    Args:
        config (`PatchTSMixerConfig`, *required*):

    """

    def __init__(self, args, loc, mod, distribution_output=None):
        super().__init__()

        self.args = args
        self.config = args.dataset_config["TSMixer"]
        
        self.head_aggregation = self.config["head_aggregation"]
        self.output_range = self.config["output_range"]
        self.num_patches = get_num_patches(args, loc, mod)
        self.dim = self.config["dim"]
        self.head_dropout = self.config["head_dropout"]
        
        # self.num_input_channels = sum([self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod]] for loc in args.dataset_config["location_names"] for mod in args.dataset_config["modality_names"])
        self.num_input_channels = args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
        
        self.num_targets = self.args.dataset_config[args.task]["num_classes"]

        if self.head_aggregation is None:
            mul_factor = self.num_patches
        else:
            mul_factor = 1
        self.distribution_output = distribution_output
        if distribution_output is None:
            self.projection = nn.Linear(
                self.dim * self.num_input_channels * mul_factor,
                self.num_targets,
            )
        else:
            self.projection = distribution_output.get_parameter_projection(
                self.dim * self.num_input_channels * mul_factor
            )

        if self.head_aggregation is None:
            self.flatten = nn.Flatten(start_dim=-3)
        else:
            self.flatten = nn.Flatten(start_dim=-2)

        self.dropout = nn.Dropout(self.head_dropout)

    def forward(self, hidden_features):
        """
        Args:
            hidden_features (`torch.Tensor` of shape `(batch_size x num_patch x d_model)` in `flatten` mode
                or `(batch_size x n_vars x num_patch x d_model)` in `common_channel`/`mix_channel` mode.): Input hidden
                features.

        Returns:
            `torch.Tensor` of shape `(batch_size x num_targets)`.
        """

        # batch_size x d_model x num_patch or batch_size x n_vars x d_model x num_patch
        hidden_features = hidden_features.transpose(-1, -2)
        if self.head_aggregation == "use_last":
            # batch_size x d_model (flatten) or # batch_size x n_vars x d_model (common_channel)
            hidden_features = hidden_features[..., -1]
        elif self.head_aggregation == "max_pool":
            # batch_size x n_vars x d_model or batch_size x d_model
            hidden_features = hidden_features.max(dim=-1).values
        elif self.head_aggregation == "avg_pool":
            # batch_size x n_vars x d_model or batch_size x d_model
            hidden_features = hidden_features.mean(dim=-1)

        if self.flatten:
            hidden_features = self.flatten(hidden_features)
        hidden_features = self.dropout(hidden_features)
        hidden_features = self.projection(hidden_features)  # batch_size x num_targets

        if (self.distribution_output is None) and (self.output_range is not None):
            hidden_features = (
                torch.sigmoid(hidden_features) * (self.output_range[1] - self.output_range[0]) + self.output_range[0]
            )
        return hidden_features


class PatchTSMixerPretrainHead(nn.Module):
    def __init__(self, args, loc, mod):
        super().__init__()
        
        self.config = args.dataset_config["TSMixer"]

        self.dropout_layer = nn.Dropout(self.config["head_dropout"])
        self.base_pt_block = nn.Linear(self.config["dim"], self.config["patch_length"][mod])

    def forward(self, hidden_features):
        """
        Args:
            hidden_features (`torch.Tensor` of shape `(batch_size x num_patch x d_model)` in `flatten` mode
                or `(batch_size x n_vars x num_patch x d_model)` in `common_channel`/`mix_channel` mode.): Input hidden
                features.

        Returns:
            `torch.Tensor` of shape `(batch_size x n_vars x num_patch x patch_length)`.
        """

        hidden_features = self.dropout_layer(hidden_features)
        forecast = self.base_pt_block(hidden_features)  # [batch_size x n_vars x num_patch x patch_length]
        return forecast


# Copied from transformers.models.patchtst.modeling_patchtst.random_masking
def random_masking(
    inputs: torch.Tensor,
    mask_ratio: float,
    unmasked_channel_indices: list = None,
    channel_consistent_masking: bool = False,
    mask_value: int = 0,
):
    """random_masking: Mask the input considering the control variables.

    Args:
        inputs (`torch.Tensor` of shape `(batch_size, num_channels, sequence_length, num_features)`):
            The input tensor to mask.
        mask_ratio (`float`):
            Masking ratio applied to mask the input data during random pretraining. It is the number between 0 and 1.
        unmasked_channel_indices (list, *optional*):
            Indices of channels that will not be masked.
        channel_consistent_masking (bool, *optional*, defaults to `False`):
            When true, masking will be same across all channels of a timeseries. Otherwise, masking positions will vary
            across channels.
        mask_value (int, *optional*, defaults to 0):
            Define the value of masked patches for pretraining.

    Returns:
        `tuple(torch.Tensor)`: inputs_mask, masked input, same shape as input Tensor and mask tensor of shape [bs x c x
        n]
    """
    if mask_ratio < 0 or mask_ratio >= 1:
        raise ValueError(f"Mask ratio {mask_ratio} has to be between 0 and 1.")

    batch_size, num_channels, sequence_length, num_features = inputs.shape
    device = inputs.device

    len_keep = int(sequence_length * (1 - mask_ratio))

    if channel_consistent_masking:
        noise = torch.rand(batch_size, 1, sequence_length, device=device)  # noise in [0, 1], bs x 1 x  L
        noise = noise.repeat(1, num_channels, 1)  # bs x num_channels x time
    else:
        # noise in [0, 1], bs x num_channels x L
        noise = torch.rand(batch_size, num_channels, sequence_length, device=device)

    # mask: [bs x num_channels x num_patch]
    mask = torch.ones(batch_size, num_channels, sequence_length, device=device)
    mask[:, :, :len_keep] = 0

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=-1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=-1)  # ids_restore: [bs x num_channels x L]

    mask = torch.gather(mask, dim=-1, index=ids_restore)
    mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_features)  # mask: [bs x num_channels x num_patches x patch_length]
    if unmasked_channel_indices is not None:
        mask[:, unmasked_channel_indices, :, :] = 0

    inputs_mask = inputs.masked_fill(mask.bool(), mask_value)
    return inputs_mask, mask[..., 0]


# Copied from transformers.models.patchtst.modeling_patchtst.PatchTSTPatchify with PatchTST->PatchTSMixer
class PatchTSMixerPatchify(nn.Module):
    """
    A class to patchify the time series sequence into different patches

    Returns:
        `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
    """

    def __init__(self, args, loc, mod):
        super().__init__()

        self.config = args.dataset_config["TSMixer"]
        self.sequence_length = args.dataset_config["num_segments"] * args.dataset_config["loc_mod_spectrum_len"][loc][mod]
        self.patch_length = self.config["patch_length"][mod]
        self.patch_stride = self.config["patch_stride"][mod]

        if self.sequence_length <= self.patch_length:
            raise ValueError(
                f"Sequence length ({self.sequence_length}) has to be greater than the patch length ({self.patch_length})"
            )

        # get the number of patches
        self.num_patches = get_num_patches(args, loc, mod)

        new_sequence_length = self.patch_length + self.patch_stride * (self.num_patches - 1)

        self.sequence_start = self.sequence_length - new_sequence_length
        
        logging.info(f"=\t[PatchTSMizerPatchify]: Initialized {loc} {mod} Patchify Module with {self.num_patches} patches.")

    def forward(self, x_input):
        """
        Parameters:
            x_input (`torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`, *required*):
                Input for patchification

        Returns:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
        """
        
        sequence_length = x_input.shape[-2]
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Input sequence length ({sequence_length}) doesn't match model configuration ({self.sequence_length})."
            )
        # output: [bs x new_sequence_length x num_channels]
        output = x_input[:, self.sequence_start :, :]
        # output: [bs x num_patches x num_input_channels x patch_length]
        output = output.unfold(dimension=-2, size=self.patch_length, step=self.patch_stride)
        # output: [bs x num_input_channels x num_patches x patch_length]
        output = output.transpose(-2, -3).contiguous()
        return output


# Copied from transformers.models.patchtst.modeling_patchtst.PatchTSTMasking with PatchTST->PatchTSMixer
class PatchTSMixerMasking(nn.Module):
    """
    Class to perform random or forecast masking.

    Parameters:
        config (`PatchTSMixerConfig`): model config
    Returns:
        x_mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`)
            Masked patched input
        mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches)`)
            Bool tensor indicating True on masked points
    """

    def __init__(self, args):
        super().__init__()
        self.config = args.dataset_config["TSMixer"]
        self.random_mask_ratio = self.config["random_mask_ratio"]
        self.channel_consistent_masking = self.config["channel_consistent_masking"]
        self.mask_type = self.config["mask_type"]
        self.num_forecast_mask_patches = self.config["num_forecast_mask_patches"]
        self.unmasked_channel_indices = self.config["unmasked_channel_indices"]
        self.mask_value = self.config["mask_value"]

        if self.unmasked_channel_indices is not None:
            self.unmasked_channel_indices = sorted(self.unmasked_channel_indices)

    def forward(self, patch_input):
        """
        Parameters:
            patch_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Patch input

        Return:
            masked_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`)
                Masked patched input
            mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches)`)
                Bool tensor indicating True on masked points

        """
        if self.mask_type == "random":
            masked_input, mask = random_masking(
                inputs=patch_input,
                mask_ratio=self.random_mask_ratio,
                unmasked_channel_indices=self.unmasked_channel_indices,
                channel_consistent_masking=self.channel_consistent_masking,
                mask_value=self.mask_value,
            )
        else:
            raise ValueError(f"Invalid mask type {self.mask_type}.")

        # mask: [bs x num_input_channels x num_patch]
        mask = mask.bool()
        return masked_input, mask


# Copied from transformers.models.patchtst.modeling_patchtst.PatchTSTStdScaler with PatchTST->PatchTSMixer
class PatchTSMixerStdScaler(nn.Module):
    """
    Standardize features by calculating the mean and scaling along the first dimension, and then normalizes it by
    subtracting from the mean and dividing by the standard deviation.
    """

    def __init__(self, args):
        super().__init__()
        # self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        # self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-5
        self.config = args.dataset_config["TSMixer"]
        self.scaling_dim = self.config["scaling_dim"]
        self.keepdim = self.config["keepdim"]
        self.minimum_scale = self.config["minimum_scale"]

    def forward(self, data, observed_indicator):
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        denominator = observed_indicator.sum(self.scaling_dim, keepdim=self.keepdim)
        denominator = denominator.clamp_min(1.0)
        loc = (data * observed_indicator).sum(self.scaling_dim, keepdim=self.keepdim) / denominator

        variance = (((data - loc) * observed_indicator) ** 2).sum(self.scaling_dim, keepdim=self.keepdim) / denominator
        scale = torch.sqrt(variance + self.minimum_scale)
        return (data - loc) / scale, loc, scale


# Copied from transformers.models.patchtst.modeling_patchtst.PatchTSTMeanScaler with PatchTST->PatchTSMixer
class PatchTSMixerMeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along the first dimension, and scales the data
    accordingly.
    """

    def __init__(self, args, loc, mod):
        super().__init__()
        self.config = args.dataset_config["TSMixer"]
        
        # self.scaling_dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        # self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-10
        # self.default_scale = config.default_scale if hasattr(config, "default_scale") else None
        
        self.scaling_dim = self.config["scaling_dim"]
        self.keepdim = self.config["keepdim"]
        self.minimum_scale = self.config["minimum_scale"]
        self.default_scale = None
        

    def forward(self, data, observed_indicator):
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        ts_sum = (data * observed_indicator).abs().sum(self.scaling_dim, keepdim=True)
        num_observed = observed_indicator.sum(self.scaling_dim, keepdim=True)

        scale = ts_sum / torch.clamp(num_observed, min=1)

        # If `default_scale` is provided, we use it, otherwise we use the scale
        # of the batch.
        if self.default_scale is None:
            batch_sum = ts_sum.sum(dim=0)
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            default_scale = self.default_scale * torch.ones_like(scale)

        # apply default scale where there are no observations
        scale = torch.where(num_observed > 0, scale, default_scale)

        # ensure the scale is at least `self.minimum_scale`
        scale = torch.clamp(scale, min=self.minimum_scale)
        scaled_data = data / scale

        if not self.keepdim:
            scale = scale.squeeze(dim=self.scaling_dim)

        return scaled_data, torch.zeros_like(scale), scale


# Copied from transformers.models.patchtst.modeling_patchtst.PatchTSTNOPScaler with PatchTST->PatchTSMixer
class PatchTSMixerNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along the first dimension, and therefore applies no scaling to the input data.
    """

    def __init__(self, args, loc=None, mod=None):
        super().__init__()
        
        # self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        self.args = args
        self.config = args.dataset_config["TSMixer"]
        self.scaling_dim = self.config["scaling_dim"]
        self.keepdim = self.config["keepdim"]

    def forward(self, data, observed_indicator):
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        scale = torch.ones_like(data, requires_grad=False).mean(dim=self.scaling_dim, keepdim=self.keepdim)
        loc = torch.zeros_like(data, requires_grad=False).mean(dim=self.scaling_dim, keepdim=self.keepdim)
        return data, loc, scale



class PatchTSMixerEncoder(nn.Module):
    """
    Encoder for PatchTSMixer which inputs patched time-series and outputs patched embeddings.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, args, loc, mod):
        super().__init__()
        
        self.config = args.dataset_config["TSMixer"]
        self.patcher = nn.Linear(self.config["patch_length"][mod], self.config["dim"])
        
        if self.config["use_positional_encoding"]:
            self.positional_encoder = PatchTSMixerPositionalEncoding(args, loc, mod)
        else:
            self.positional_encoder = None
        
        self.mlp_mixer_encoder = PatchTSMixerBlock(args, loc, mod)
        
        logging.info(f"=\t[PatchTSMixerEncoder]: Initialized {loc} {mod} PatchTSMixerEncoder")
    
    def forward(self, x_input, output_hidden_states=False):
        patches = self.patcher(x_input)

        # add positional encoder
        if self.positional_encoder is not None:
            patches = self.positional_encoder(patches)

        last_hidden_state, hidden_states = self.mlp_mixer_encoder(patches, output_hidden_states=output_hidden_states)

        return (last_hidden_state, hidden_states)

class PatchTSMixerModel(nn.Module):
    def __init__(self, args, loc, mod):
        super().__init__()

        self.args = args
        self.config = args.dataset_config["TSMixer"]
        self.patching = PatchTSMixerPatchify(args, loc, mod)
        self.encoder = PatchTSMixerEncoder(args, loc, mod)

        if self.args.learn_framework in {"MAE"}:
            self.masking = PatchTSMixerMasking(args, loc, mod)

        logging.info(f"=\t[PatchTSMixerModel] Initializing {self.config['scaling']} Scaler for {loc} {mod} PatchTSMixerModel")
        if self.config["scaling"] == "mean":
            self.scaler = PatchTSMixerMeanScaler(args, loc, mod)
        elif self.config["scaling"] == "std":
            self.scaler = PatchTSMixerStdScaler(args, loc, mod)
        else:
            self.scaler = PatchTSMixerNOPScaler(args, loc, mod)
            
        logging.info(f"=\t[PatchTSMixerModel] Initialized PatchTSMixerModel")

    def forward(self, x_input, observed_mask=None):
        if observed_mask is None:
            observed_mask = torch.ones_like(x_input)
        scaled_past_values, loc, scale = self.scaler(x_input, observed_mask)

        patched_x = self.patching(scaled_past_values)  # [batch_size x num_input_channels x num_patch x patch_length

        enc_input = patched_x
        if self.args.learn_framework in {"MAE"}:
            enc_input, mask = self.masking(patched_x)
            # enc_input: [batch_size x num_input_channels x num_patch x patch_length]
            # mask: [batch_size x num_input_channels x num_patch]

        last_hidden_state, hidden_states = self.encoder(enc_input)
        
        return last_hidden_state, hidden_states, loc, scale

class TSMixer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.config = args.dataset_config["TSMixer"]

        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        
        
        self.loc_mod_extractor = nn.ModuleDict()
        self.inject_scale = nn.ModuleDict()
        
        self.dropout = nn.Dropout(self.config["head_dropout"])
        
        self.mod_fusion_layers = MeanFusionBlock()
        
        for loc in self.locations:
            self.loc_mod_extractor[loc] = nn.ModuleDict()
            self.inject_scale[loc] = nn.ModuleDict()
            for mod in self.modalities:
                self.loc_mod_extractor[loc][mod] = PatchTSMixerModel(args, loc, mod)
                if self.config["scaling"] in ["std", "mean", True]:
                    self.inject_scale[loc][mod] = InjectScalerStatistics4D(args, loc, mod)
                else:
                    self.inject_scale[loc][mod] = None
                    
        
        fc_dim = 0
        for loc in self.locations:
            for mod in self.modalities:
                fc_dim = self.config["dim"] * self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
        
        self.mod_layer_norm = nn.LayerNorm(fc_dim)
        self.sample_embd_layer = nn.Sequential(
            nn.Linear(fc_dim, self.config["fc_dim"]),
            nn.GELU(),
        )
        
        if self.args.learn_framework in {"CMCV2"}:
            out_dim = self.args.dataset_config["CMCV2"]["emb_dim"]
            self.mod_projectors = nn.ModuleDict()
            for mod in self.modalities:
                mod_dim = self.config["dim"] * self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
                self.mod_projectors[mod] = nn.Sequential(
                    nn.Linear(mod_dim, out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim),
                )
        
        # if self.args.train_mode == "supervised":
        self.class_layer = nn.Linear(self.config["fc_dim"], self.args.dataset_config[self.args.task]["num_classes"])

    def forward(self, loc_mod_input, class_head=True, proj_head=False):
        
        
        # mod_embeddings = []
        # loc_mod_features = {}
        mod_loc_features = {mod: [] for mod in self.modalities}
        for loc in self.locations:
            # loc_mod_features[loc] = []
            for mod in self.modalities:
                
                # Preprocess input to match the TSMixer input
                loc_mod_input[loc][mod] = loc_mod_input[loc][mod] # [b, c, i, s]
                loc_mod_input[loc][mod] = loc_mod_input[loc][mod].flatten(start_dim=2) # [b, c, i*s]
                loc_mod_input[loc][mod] = torch.permute(loc_mod_input[loc][mod], (0, 2, 1)) # [b, i*s, c]                
                model_output = self.loc_mod_extractor[loc][mod](
                    loc_mod_input[loc][mod]
                )
                
                last_hidden_state, hidden_states, scale_loc, scale_scale = model_output
                
                if self.config["scaling"] in ["std", "mean", True]:
                    last_hidden_state = self.inject_scale[loc][mod](
                        last_hidden_state,
                        loc=scale_loc,
                        scale=scale_scale,
                    )

                # [b, c, n, dim] -> [b, c, dim, n]
                last_hidden_state = last_hidden_state.transpose(-1, -2)
                # [b, c, dim, n] -> [b, c, dim]
                last_hidden_state = last_hidden_state.mean(-1)
                
                last_hidden_state = self.dropout(last_hidden_state)

                # mod_embeddings.append(last_hidden_state.flatten(start_dim=1))
                # loc_mod_features[loc].append(last_hidden_state.flatten(start_dim=1))
                mod_loc_features[mod].append(last_hidden_state.flatten(start_dim=1))

            # loc_mod_features[loc] = torch.stack(loc_mod_features[loc], dim=-1)
        

        for mod in self.modalities:
            mod_loc_features[mod] = torch.stack(mod_loc_features[mod], dim=-1).flatten(start_dim=1)
        mod_features = []
        for mod in self.modalities:
            mod_features.append(mod_loc_features[mod])
        
        if not class_head:
            """Pretrainin"""
            if self.args.learn_framework not in {"CMCV2"}:
                raise ValueError(f"Invalid learn_framework {self.args.learn_framework}")
            if proj_head:
                sample_features = {}
                for i, mod in enumerate(self.modalities):
                    sample_features[mod] = self.mod_projectors[mod](mod_features[i])
                return sample_features
            return dict(zip(self.modalities, mod_features))
        
        

        
        if self.args.train_mode == "supervised":
            """Supervised Learning"""
            # mod_features = torch.cat(mod_features, dim=-1) # [b, c] -> [b, c, sensors]
            mod_features = torch.stack(mod_features, dim=-1) # [b, c] -> [b, c, sensors]
            mod_features = mod_features.unsqueeze(-2) # [b, c, 1, sensors]
            fused_features = self.mod_fusion_layers(mod_features).flatten(start_dim=1) # [b, c, 1, sensors] -> [b, c]
            loc_embeddings = self.mod_layer_norm(fused_features)
            mod_features = self.sample_embd_layer(loc_embeddings)
        elif self.args.train_mode in {"contrastive", "generative"}:
            mod_features = torch.cat(mod_features, dim=-1)
        
        # mod_features = torch.cat(mod_features, dim=-1)
        logits = self.class_layer(mod_features)
        return logits



class InjectScalerStatistics4D(nn.Module):
    def __init__(self, args, loc, mod, expansion=2):
        super().__init__()
        
        self.config = args.dataset_config["TSMixer"]
        
        self.num_patches = get_num_patches(args, loc, mod)
        self.dim = self.config["dim"]

        self.inverse_trans_expansion = nn.Linear(self.dim + 2, expansion * self.dim)
        self.inverse_trans_compression = nn.Linear(expansion * self.dim, self.dim)
        self.map_scale_expansion = nn.Linear(2, 2 * expansion)
        self.map_scale_compression = nn.Linear(2 * expansion, 2)

    def forward(self, inputs: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `(batch_size, num_input_channels, num_patch, self.dim)`)
            loc (`torch.Tensor` of shape `(batch_size, 1, num_input_channels)`)
            scale (`torch.Tensor` of shape `(batch_size, 1, num_input_channels)`)
        Returns:
            `torch.Tensor` of shape `(batch_size, num_input_channels, num_patch, d_model)`
        """

        mean = loc.transpose(-1, -2)  # [batch_size x n_channels x 1 ]
        mean = mean.unsqueeze(-2)  # [batch_size x n_channels x 1 x 1]
        mean = mean.repeat(1, 1, self.num_patches, 1)  # [batch_size x n_channels x num_patch x 1]

        stdev = scale.transpose(-1, -2)  # [batch_size x n_channels x 1 ]
        stdev = stdev.unsqueeze(-2)  # [batch_size x n_channels x 1 x 1]
        stdev = stdev.repeat(1, 1, self.num_patches, 1)  # [batch_size x n_channels x num_patch x 1]

        concat_stats = torch.cat([mean, stdev], dim=-1)  # [batch_size x n_channels x num_patch x 2]

        concat_stats = self.map_scale_expansion(concat_stats)  # [batch_size x n_channels x num_patch x (2*expansion)]
        concat_stats = self.map_scale_compression(concat_stats)  # [batch_size x n_channels x num_patch x 2]

        inputs = torch.cat([inputs, concat_stats], dim=-1)  # [batch_size x channels x num_patch x d_model+2]
        inputs = self.inverse_trans_expansion(inputs)  # [batch_size x channels x num_patch x (expansion*d_model)]
        inputs = self.inverse_trans_compression(inputs)  # [batch_size x channels x num_patch x d_model]

        return inputs