import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.nn import TransformerEncoderLayer
from input_utils.padding_utils import get_padded_size
from models.FusionModules import TransformerFusionBlock

from timm.models.layers import trunc_normal_

import logging

from models.SwinModules import (
    BasicLayer,
    PatchEmbed,
    PatchExpanding,
    PatchMerging,
)

from models.MAEModule import window_masking


class TransformerV4_CMC(nn.Module):
    """
    SWIN Transformer model

    Parameters
    ----------

    Returns
    -------
    int or float
        The square of `x`
    """

    def __init__(self, args) -> None:
        """
        SWIN Transformer model constructor

        Parameters
        ----------
        args:
            list of configuration
        """

        super().__init__()
        self.args = args
        self.config = args.dataset_config["TransformerV4"]
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.num_segments = args.dataset_config["num_segments"]

        # Transformer Variables
        self.drop_rate = self.config["dropout_ratio"]
        self.norm_layer = nn.LayerNorm
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Single sensor
        self.freq_interval_layers = nn.ModuleDict()  # Frequency & Interval layers
        self.patch_embed = nn.ModuleDict()
        self.absolute_pos_embed = nn.ModuleDict()
        self.mod_patch_embed = nn.ModuleDict()
        self.mod_in_layers = nn.ModuleDict()

        for loc in self.locations:
            self.freq_interval_layers[loc] = nn.ModuleDict()
            self.patch_embed[loc] = nn.ModuleDict()
            self.absolute_pos_embed[loc] = nn.ParameterDict()
            self.mod_in_layers[loc] = nn.ModuleDict()

            for mod in self.modalities:
                # Decide the spatial size for "image"
                stride = self.config["in_stride"][mod]
                spectrum_len = args.dataset_config["loc_mod_spectrum_len"][loc][mod]
                img_size = (self.num_segments, spectrum_len // stride)

                # get the padded image size
                padded_img_size = get_padded_size(
                    img_size,
                    self.config["window_size"][mod],
                    self.config["patch_size"]["freq"][mod],
                    len(self.config["time_freq_block_num"][mod]),
                )
                logging.info(f"=\tPadded image size for {mod}: {padded_img_size}")

                # Patch embedding and Linear embedding (H, W, in_channel) -> (H / p_size, W / p_size, C)
                self.patch_embed[loc][mod] = PatchEmbed(
                    img_size=padded_img_size,
                    patch_size=self.config["patch_size"]["freq"][mod],
                    in_chans=args.dataset_config["loc_mod_in_freq_channels"][loc][mod] * stride,
                    embed_dim=self.config["time_freq_out_channels"],
                    norm_layer=self.norm_layer,
                )
                patches_resolution = self.patch_embed[loc][mod].patches_resolution
                logging.info(f"=\tPatch resolution for {mod}: {patches_resolution}")

                # Absolute positional embedding (optional)
                self.absolute_pos_embed[loc][mod] = nn.Parameter(
                    torch.zeros(1, self.patch_embed[loc][mod].num_patches, self.config["time_freq_out_channels"])
                )
                trunc_normal_(self.absolute_pos_embed[loc][mod], std=0.02)

                # Swin Transformer Block
                self.freq_interval_layers[loc][mod] = nn.ModuleList()

                # Drop path rate
                dpr = [
                    x.item()
                    for x in torch.linspace(
                        0, self.config["drop_path_rate"], sum(self.config["time_freq_block_num"][mod])
                    )
                ]  # stochastic depth decay rule

                for i_layer, block_num in enumerate(
                    self.config["time_freq_block_num"][mod]
                ):  # different downsample ratios
                    down_ratio = 2**i_layer
                    layer_dim = int(self.config["time_freq_out_channels"] * down_ratio)
                    layer = BasicLayer(
                        dim=layer_dim,  # C in SWIN
                        input_resolution=(
                            patches_resolution[0] // down_ratio,  # Patch resolution = (H/4, W/4)
                            patches_resolution[1] // down_ratio,
                        ),
                        num_heads=self.config["time_freq_head_num"],
                        window_size=self.config["window_size"][mod].copy(),
                        depth=block_num,
                        drop=self.drop_rate,
                        attn_drop=self.config["attn_drop_rate"],
                        drop_path=dpr[
                            sum(self.config["time_freq_block_num"][mod][:i_layer]) : sum(
                                self.config["time_freq_block_num"][mod][: i_layer + 1]
                            )
                        ],
                        norm_layer=self.norm_layer,
                        downsample=PatchMerging
                        if (i_layer < len(self.config["time_freq_block_num"][mod]) - 1)
                        else None,
                    )
                    self.freq_interval_layers[loc][mod].append(layer)

                # Unify the input channels for each modality
                self.mod_in_layers[loc][mod] = nn.Linear(
                    (patches_resolution[0] // down_ratio) * (patches_resolution[1] // down_ratio) * layer_dim,
                    self.config["loc_out_channels"],
                )

        # Loc fusion, [b, i, c], loc contextual feature extraction + loc fusion
        self.loc_context_layers = nn.ModuleDict()
        self.loc_fusion_layer = nn.ModuleDict()
        for mod in self.modalities:
            """Single mod contextual feature extraction"""
            module_list = [
                TransformerEncoderLayer(
                    d_model=self.config["loc_out_channels"],
                    nhead=self.config["loc_head_num"],
                    dim_feedforward=self.config["loc_out_channels"],
                    dropout=self.config["dropout_ratio"],
                    batch_first=True,
                )
                for _ in range(self.config["loc_block_num"])
            ]
            self.loc_context_layers[mod] = nn.Sequential(*module_list)

            """Mod fusion layer for each loc"""
            self.loc_fusion_layer[mod] = TransformerFusionBlock(
                self.config["loc_out_channels"],
                self.config["loc_head_num"],
                self.config["dropout_ratio"],
                self.config["dropout_ratio"],
            )

        # mod fusion layer
        if args.contrastive_framework == "Cosmo":
            "Attention fusion for Cosmo"
            "Attentio fusion for MAE?"
            self.mod_fusion_layer = TransformerFusionBlock(
                self.config["loc_out_channels"],
                self.config["loc_head_num"],
                self.config["dropout_ratio"],
                self.config["dropout_ratio"],
            )
            sample_dim = self.config["loc_out_channels"]
        else:
            sample_dim = self.config["loc_out_channels"] * len(self.modalities)

        if args.train_mode == "MAE":
            "Linear layers for encoded features"

            # [mod1_feature, mod2_feature, mod3_feature, ...] -> fc_dim layer
            self.encoded_features_layer = nn.Sequential(
                nn.Linear(sample_dim, self.config["fc_dim"]),
                nn.GELU(),
                nn.Linear(self.config["fc_dim"], self.config["fc_dim"]),
            )

            # [fc_dim layer] -> [mod1_feature, mod2_feature, mod3_feature]
            self.encoded_mod_extract_layer = nn.ModuleDict()
            for loc in self.locations:
                self.encoded_mod_extract_layer[loc] = nn.ModuleDict()
                for mod in self.modalities:
                    self.encoded_mod_extract_layer[loc][mod] = nn.Linear(
                        self.config["fc_dim"], self.config["loc_out_channels"]
                    )

            # init decoder layers
            self.init_decoder()

        # Classification layer
        if args.train_mode == "supervised" or self.config["pretrained_head"] == "linear":
            """Linear classification layers for supervised learning or finetuning."""
            self.class_layer = nn.Sequential(
                nn.Linear(sample_dim, args.dataset_config[args.task]["num_classes"]),
                nn.Sigmoid() if args.multi_class else nn.Softmax(dim=1),
            )
        else:
            """Non-linear classification layers for self-supervised learning."""
            self.class_layer = nn.Sequential(
                nn.Linear(sample_dim, self.config["fc_dim"]),
                nn.GELU(),
                nn.Linear(self.config["fc_dim"], args.dataset_config[args.task]["num_classes"]),
                nn.Sigmoid() if args.multi_class else nn.Softmax(dim=1),
            )

    def init_decoder(self) -> None:
        self.patch_expand = nn.ModuleDict()
        self.mask_token = nn.ModuleDict()
        self.decoder_blocks = nn.ModuleDict()
        self.decoder_pred = nn.ModuleDict()
        self.decoder_norm = nn.LayerNorm(self.config["decoder_channels"])
        self.masked_ratio = self.config["masked_ratio"]

        for loc in self.locations:
            self.patch_expand[loc] = nn.ModuleDict()
            self.mask_token[loc] = nn.ParameterDict()
            self.decoder_blocks[loc] = nn.ModuleDict()
            self.decoder_pred[loc] = nn.ModuleDict()

            for mod in self.modalities:
                self.mask_token[loc][mod] = nn.Parameter(torch.zeros(1, 1, self.config["time_freq_out_channels"]))

                self.patch_expand[loc][mod] = PatchExpanding(
                    embed_dim=self.config["time_freq_out_channels"]
                    * 2 ** (len(self.config["time_freq_block_num"][mod]) - 1),
                    norm_layer=self.norm_layer,
                )

                self.decoder_blocks[loc][mod] = nn.ModuleList()

                patches_resolution = self.patch_embed[loc][mod].patches_resolution

                # Drop path rate
                dpr = [
                    x.item()
                    for x in torch.linspace(
                        0, self.config["drop_path_rate"], sum(self.config["time_freq_block_num"][mod])
                    )
                ]  # stochastic depth decay rule

                for i_layer, block_num in enumerate(
                    self.config["decoder_time_freq_block_num"][mod][:-1]
                ):  # different downsample ratios
                    inverse_i_layer = len(self.config["time_freq_block_num"][mod]) - i_layer - 2
                    down_ratio = 2**inverse_i_layer
                    layer_dim = int(self.config["decoder_channels"] * down_ratio)
                    layer = BasicLayer(
                        dim=layer_dim,  # C in SWIN
                        input_resolution=(
                            patches_resolution[0] // down_ratio,  # Patch resolution = (H/4, W/4)
                            patches_resolution[1] // down_ratio,
                        ),
                        num_heads=self.config["time_freq_head_num"],
                        window_size=self.config["window_size"][mod].copy(),
                        depth=block_num,
                        drop=self.drop_rate,
                        attn_drop=self.config["attn_drop_rate"],
                        drop_path=dpr[
                            sum(self.config["time_freq_block_num"][mod][:inverse_i_layer]) : sum(
                                self.config["time_freq_block_num"][mod][: inverse_i_layer + 1]
                            )
                        ],
                        norm_layer=self.norm_layer,
                        patch_expanding=PatchExpanding
                        if (i_layer < len(self.config["time_freq_block_num"][mod]) - 2)
                        else None,
                    )
                    self.decoder_blocks[loc][mod].append(layer)
                patch_area = self.config["patch_size"]["freq"][mod][0] * self.config["patch_size"]["freq"][mod][1]
                self.decoder_pred[loc][mod] = nn.Linear(
                    self.config["decoder_channels"],
                    patch_area * self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod],
                    bias=True,
                )  #

    def pad_input(self, freq_x, loc, mod):
        stride = self.config["in_stride"][mod]
        spectrum_len = self.args.dataset_config["loc_mod_spectrum_len"][loc][mod]
        img_size = (self.num_segments, spectrum_len // stride)
        freq_input = freq_x[loc][mod]

        # [b, c, i, spectrum] -- > [b, i, spectrum, c]
        freq_input = torch.permute(freq_input, [0, 2, 3, 1])
        b, i, s, c = freq_input.shape

        # Forces both audio and seismic to have the same "img" size
        freq_input = torch.reshape(freq_input, (b, i, s // stride, c * stride))

        # Repermute back to [b, c, i, spectrum], (b, c, h, w) required in PatchEmbed
        freq_input = torch.permute(freq_input, [0, 3, 1, 2])

        # Pad [i, spectrum] to the required padding size
        padded_img_size = self.patch_embed[loc][mod].img_size
        padded_height = padded_img_size[0] - img_size[0]
        padded_width = padded_img_size[1] - img_size[1]

        # test different padding
        freq_input = F.pad(input=freq_input, pad=(0, padded_width, 0, padded_height), mode="constant", value=0)

        return freq_input, padded_img_size

    def forward(self, freq_x, class_head=True):
        """
        If class_head is False, we return the modality features; otherwise, we return the classification results.
        time-freq feature extraction --> loc fusion --> mod concatenation --> class layer
        """

        # Step 1: Feature extractions on time interval (i) and spectrum (s) domains
        mod_loc_features = {mod: [] for mod in self.modalities}
        for loc in self.locations:
            for mod in self.modalities:
                # Pad the input
                freq_input, padded_img_size = self.pad_input(freq_x, loc, mod)

                # Patch Partition and Linear Embedding
                embeded_input = self.patch_embed[loc][mod](freq_input)

                # for masked auto encoder
                print(f"masking for {mod}\n\n")
                if self.args.train_mode == "MAE":
                    window_masking(
                        embeded_input,
                        padded_img_size,
                        self.patch_embed[loc][mod].patches_resolution,
                        self.config["window_size"][mod],
                        self.mask_token[loc][mod],
                        remove=False,
                        mask_len_sparse=False,
                    )

                # Absolute positional embedding
                if self.config["APE"]:
                    embeded_input = embeded_input + self.absolute_pos_embed[loc][mod]

                # SwinTransformer Layer block
                for layer in self.freq_interval_layers[loc][mod]:
                    freq_interval_output = layer(embeded_input)
                    embeded_input = freq_interval_output

                # Unify the input channels for each modality
                freq_interval_output = self.mod_in_layers[loc][mod](freq_interval_output.reshape([b, -1]))
                freq_interval_output = freq_interval_output.reshape(b, 1, -1)

                # Append the modality feature to the list
                mod_loc_features[mod].append(freq_interval_output)

        # Concatenate the location features, [b, i, location, c]
        for mod in self.modalities:
            mod_loc_features[mod] = torch.stack(mod_loc_features[mod], dim=2)

        # Step 2: Loc feature fusion and extraction for each mod, [b, i, location, c]
        mod_features = []
        for mod in mod_loc_features:
            if len(self.locations) > 1:
                """Extract mod feature with peer-feature context"""
                b, i, locs, c = mod_loc_features[mod].shape
                mod_loc_input = mod_loc_features[mod].reshape([b * i, locs, c])
                mod_loc_context_feature = self.loc_context_layers[mod](mod_loc_input)
                mod_loc_context_feature = mod_loc_context_feature.reshape([b, i, locs, c])

                """Mod feature fusion, [b, 1, 1, c] -- > [b, c]"""
                mod_feature = self.loc_fusion_layer[mod](mod_loc_context_feature)
                mod_feature = mod_feature.flatten(start_dim=1)
                mod_features.append(mod_feature)
            else:
                mod_features.append(mod_loc_features[mod].flatten(start_dim=1))

        # Step 3: Mod concatenation, [b, 1, mod, c]
        if not class_head:
            return dict(zip(self.modalities, mod_features))
        else:
            if self.args.contrastive_framework == "Cosmo":
                """Attention-based fusion."""
                mod_features = torch.stack(mod_features, dim=1)
                mod_features = mod_features.unsqueeze(dim=1)
                sample_features = self.mod_fusion_layer(mod_features).flatten(start_dim=1)
            else:
                """Concatenation-based fusion."""
                sample_features = torch.cat(mod_features, dim=1)

            logits = self.class_layer(sample_features)
            return logits

    def decode(self, mod_features):
        """
        Decode the latent features
        """
        encoded_features = {mod_features[mod] for mod in self.modalities}
        decoded_out = {}
        for loc in self.locations:
            decoded_out[loc] = {}
            for mod in self.modalities:
                encoder_tokens = self.patch_expand[loc][mod](encoded_features[mod])

                # SwinTransformer Layer block
                for layer in self.decoder_blocks[loc][mod]:
                    decoded_tokens = layer(encoder_tokens)
                    encoder_tokens = decoded_tokens

                decoded_tokens = self.decoder_norm(decoded_tokens)
                # predictor projection
                decoded_tokens = self.decoder_pred[loc][mod](decoded_tokens)
                decoded_out[loc][mod].append(decoded_tokens)

        return decoded_out

    def encode_features(self, mod_features):
        """Encode the features by merging and then demerge

        Args:
            mod_features (dict): mod_features[mod] = features extracted for mod
        """
        mod_features = [mod_features[mod] for mod in mod_features]

        # fusion basede on attention or concatnation
        if self.config["fusion"] == "attention":
            """Attention-based fusion."""
            mod_features = torch.stack(mod_features, dim=1)
            mod_features = mod_features.unsqueeze(dim=1)
            sample_features = self.mod_fusion_layer(mod_features).flatten(start_dim=1)
        else:
            """Concatenation-based fusion."""
            sample_features = torch.cat(mod_features, dim=1)

        # fully connected layer
        encoded_sample_features = self.encoded_features_layer(sample_features)

        # extract feature for each modality
        encoded_mod_features = []
        for loc in self.locations:
            for mod in self.modalities:
                encoded_mod_feature = self.encoded_mod_extract_layer[loc][mod](encoded_sample_features)
                encoded_mod_features.append(encoded_mod_feature)

        return dict(zip(self.modalities, mod_features))

    def mae_fowrard(self, freq_x):
        mod_features = self.forward(freq_x, False)
        encoded_mod_features = self.encode_features(mod_features)
        recovered_x = self.decode(freq_x, encoded_mod_features)
        return recovered_x
