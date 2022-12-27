# python3 train.py -gpu=0 -dataset=Parkland -stage=pretrain_classifier -model="TransformerV4"
import os
import time
import math
import torch
import torch.nn as nn

from torch.nn import TransformerEncoderLayer
from input_utils.fft_utils import fft_preprocess
from models.FusionModules import TransformerFusionBlock

from models.SwinModules import BasicLayer, PatchEmbed, PatchMerging, SwinTransformerBlock
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class PositionalEncoding(nn.Module):
    def __init__(self, out_channel, dropout=0.1, max_len=20):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.out_channel = out_channel

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, out_channel, 2) * (-math.log(10000.0) / out_channel))
        pe = torch.zeros(max_len, 1, out_channel)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x.permute(1, 0, 2) * math.sqrt(self.out_channel)
        x = x + self.pe[: x.size(0)]
        x = x.permute(1, 0, 2)
        return self.dropout(x)


class TransformerV4(nn.Module):
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
        self.device = args.device
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.num_segments = args.dataset_config["num_segments"]
        self.spectrum_length = args.dataset_config["loc_mod_spectrum_len"]["shake"]
        
        # Setting up SWIN Transformer modules
        self.depths = self.config["depths"]
        self.num_layers = len(self.depths)
        self.num_heads= self.config["num_heads"]
        self.ape = self.config["APE"]
        self.patch_norm = self.config["patch_norm"]
        
        self.embed_dim_ = 2
        self.num_features = int(self.embed_dim_ * 2 ** (self.num_layers - 1))
        self.norm_layer = nn.LayerNorm
        self.norm = self.norm_layer(self.num_features)
        self.mlp_ratio = self.config["mlp_ratio"]
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.pos_drop = nn.Dropout(p=self.config["dropout_ratio"])

        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(self.depths))]  # stochastic depth decay rule

        # decide max seq len
        max_len = 0
        for loc in self.locations:
            for mod in self.modalities:
                stride = self.config["in_stride"][mod]
                max_len = max(max_len, int(args.dataset_config["loc_mod_spectrum_len"][loc][mod] / stride))

        # stochastic depth

        # Single mod,  [b * i, s/stride, c * stride]
        self.freq_context_layers = nn.ModuleDict()
        self.freq_fusion_layers = nn.ModuleDict()
        self.interval_context_layers = nn.ModuleDict()
        self.interval_fusion_layers = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        self.patch_embed = nn.ModuleDict()
        self.transition_fc_layers = nn.ModuleDict()
        self.num_patches = dict()
        self.patches_resolution = dict()
        self.embed_dim = dict()

        for loc in self.locations:
            self.patch_embed[loc] = nn.ModuleDict()
            self.norms[loc] = nn.ModuleDict()
            self.transition_fc_layers[loc] = nn.ModuleDict()
            self.num_patches[loc] = dict()
            self.patches_resolution[loc] = dict()
            self.embed_dim[loc] = dict()
            for mod in self.modalities:
                stride = self.config["in_stride"][mod]
                input_channels = args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
                spectrum_len = args.dataset_config["loc_mod_spectrum_len"][loc][mod]
                image_size = (10, 20)
                self.embed_dim[loc][mod] = self.config["embed_dim"][mod]
                self.norms[loc][mod] = self.norm_layer(int(self.embed_dim[loc][mod] * 2 ** (self.num_layers - 1)))
                print("Mod: ", mod, " has ", int(self.embed_dim[loc][mod] * 2 ** (self.num_layers - 1)))
                self.transition_fc_layers[loc][mod] = nn.Linear(int(self.embed_dim[loc][mod] * 2 ** (self.num_layers - 1)), self.config["fc_dim"])
                self.patch_embed[loc][mod] = PatchEmbed(img_size=image_size,
                                                patch_size=self.config["patch_embed"][mod]["patch_size"],
                                                in_chans=input_channels,
                                                embed_dim=self.embed_dim[loc][mod],
                                                norm_layer=self.norm_layer if self.patch_norm else None)
                self.num_patches[loc][mod] = self.patch_embed[loc][mod].num_patches
                self.patches_resolution[loc][mod] = self.patch_embed[loc][mod].patches_resolution

        for loc in self.locations:
            self.freq_context_layers[loc] = nn.ModuleDict()
            self.freq_fusion_layers[loc] = nn.ModuleDict()
            self.interval_context_layers[loc] = nn.ModuleDict()
            self.interval_fusion_layers[loc] = nn.ModuleDict()

            for mod in self.modalities:
                stride = self.config["in_stride"][mod]
                input_channels = args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
                patch_resolutions = self.patches_resolution[loc][mod]
                module_list = [
                        BasicLayer(dim=int(self.embed_dim[loc][mod] * 2 ** i_layer),
                                    input_resolution=(
                                        patch_resolutions[0] // (2 ** i_layer),
                                        patch_resolutions[1] // (2 ** i_layer)
                                    ),
                                    depth=self.depths[i_layer],
                                    num_heads=self.num_heads[i_layer],
                                    window_size=self.config["window_size"][mod],
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=self.config["qkv_bias"], qk_scale=None,
                                    drop=self.config["dropout_ratio"], attn_drop=self.config["dropout_ratio"],
                                    drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                                    norm_layer=self.norm_layer,
                                    downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
                                    )
                        for i_layer in range(self.num_layers)
                    ]
                self.freq_context_layers[loc][mod] = nn.Sequential(*module_list)
        # Classification
        self.class_layer = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.config["fc_dim"] * 2, args.dataset_config["num_classes"]),
            nn.Sigmoid() if args.multi_class else nn.Softmax(dim=1),
        )

    def forward(self, org_time_x, augmenter):
        """
        The forward function of SWIM Transformer.
        Args:
            org_time_x (_type_): time_x is a dictionary consisting of the Tensor input of each input modality.
                        For each modality, the data is in (b, c (2 * 3 or 1), i (intervals), s (spectrum)) format.
        """
        args = self.args

        # Step 0: Move data to target device
        for loc in org_time_x:
            for mod in org_time_x[loc]:
                org_time_x[loc][mod] = org_time_x[loc][mod].to(args.device)

        # Step 1 Optional data augmentation
        # augmented_time_x = augmenter.augment_forward(org_time_x)
        augmented_time_x = org_time_x
        
        # Step 3: FFT on the time domain data
        freq_x = fft_preprocess(augmented_time_x, args)
    
        # Step 5: Single (loc, mod, freq) feature extraction, [b * i, int(s / stride), stride * c]
        for loc in self.locations:
            result = []
            for mod in self.modalities:
                """freq feature extraction"""
                freq_input = freq_x[loc][mod]
                b, c, i, s = freq_input.shape
                stride = self.config["in_stride"][mod]
                freq_input = torch.reshape(freq_input, (b * stride, c, i, s // stride))
                
                # print(freq_input.shape)
                freq_input = self.patch_embed[loc][mod](freq_input)
                freq_input = self.pos_drop(freq_input)
                # print(freq_input.shape)
                freq_context_feature = self.freq_context_layers[loc][mod](freq_input)
                pB, pS, pC = freq_context_feature.shape
                freq_context_feature = torch.reshape(freq_context_feature, (pB // stride, pS * stride, pC))
                # print(freq_context_feature.shape)
                freq_context_feature = self.norms[loc][mod](freq_context_feature)  # B L C
                freq_context_feature = self.avgpool(freq_context_feature.transpose(1, 2))  # B C 1
                freq_context_feature = torch.flatten(freq_context_feature, 1)
                freq_context_feature = self.transition_fc_layers[loc][mod](freq_context_feature)
                # print(freq_context_feature.shape)
                result.append(freq_context_feature)
            res = torch.stack(result, dim=2)
            sample_features = res.squeeze(dim=2)
            sample_features = sample_features.flatten(start_dim=1)
            logit = self.class_layer(sample_features)
            return logit