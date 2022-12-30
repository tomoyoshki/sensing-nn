# python3 train.py -gpu=0 -dataset=Parkland -stage=pretrain_classifier -model="TransformerV4"
import torch
import torch.nn as nn

from torch.nn import TransformerEncoderLayer
from input_utils.fft_utils import fft_preprocess
from models.FusionModules import TransformerFusionBlock

from models.SwinModules import (
    BasicLayer,
    PatchEmbed,
    PatchMerging,
)


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

        # Transformer Variables
        self.window_size = self.config["window_size"]
        self.drop_rate = self.config["dropout_ratio"]
        self.norm_layer = nn.LayerNorm
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Single sensor
        self.freq_interval_layers = nn.ModuleDict()  # Frequency & Interval layers
        self.patch_embed = nn.ModuleDict()
        self.mod_patch_embed = nn.ModuleDict()
        self.mod_in_layers = nn.ModuleDict()
        for loc in self.locations:
            self.freq_interval_layers[loc] = nn.ModuleDict()
            self.patch_embed[loc] = nn.ModuleDict()
            self.mod_in_layers[loc] = nn.ModuleDict()

            for mod in self.modalities:
                # Decide the spatial size for "image"
                # TODO: Set the padded image size, 10 * 20 --> 16 * 24
                stride = self.config["in_stride"][mod]
                spectrum_len = args.dataset_config["loc_mod_spectrum_len"][loc][mod]
                img_size = (self.num_segments, spectrum_len // stride)

                # Patch embedding and Linear embedding (H, W, in_channel) -> (H / p_size, W / p_size, C)
                self.patch_embed[loc][mod] = PatchEmbed(
                    img_size=img_size,
                    patch_size=self.config["patch_size"]["freq"][mod],
                    in_chans=args.dataset_config["loc_mod_in_freq_channels"][loc][mod] * stride,
                    embed_dim=self.config["time_freq_out_channels"],
                    norm_layer=self.norm_layer,
                )
                patches_resolution = self.patch_embed[loc][mod].patches_resolution

                # Swin Transformer Block
                self.freq_interval_layers[loc][mod] = nn.ModuleList()
                for i_layer, block_num in enumerate(self.config["time_freq_block_num"]):  # different downsample ratios
                    down_ratio = 2**i_layer
                    layer_dim = int(self.config["time_freq_out_channels"] * down_ratio)
                    layer = BasicLayer(
                        dim=layer_dim,  # C in SWIN
                        input_resolution=(
                            patches_resolution[0] // down_ratio,  # Patch resolution = (H/4, W/4)
                            patches_resolution[1] // down_ratio,
                        ),
                        num_heads=self.config["time_freq_head_num"],
                        window_size=self.window_size,
                        depth=block_num,
                        drop=self.drop_rate,
                        norm_layer=self.norm_layer,
                        downsample=PatchMerging if (i_layer < len(self.config["time_freq_block_num"]) - 1) else None,
                    )
                    self.freq_interval_layers[loc][mod].append(layer)

                # Unify the input channels for each modality
                self.mod_in_layers[loc][mod] = nn.Linear(
                    (patches_resolution[0] // down_ratio) * (patches_resolution[1] // down_ratio) * layer_dim,
                    self.config["mod_out_channels"],
                )

        # Mod fusion, [b, i, c], mod contextual feature extraction + mod fusion
        self.mod_context_layers = nn.ModuleDict()
        self.mod_fusion_layer = nn.ModuleDict()
        for loc in self.locations:
            """Single mod contextual feature extraction"""
            module_list = [
                TransformerEncoderLayer(
                    d_model=self.config["mod_out_channels"],
                    nhead=self.config["mod_head_num"],
                    dim_feedforward=self.config["mod_out_channels"],
                    dropout=self.config["dropout_ratio"],
                    batch_first=True,
                )
                for _ in range(self.config["mod_block_num"])
            ]
            self.mod_context_layers[loc] = nn.Sequential(*module_list)

            """Mod fusion layer for each loc"""
            self.mod_fusion_layer[loc] = TransformerFusionBlock(
                self.config["mod_out_channels"],
                self.config["mod_head_num"],
                self.config["dropout_ratio"],
                self.config["dropout_ratio"],
            )

        # Loc fusion, [b, i, c], loc contextual feature extraction + loc fusion
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
        self.loc_context_layer = nn.Sequential(*module_list)
        self.loc_fusion_layer = TransformerFusionBlock(
            self.config["loc_out_channels"],
            self.config["loc_head_num"],
            self.config["dropout_ratio"],
            self.config["dropout_ratio"],
        )

        # Classification layer
        self.class_layer = nn.Sequential(
            nn.Linear(self.config["loc_out_channels"], self.config["fc_dim"]),
            nn.GELU(),
            nn.Linear(self.config["fc_dim"], args.dataset_config["num_classes"]),
            nn.Sigmoid() if args.multi_class else nn.Softmax(dim=1),
        )

    def forward(self, org_time_x, augmenter):
        args = self.args

        # Step 0: Move data to target device
        for loc in org_time_x:
            for mod in org_time_x[loc]:
                org_time_x[loc][mod] = org_time_x[loc][mod].to(self.device)

        # Step 1 Optional data augmentation
        augmented_time_x = augmenter.augment_forward(org_time_x)

        # Step 2: FFT on the time domain data; c: 1 -> 2
        freq_x = fft_preprocess(augmented_time_x, args)

        # Step 3: Feature extractions on time interval (i) and spectrum (s) domains
        loc_mod_features = dict()
        for loc in self.locations:
            loc_mod_features[loc] = []
            for mod in self.modalities:
                stride = self.config["in_stride"][mod]

                # TODO: Add a padding function to support more levels in SWIN hierarchy, e.g., 3 levels --> multiples of 4
                freq_input = freq_x[loc][mod]

                # [b, c, i, spectrum] -- > [b, i, spectrum, c]
                freq_input = torch.permute(freq_x[loc][mod], [0, 2, 3, 1])
                b, i, s, c = freq_input.shape

                # Forces both audio and seismic to have the same "img" size
                freq_input = torch.reshape(freq_input, (b, i, s // stride, c * stride))

                # Repermute back to [b, c, i, spectrum], (b, c, h, w) required in PatchEmbed
                freq_input = torch.permute(freq_input, [0, 3, 1, 2])

                # Patch Partition and Linear Embedding
                embeded_input = self.patch_embed[loc][mod](freq_input)

                # SwinTransformer Layer block
                for layer in self.freq_interval_layers[loc][mod]:
                    freq_interval_output = layer(embeded_input)
                    embeded_input = freq_interval_output

                # Unify the input channels for each modality
                freq_interval_output = self.mod_in_layers[loc][mod](freq_interval_output.reshape([b, -1]))
                freq_interval_output = freq_interval_output.reshape(b, 1, -1)

                # Append the result
                loc_mod_features[loc].append(freq_interval_output)

            # Stack results from different modalities, [b, 1, s, c]
            loc_mod_features[loc] = torch.stack(loc_mod_features[loc], dim=2)

        # Step 4: Loc mod feature extraction, [b, i, location, c]
        loc_features = []
        for loc in loc_mod_features:
            """Extract mod feature with peer-feature context"""
            b, i, mods, c = loc_mod_features[loc].shape
            loc_mod_input = loc_mod_features[loc].reshape([b * i, mods, c])
            loc_mod_context_feature = self.mod_context_layers[loc](loc_mod_input)
            loc_mod_context_feature = loc_mod_context_feature.reshape([b, i, mods, c])

            """Mod feature fusion, [b, 1, 1, c]"""
            loc_feature = self.mod_fusion_layer[loc](loc_mod_context_feature)
            loc_features.append(loc_feature)
        loc_features = torch.stack(loc_features, dim=2)

        # Step 5: Location-level fusion, [b, 1, l, c]
        if len(self.locations) > 1:
            b, i, l, c = loc_features.shape
            loc_input = loc_features.reshape([b * i, l, c])
            loc_context_feature = self.loc_context_layer(loc_input)
            loc_context_feature = loc_context_feature.reshape([b, i, l, c])
            sample_features = self.loc_fusion_layer(loc_context_feature)
        else:
            sample_features = loc_features.squeeze(dim=2)

        # Step 6: Classification Layers
        sample_features = sample_features.flatten(start_dim=1)
        logit = self.class_layer(sample_features)

        return logit
