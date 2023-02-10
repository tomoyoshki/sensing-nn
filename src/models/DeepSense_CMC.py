import os
import time
import torch
import torch.nn as nn

from models.ConvModules import ConvBlock
from models.FusionModules import MeanFusionBlock, SelfAttentionFusionBlock
from models.RecurrentModule import RecurrentBlock
from input_utils.normalize import normalize_input
from models.FusionModules import TransformerFusionBlock


class DeepSense_CMC(nn.Module):
    def __init__(self, args, self_attention=False) -> None:
        """The initialization for the DeepSense class.
        Design: Single (interval, loc, mod) feature -->
                Single (interval, loc) feature -->
                Single interval feature -->
                GRU -->
                Logits
        Args:
            num_classes (_type_): _description_
        """
        super().__init__()
        self.args = args
        self.self_attention = self_attention
        self.config = args.dataset_config["DeepSense"]
        self.device = args.device
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.multi_location_flag = len(self.locations) > 1

        """define the architecture"""
        # Step 1: Single (loc, mod) feature
        self.loc_mod_extractors = nn.ModuleDict()
        for loc in self.locations:
            self.loc_mod_extractors[loc] = nn.ModuleDict()
            for mod in self.modalities:
                if type(self.config["loc_mod_conv_lens"]) is dict:
                    """for acoustic processing in Parkland data"""
                    conv_lens = self.config["loc_mod_conv_lens"][mod]
                    in_stride = self.config["loc_mod_in_conv_stride"][mod]
                else:
                    conv_lens = self.config["loc_mod_conv_lens"]
                    in_stride = 1

                # define the extractor
                self.loc_mod_extractors[loc][mod] = ConvBlock(
                    in_channels=args.dataset_config["loc_mod_in_freq_channels"][loc][mod],
                    out_channels=self.config["loc_mod_out_channels"],
                    in_spectrum_len=args.dataset_config["loc_mod_spectrum_len"][loc][mod],
                    conv_lens=conv_lens,
                    dropout_ratio=self.config["dropout_ratio"],
                    num_inter_layers=self.config["loc_mod_conv_inter_layers"],
                    in_stride=in_stride,
                )

        # Step 3: Loc fusion
        self.loc_fusion_layers = nn.ModuleDict()
        self.mod_extractors = nn.ModuleDict()
        for mod in self.modalities:
            if self.self_attention:
                self.loc_fusion_layers[mod] = SelfAttentionFusionBlock()
            else:
                self.loc_fusion_layers[mod] = MeanFusionBlock()

            self.mod_extractors[mod] = ConvBlock(
                in_channels=1,
                out_channels=self.config["loc_out_channels"],
                in_spectrum_len=self.config["loc_mod_out_channels"],
                conv_lens=self.config["loc_conv_lens"],
                dropout_ratio=self.config["dropout_ratio"],
                num_inter_layers=self.config["loc_conv_inter_layers"],
            )

        # Step 5: GRU
        self.recurrent_layers = nn.ModuleDict()
        for mod in self.modalities:
            self.recurrent_layers[mod] = RecurrentBlock(
                in_channel=self.config["loc_out_channels"],
                out_channel=self.config["recurrent_dim"],
                num_layers=self.config["recurrent_layers"],
                dropout_ratio=self.config["dropout_ratio"],
            )

        # mod fusion layer
        if args.contrastive_framework == "Cosmo":
            "Attention fusion for Cosmo"
            self.mod_fusion_layer = TransformerFusionBlock(
                self.config["recurrent_dim"] * 2,
                4,
                self.config["dropout_ratio"],
                self.config["dropout_ratio"],
            )
            sample_dim = self.config["recurrent_dim"] * 2
        else:
            sample_dim = self.config["recurrent_dim"] * 2 * len(self.modalities)

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

    def forward(self, freq_x, class_head=True):
        """The forward function of DeepSense.
        Args:
            time_x (_type_): time_x is a dictionary consisting of the Tensor input of each input modality.
                        For each modality, the data is in (b, c (2 * 3 or 1), i (intervals), s (spectrum)) format.
        """
        # Step 1: Single (loc, mod) feature extraction, (b, c, i)
        loc_mod_features = {mod: [] for mod in self.modalities}
        for loc in self.locations:
            for mod in self.modalities:
                loc_mod_features[mod].append(self.loc_mod_extractors[loc][mod](freq_x[loc][mod]))

        for mod in loc_mod_features:
            loc_mod_features[mod] = torch.stack(loc_mod_features[mod], dim=3)

        # Step 2: Location fusion, (b, c, i)
        mod_interval_features = {mod: [] for mod in self.modalities}
        for mod in self.modalities:
            if not self.multi_location_flag:
                mod_interval_features[mod] = loc_mod_features[mod].squeeze(3)
            else:
                fused_mod_feature = self.loc_fusion_layers[mod](loc_mod_features[mod])
                extracted_mod_features = self.mod_extractors[mod](fused_mod_feature)
                mod_interval_features[mod] = extracted_mod_features

        # Step 3: Interval Fusion for each modality, [b, c, i]
        mod_features = []
        for mod in self.modalities:
            mod_feature = self.recurrent_layers[mod](mod_interval_features[mod])
            mod_features.append(mod_feature.flatten(start_dim=1))

        # Step 4: Mod concatenation, [b, 1, mod, c]
        if not class_head:
            return dict(zip(self.modalities, mod_features))
        else:
            if self.args.contrastive_framework == "Cosmo":
                """Attention-based Fusion"""
                mod_features = torch.stack(mod_features, dim=1)
                mod_features = mod_features.unsqueeze(dim=1)
                sample_features = self.mod_fusion_layer(mod_features).flatten(start_dim=1)
            else:
                """Concatenation-based Fusion"""
                sample_features = torch.cat(mod_features, dim=1)

            logits = self.class_layer(sample_features)
            return logits
