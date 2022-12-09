import os
import time
import torch
import torch.nn as nn

from models.ConvModules import ConvBlock
from models.FusionModules import MeanFusionBlock, SelfAttentionFusionBlock
from models.RecurrentModule import RecurrentBlock
from input_utils.fft_utils import fft_preprocess
from input_utils.normalize import normalize_input


class DeepSense(nn.Module):
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

        # Step 3: Single (loc), modality fusion
        self.mod_fusion_layers = nn.ModuleDict()
        for loc in self.locations:
            if self.self_attention:
                self.mod_fusion_layers[loc] = SelfAttentionFusionBlock()
            else:
                self.mod_fusion_layers[loc] = MeanFusionBlock()
        self.loc_extractors = nn.ModuleDict()
        for loc in self.locations:
            self.loc_extractors[loc] = ConvBlock(
                in_channels=1,
                out_channels=self.config["loc_out_channels"],
                in_spectrum_len=self.config["loc_mod_out_channels"],
                conv_lens=self.config["loc_conv_lens"],
                dropout_ratio=self.config["dropout_ratio"],
                num_inter_layers=self.config["loc_conv_inter_layers"],
            )

        # Step 4: Location fusion (Make it identity if we only have one location)
        if self.multi_location_flag:
            self.loc_fusion_layer = MeanFusionBlock()
            self.interval_extractor = ConvBlock(
                in_channels=1,
                out_channels=self.config["loc_out_channels"],
                in_spectrum_len=self.config["loc_out_channels"],
                conv_lens=self.config["loc_conv_lens"],
                dropout_ratio=self.config["dropout_ratio"],
                num_inter_layers=self.config["loc_conv_inter_layers"],
            )

        # Step 5: GRU
        self.recurrent_layer = RecurrentBlock(
            in_channel=self.config["loc_out_channels"],
            out_channel=self.config["recurrent_dim"],
            num_layers=self.config["recurrent_layers"],
            dropout_ratio=self.config["dropout_ratio"],
        )

        # Step 6: Classification layer
        self.class_layer = nn.Sequential(
            nn.Linear(self.config["recurrent_dim"] * 2, self.config["fc_dim"]),
            nn.ReLU(),
            nn.Linear(self.config["fc_dim"], args.dataset_config["num_classes"]),
            nn.Sigmoid() if args.multi_class else nn.Softmax(dim=1),
        )

    def forward(self, org_time_x, augmenter):
        """The forward function of DeepSense.
        Args:
            time_x (_type_): time_x is a dictionary consisting of the Tensor input of each input modality.
                        For each modality, the data is in (b, c (2 * 3 or 1), i (intervals), s (spectrum)) format.
        """
        args = self.args

        # Step 0: Move data to target device
        for loc in org_time_x:
            for mod in org_time_x[loc]:
                org_time_x[loc][mod] = org_time_x[loc][mod].to(args.device)

        # Step 1 Optional data augmentation
        augmented_time_x = augmenter.augment_forward(org_time_x)

        # Step 2: FFT on the time domain data
        freq_x = fft_preprocess(augmented_time_x, args)

        # Step 3: Single (loc, mod) feature extraction, (b, c, i)
        loc_mod_features = dict()
        for loc in self.locations:
            loc_mod_features[loc] = []
            for mod in self.modalities:
                loc_mod_features[loc].append(self.loc_mod_extractors[loc][mod](freq_x[loc][mod]))
            loc_mod_features[loc] = torch.stack(loc_mod_features[loc], dim=3)

        # Step 4: Feature fusion for different mods in the same location
        fused_loc_features = dict()
        for loc in self.locations:
            fused_loc_features[loc] = self.mod_fusion_layers[loc](loc_mod_features[loc])

        # Step 5: Feature extraction for each location
        extracted_loc_features = dict()
        for loc in self.locations:
            extracted_loc_features[loc] = self.loc_extractors[loc](fused_loc_features[loc])

        # Step 6: Location fusion, (b, c, i)
        if not self.multi_location_flag:
            extracted_interval_feature = extracted_loc_features[self.locations[0]]
        else:
            interval_fusion_input = torch.stack([extracted_loc_features[loc] for loc in self.locations], dim=3)
            fused_interval_feature = self.loc_fusion_layer(interval_fusion_input)
            extracted_interval_feature = self.interval_extractor(fused_interval_feature)

        # Step 7: Time recurrent layer
        recurrent_feature = self.recurrent_layer(extracted_interval_feature)

        # Step 8: Classification
        logits = self.class_layer(recurrent_feature)

        return logits