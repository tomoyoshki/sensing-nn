import os
import time
import math
import torch
import torch.nn as nn

from torch.nn import TransformerEncoderLayer
from input_utils.fft_utils import fft_preprocess
from models.FusionModules import TransformerFusionBlock


class PositionalEncoding(nn.Module):
    def __init__(self, out_channel, dropout=0.1, max_len=20):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

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
        x = x.permute(1, 0, 2)
        x = x + self.pe[: x.size(0)]
        x = x.permute(1, 0, 2)
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.config = args.dataset_config["Transformer"]
        self.device = args.device
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.num_segments = args.dataset_config["num_segments"]

        # Single mod,  [b, i, s*c]
        self.loc_mod_feature_extraction_layers = nn.ModuleDict()
        for loc in self.locations:
            self.loc_mod_feature_extraction_layers[loc] = nn.ModuleDict()
            for mod in self.modalities:
                spectrum_len = args.dataset_config["loc_mod_spectrum_len"][loc][mod]
                feature_channels = args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
                module_list = [nn.Linear(spectrum_len * feature_channels, self.config["loc_mod_out_channels"])] + [
                    TransformerEncoderLayer(
                        d_model=self.config["loc_mod_out_channels"],
                        nhead=self.config["loc_mod_head_num"],
                        dim_feedforward=self.config["loc_mod_out_channels"],
                        dropout=self.config["dropout_ratio"],
                        batch_first=True,
                    )
                    for _ in range(self.config["loc_mod_block_num"])
                ]
                self.loc_mod_feature_extraction_layers[loc][mod] = nn.Sequential(*module_list)

        # Single loc, [b, i, c]
        self.mod_fusion_layers = nn.ModuleDict()
        self.loc_feature_extraction_layers = nn.ModuleDict()
        for loc in self.locations:
            self.mod_fusion_layers[loc] = TransformerFusionBlock(
                self.config["loc_mod_out_channels"],
                self.config["loc_head_num"],
                self.config["dropout_ratio"],
                self.config["dropout_ratio"],
            )
            module_list = [nn.Linear(self.config["loc_mod_out_channels"], self.config["loc_out_channels"])] + [
                TransformerEncoderLayer(
                    d_model=self.config["loc_out_channels"],
                    nhead=self.config["loc_head_num"],
                    dim_feedforward=self.config["loc_out_channels"],
                    dropout=self.config["dropout_ratio"],
                    batch_first=True,
                )
                for _ in range(self.config["loc_block_num"])
            ]
            self.loc_feature_extraction_layers[loc] = nn.Sequential(*module_list)

        # Single interval, [b, i, c]
        self.loc_fusion_layer = TransformerFusionBlock(
            self.config["loc_out_channels"],
            self.config["sample_head_num"],
            self.config["dropout_ratio"],
            self.config["dropout_ratio"],
        )
        module_list = [nn.Linear(self.config["loc_out_channels"], self.config["sample_out_channels"])] + [
            TransformerEncoderLayer(
                d_model=self.config["sample_out_channels"],
                nhead=self.config["sample_head_num"],
                dim_feedforward=self.config["sample_out_channels"],
                dropout=self.config["dropout_ratio"],
                batch_first=True,
            )
            for _ in range(self.config["sample_block_num"])
        ]
        self.sample_feature_extraction_layer = nn.Sequential(*module_list)

        # Time fusion, [b, c]
        self.time_fusion_layer = TransformerFusionBlock(
            self.config["sample_out_channels"],
            self.config["sample_head_num"],
            self.config["dropout_ratio"],
            self.config["dropout_ratio"],
        )

        # Classification
        self.class_layer = nn.Sequential(
            nn.Linear(self.config["sample_out_channels"], self.config["fc_dim"]),
            nn.GELU(),
            nn.Linear(self.config["fc_dim"], args.dataset_config["num_classes"]),
            nn.Sigmoid() if args.multi_class else nn.Softmax(dim=1),
        )

    def forward(self, org_time_x, miss_simulator):
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

        # Step 1: Add noise to the time-domain data
        noisy_time_x, gt_loc_miss_ids = miss_simulator.generate_noise(org_time_x)

        # Step 2: Noise experiment at time level
        proc_time_x = (
            miss_simulator.detect_and_handle_noise(noisy_time_x, gt_loc_miss_ids)
            if args.noise_position == "time"
            else noisy_time_x
        )
        time_handler_loss = (
            self.calc_input_handler_loss(miss_simulator, org_time_x, proc_time_x)
            if args.noise_position == "time"
            else 0
        )

        # Step 3: FFT on the time domain data
        org_freq_x = fft_preprocess(proc_time_x, args)

        # Step 4: Noise experiment at frequency level
        proc_freq_x = (
            miss_simulator.detect_and_handle_noise(org_freq_x, gt_loc_miss_ids)
            if args.noise_position == "frequency"
            else org_freq_x
        )
        freq_handler_loss = (
            self.calc_input_handler_loss(miss_simulator, org_freq_x, proc_freq_x)
            if args.noise_position == "frequency"
            else 0
        )

        # Step 5: Single (loc, mod) feature extraction, [b, i, s, c]
        org_loc_mod_features = dict()
        for loc in self.locations:
            org_loc_mod_features[loc] = []
            for mod in self.modalities:
                # [b, c, i, s] -- > [b, i, s, c]
                loc_mod_input = torch.permute(proc_freq_x[loc][mod], [0, 2, 3, 1])
                b, i, s, c = loc_mod_input.shape
                loc_mod_input = torch.reshape(loc_mod_input, (b, i, s * c))
                org_loc_mod_features[loc].append(self.loc_mod_feature_extraction_layers[loc][mod](loc_mod_input))
            org_loc_mod_features[loc] = torch.stack(org_loc_mod_features[loc], dim=2)

        # Step 6:  Noise experiment at feature level
        proc_loc_mod_features = (
            miss_simulator.detect_and_handle_noise(org_loc_mod_features, gt_loc_miss_ids)
            if args.noise_position == "feature"
            else org_loc_mod_features
        )

        # Step 7: Fusion + Classification layers
        proc_fused_loc_features, proc_logits = self.classification_forward(proc_loc_mod_features, miss_simulator)

        # Step 8: Compute the handler loss for feature level
        if args.noise_position == "feature":
            feature_handler_loss = self.calc_feature_handler_loss(
                miss_simulator,
                org_loc_mod_features,
                proc_loc_mod_features,
                proc_fused_loc_features,
                proc_logits,
            )
        else:
            feature_handler_loss = 0

        # get the right handler loss
        if args.noise_position == "time":
            handler_loss = time_handler_loss
        elif args.noise_position == "frequency":
            handler_loss = freq_handler_loss
        else:
            handler_loss = feature_handler_loss

        return proc_logits, handler_loss

    def classification_forward(self, loc_mod_features, miss_simulator):
        """Separate the fusion and classification layer forward into this function.

        Args:
            loc_mod_features (_type_): dict of {loc: loc_features}
            return_fused_features (_type_, optional): Flag indicator. Defaults to False.
        """
        # Step 1: Modality-level fusion
        loc_fused_features = {}
        for loc in loc_mod_features:
            if self.args.miss_handler in {
                "ResilientHandler",
                "FakeHandler",
                "GateHandler",
                "NonlinearResilientHandler",
            }:
                loc_fused_features[loc] = self.mod_fusion_layers[loc](
                    loc_mod_features[loc],
                    miss_simulator.miss_handler.rescale_factors[loc],
                )
            else:
                loc_fused_features[loc] = self.mod_fusion_layers[loc](loc_mod_features[loc])

        # Step 2: Location feature extraction, [b, i, s, c]
        loc_features = []
        for loc in loc_mod_features:
            outputs = self.loc_feature_extraction_layers[loc](loc_fused_features[loc])
            loc_features.append(outputs)
        loc_features = torch.stack(loc_features, dim=2)

        # Step 3: Location-level fusion, [b, i, c]
        interval_features = self.loc_fusion_layer(loc_features)
        interval_features = self.sample_feature_extraction_layer(interval_features)
        interval_features = torch.unsqueeze(interval_features, dim=1)

        # Step 4: Time fusion
        sample_features = self.time_fusion_layer(interval_features)
        sample_features = torch.flatten(sample_features, start_dim=1)

        # Step 5: Classification
        outputs = torch.flatten(sample_features, start_dim=1)
        logits = self.class_layer(outputs)

        return loc_fused_features, logits

    def calc_input_handler_loss(self, miss_simulator, org_loc_inputs, proc_loc_inputs):
        """Calculate the loss for the input handler at the time/frequency level.

        Args:
            org_loc_inputs (_type_): {loc: {mod: [b, c, i, s]}}
            proc_loc_inputs (_type_): {loc: {mod: [b, c, i, s]}}
        """
        input_handler_loss = miss_simulator.miss_handler.handler_loss_all_locs(org_loc_inputs, proc_loc_inputs)

        return input_handler_loss

    def calc_feature_handler_loss(
        self,
        miss_simulator,
        org_loc_mod_features,
        proc_loc_mod_features=None,
        proc_fused_loc_features=None,
        proc_logits=None,
    ):
        """
        Calculate the handler loss according to the given loss level.
        """
        if miss_simulator.miss_handler.loss_feature_level == "mod":
            feature_handler_loss = miss_simulator.miss_handler.handler_loss_all_locs(
                org_loc_mod_features,
                proc_loc_mod_features,
            )
        elif miss_simulator.miss_handler.loss_feature_level == "loc":
            org_fused_loc_features, org_logits = self.classification_forward(org_loc_mod_features, miss_simulator)
            feature_handler_loss = miss_simulator.miss_handler.handler_loss_all_locs(
                org_fused_loc_features,
                proc_fused_loc_features,
            )
        elif miss_simulator.miss_handler.loss_feature_level == "logit":
            org_fused_loc_features, org_logits = self.classification_forward(org_loc_mod_features, miss_simulator)
            feature_handler_loss = miss_simulator.miss_handler.handler_loss_all_locs(
                org_logits,
                proc_logits,
            )
        elif miss_simulator.miss_handler.loss_feature_level == "mod+logit":
            org_fused_loc_features, org_logits = self.classification_forward(org_loc_mod_features, miss_simulator)
            feature_handler_loss = miss_simulator.miss_handler.handler_loss_all_locs(
                org_loc_mod_features,
                org_logits,
                proc_loc_mod_features,
                proc_logits,
            )
        else:
            raise Exception("Unknown loss feature level: {}".format(miss_simulator.miss_handler.loss_feature_level))

        return feature_handler_loss
