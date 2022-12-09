import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from models.ConvModules import ConvBlock
from models.FusionModules import MeanFusionBlock
from input_utils.fft_utils import fft_preprocess


class ResNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = args.dataset_config["ResNet"]
        self.device = args.device
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.multi_location_flag = len(self.locations) > 1

        """(loc, mod) feature extraction"""
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
                    interval_num=self.config["interval_num"],
                    conv_lens=conv_lens,
                    dropout_ratio=self.config["dropout_ratio"],
                    num_inter_layers=self.config["loc_mod_conv_inter_layers"],
                    in_stride=in_stride,
                )

        """mod feature fusion and loc feature extraction"""
        self.mod_fusion_layers = nn.ModuleDict()
        for loc in self.locations:
            self.mod_fusion_layers[loc] = MeanFusionBlock()
        self.loc_extractors = nn.ModuleDict()
        for loc in self.locations:
            self.loc_extractors[loc] = ConvBlock(
                in_channels=1,
                out_channels=self.config["loc_out_channels"],
                in_spectrum_len=self.config["loc_mod_out_channels"],
                interval_num=1,
                conv_lens=self.config["loc_conv_lens"],
                dropout_ratio=self.config["dropout_ratio"],
                num_inter_layers=self.config["loc_conv_inter_layers"],
            )

        """loc feature fusion and sample feature extraction"""
        if self.multi_location_flag:
            self.loc_fusion_layer = MeanFusionBlock()
            self.interval_extractor = ConvBlock(
                in_channels=1,
                out_channels=self.config["loc_out_channels"],
                in_spectrum_len=self.config["loc_out_channels"],
                interval_num=1,
                conv_lens=self.config["loc_conv_lens"],
                dropout_ratio=self.config["dropout_ratio"],
                num_inter_layers=self.config["loc_conv_inter_layers"],
            )

        # Step 5: Classification layer
        self.class_layer = nn.Sequential(
            nn.Linear(self.config["loc_out_channels"], self.config["fc_dim"]),
            nn.ReLU(),
            nn.Linear(self.config["fc_dim"], args.dataset_config["num_classes"]),
            nn.Sigmoid() if args.multi_class else nn.Softmax(dim=1),
        )

    def forward(self, org_time_x, miss_simulator):
        """The forward function of ResNet.
        Args:
            x (_type_): x is a dictionary consisting of the Tensor input of each input modality.
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

        # Step 5: Single (loc, mod) feature extraction
        org_loc_mod_features = dict()
        for loc in self.locations:
            org_loc_mod_features[loc] = []
            for mod in self.modalities:
                org_loc_mod_features[loc].append(self.loc_mod_extractors[loc][mod](proc_freq_x[loc][mod]))
            org_loc_mod_features[loc] = torch.stack(org_loc_mod_features[loc], dim=3)

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
        # Step 3.1: Feature fusion for different mods in the same location
        fused_loc_features = dict()
        for loc in self.locations:
            if self.args.miss_handler in {
                "ResilientHandler",
                "FakeHandler",
                "GateHandler",
                "NonlinearResilientHandler",
            }:
                fused_loc_features[loc] = self.mod_fusion_layers[loc](
                    loc_mod_features[loc],
                    miss_simulator.miss_handler.rescale_factors[loc],
                )
            else:
                fused_loc_features[loc] = self.mod_fusion_layers[loc](loc_mod_features[loc])

        # Step 3.2: Feature extraction for each location
        extracted_loc_features = dict()
        for loc in self.locations:
            extracted_loc_features[loc] = self.loc_extractors[loc](fused_loc_features[loc])

        # Step 4: Location fusion, (b, c, i)
        if not self.multi_location_flag:
            final_feature = extracted_loc_features[self.locations[0]]
        else:
            interval_fusion_input = torch.stack([extracted_loc_features[loc] for loc in self.locations], dim=3)
            fused_feature = self.loc_fusion_layer(interval_fusion_input)
            final_feature = self.interval_extractor(fused_feature)

        # Classification on features
        final_feature = torch.flatten(final_feature, start_dim=1)
        logits = self.class_layer(final_feature)

        return fused_loc_features, logits

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
