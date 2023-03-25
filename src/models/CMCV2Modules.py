# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class CMCV2(nn.Module):
    """
    An enhanced CMC version that not only extract shared information but also extract modality specific information.
    """

    def __init__(self, args, backbone):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(CMCV2, self).__init__()

        self.args = args
        self.config = args.dataset_config["CMC"]
        self.backbone_config = args.dataset_config[args.model]
        self.modalities = args.dataset_config["modality_names"]

        # build encoders
        self.backbone = backbone

        # define the projector
        self.in_fc_dim = (
            self.backbone_config["recurrent_dim"] * 2
            if args.model == "DeepSense"
            else self.backbone_config["loc_out_channels"]
        )
        self.out_dim = self.config["emb_dim"]
        self.mod_projectors = nn.ModuleDict()
        for mod in self.modalities:
            self.mod_projectors[mod] = nn.Sequential(
                nn.Linear(self.in_fc_dim, self.config["emb_dim"]),
                nn.ReLU(),
                nn.Linear(self.config["emb_dim"], self.config["emb_dim"]),
            )

    def forward(self, aug_freq_input1, aug_freq_input2):
        """
        Input:
            freq_input1: Input of the first augmentation.
            freq_input2: Input of the second augmentation.
        Output:
            mod_features1: Projected mod features of the first augmentation.
            mod_features2: Projected mod features of the second augmentation.
        """
        # compute features
        mod_features1 = self.backbone(aug_freq_input1, class_head=False)
        mod_features2 = self.backbone(aug_freq_input2, class_head=False)

        # project mod features
        out_mod_features1, out_mod_features2 = {}, {}
        for mod in self.modalities:
            out_mod_features1[mod] = self.mod_projectors[mod](mod_features1[mod])
            out_mod_features2[mod] = self.mod_projectors[mod](mod_features2[mod])

        return out_mod_features1, out_mod_features2
