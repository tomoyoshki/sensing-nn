# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class CMC(nn.Module):
    """
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, args, backbone):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(CMC, self).__init__()

        self.args = args
        self.config = args.dataset_config["CMC"]

        # build encoders
        self.backbone = backbone(args, cmc_modality="seismic")
        self.audio_backbone = backbone(args, cmc_modality="audio")

        self.backbone_config = self.backbone.config

    def forward(self, freq_input):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            features
        """
        # compute features
        seismic_features = self.backbone(freq_input, class_head=False)
        audio_features = self.audio_backbone(freq_input, class_head=False)

        return seismic_features, audio_features
