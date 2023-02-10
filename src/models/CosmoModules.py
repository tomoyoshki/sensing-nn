# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class Cosmo(nn.Module):
    """
    MobiCom' 22: Cosmo: contrastive fusion learning with small data for multimodal human activity recognition
    """

    def __init__(self, args, backbone):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(Cosmo, self).__init__()

        self.args = args
        self.config = args.dataset_config["CMC"]
        self.modalities = args.dataset_config["modality_names"]

        # build encoders
        self.backbone = backbone

    def forward(self, freq_input):
        """
        First extract mod features, then augment by random fusion.
        """
        # compute features
        mod_features = self.backbone(freq_input, class_head=False)
        cat_mod_features = [mod_features[mod] for mod in self.modalities]
        cat_mod_features = torch.stack(cat_mod_features, dim=1)  # [b, mod, dim]

        rand_fused_feature1 = self.rand_fusion(cat_mod_features)
        rand_fused_feature2 = self.rand_fusion(cat_mod_features)

        return rand_fused_feature1, rand_fused_feature2

    def rand_fusion(self, mod_features):
        """
        Data augmentation by random fusion.
        mod_features: [b, mod, dim]
        """
        b, mod, dim = mod_features.shape
        rand_weights = torch.rand(b, mod, 1).to(mod_features.device)
        rand_weights = rand_weights / rand_weights.sum(dim=1, keepdim=True)
        fused_features = (mod_features * rand_weights).sum(dim=1)

        return fused_features
