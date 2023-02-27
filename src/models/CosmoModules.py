# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.backbone_config = args.dataset_config[args.model]
        self.config = args.dataset_config["Cosmo"]
        self.modalities = args.dataset_config["modality_names"]
        self.in_fc_dim = (
            self.backbone_config["recurrent_dim"] * 2
            if args.model == "DeepSense"
            else self.backbone_config["loc_out_channels"]
        )
        self.out_fc_dim = self.config["emb_dim"]
        self.num_positive = self.config["num_positive"]

        # build encoders
        self.backbone = backbone

        # build projection head
        self.mod_proj_head = nn.ModuleDict()
        for mod in self.modalities:
            self.mod_proj_head[mod] = nn.Sequential(
                nn.Linear(self.in_fc_dim, self.out_fc_dim),
                nn.ReLU(),
                nn.Linear(self.out_fc_dim, self.out_fc_dim),
            )

    def forward(self, freq_input):
        """
        First extract mod features, then augment by random fusion.
        """
        # compute features
        in_mod_features = self.backbone(freq_input, class_head=False)

        # project mod features
        proj_mod_features = []
        for mod in self.modalities:
            mod_out = F.normalize(self.mod_proj_head[mod](in_mod_features[mod]), dim=1)
            proj_mod_features.append(mod_out)

        # random fusion
        fused_feature = self.cosmo_rand_fusion(proj_mod_features)

        return fused_feature

    def cosmo_rand_fusion(self, mod_features):
        """
        Random fusion function from Cosmo paper. Assume only two modalities are used here.
        Ref: https://github.com/xmouyang/Cosmo/blob/6e64da45049f074853026e522d826c3803526e06/sample-code-UTD/Cosmo/cosmo_design.py#L8
        """
        mod1_feature = mod_features[0].unsqueeze(1).tile(1, self.num_positive, 1)  # [b, num_positive, dim]
        mod2_feature = mod_features[1].unsqueeze(1).tile(1, self.num_positive, 1)  # [b, num_positive, dim]
        b, _, dim = mod1_feature.shape

        fusion_weight = torch.arange(1, self.num_positive + 1) / (self.num_positive + 1)  # [num_positive]
        fusion_weight = fusion_weight.to(self.args.device)
        fusion_weight = fusion_weight.unsqueeze(0).unsqueeze(2).tile([b, 1, dim])  # [b, num_positive, dim]

        fused_feature = mod1_feature * fusion_weight + mod2_feature * (1 - fusion_weight)  # [b, num_positive, dim]

        return fused_feature

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
