# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class CMC(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
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
        self.backbone = backbone(args)
        self.decoder = backbone(args)
        
        self.backbone_config = self.backbone.config

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            features
        """
        # compute features
        q1 = self.backbone(x1, class_head=False)
        q2 = self.decoder(x2, class_head=False)
        
        return q1, q2
    
