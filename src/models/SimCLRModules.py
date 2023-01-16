import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLR(nn.Module):
    def __init__(self, args, backbone, out_dim=2048):
        super().__init__()
        self.args = args
        self.config = backbone.config
        self.backbone = backbone
        dim_mlp = self.config["loc_out_channels"]
        self.backbone.class_layer = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim),
        )

    def forward(self, x_i, x_j):

        # get representation
        h_i = self.backbone(x_i)
        h_j = self.backbone(x_j)

        # nonlienar MLP
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return z_i, z_j
