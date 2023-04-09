import torch
import torch.nn as nn
import torch.nn.functional as F


class TNC(nn.Module):
    def __init__(self, args, backbone):
        super().__init__()
        self.args = args
        self.backbone_config = backbone.config
        self.config = args.dataset_config["TNC"]

        # components
        self.backbone = backbone

        # Discriminator
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(2 * self.input_size, 4 * self.input_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4 * self.input_size, 1),
        )

        torch.nn.init.xavier_uniform_(self.discriminator[0].weight)
        torch.nn.init.xavier_uniform_(self.discriminator[3].weight)

    def forward(self, x_1, x_2):
        # get representation
        h_1 = self.backbone(x_1, class_head=False)
        h_2 = self.backbone(x_2, class_head=False)

        # nonlienar MLP
        z_1 = self.projector(h_1)
        z_2 = self.projector(h_2)

        return z_1, z_2


class Discriminator(torch.nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.device = device
        self.input_size = input_size

        self.model = torch.nn.Sequential(
            torch.nn.Linear(2 * self.input_size, 4 * self.input_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4 * self.input_size, 1),
        )

        torch.nn.init.xavier_uniform_(self.model[0].weight)
        torch.nn.init.xavier_uniform_(self.model[3].weight)

    def forward(self, x, x_tild):
        """
        Predict the probability of the two inputs belonging to the same neighbourhood.
        """
        x_all = torch.cat([x, x_tild], -1)
        p = self.model(x_all)
        return p.view((-1,))
