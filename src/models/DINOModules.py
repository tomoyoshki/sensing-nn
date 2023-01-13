import torch
import torch.nn as nn


class DINOWrapper(nn.Module):
    def __init__(self, backbone, new_head, args):
        super().__init__()
        backbone.head = nn.Identity()  # deactivate original head
        self.backbone = backbone
        self.new_head = new_head
        self.args = args

    def forward(self, x, isTeacher=False):
        """Run the forward pass.
        TODO: need to implement
        Parameters
        ----------
        x : list
            List of `torch.Tensor` each of shape `(n_samples, 3, size, size)`.
        Returns
        -------
        tuple
            Tuple of `torch.Tensor` each of shape `(n_samples, out_dim)` where
            `output_dim` is determined by `Head`.
        """
        x_ = x.copy()
        if isTeacher:
            for loc in self.args["location_names"]:
                for mod in self.args["modality_names"]:
                    x_[loc][mod] = x_[loc][mod][:2]
        cls_embedding = self.backbone(x_)
        logits = self.new_head(cls_embedding)

        return logits


class DINOHead(nn.Module):
    """Network hooked up to the CLS token embedding.
    Just a MLP with the last layer being normalized in a particular way.
    Parameters
    ----------
    in_dim : int
        The dimensionality of the token embedding.
    out_dim : int
        The dimensionality of the final layer (we compute the softmax over).
    hidden_dim : int
        Dimensionality of the hidden layers.
    bottleneck_dim : int
        Dimensionality of the second last layer.
    n_layers : int
        The number of layers.
    norm_last_layer : bool
        If True, then we freeze the norm of the weight of the last linear layer
        to 1.
    Attributes
    ----------
    mlp : nn.Sequential
        Vanilla multi-layer perceptron.
    last_layer : nn.Linear
        Reparametrized linear layer with weight normalization. That means
        that that it will have `weight_g` and `weight_v` as learnable
        parameters instead of a single `weight`.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim=512,
        bottleneck_dim=256,
        n_layers=3,
        norm_last_layer=False,
    ):
        super().__init__()
        if n_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        """Initialize learnable parameters."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Of shape `(n_samples, in_dim)`.
        Returns
        -------
        torch.Tensor
            Of shape `(n_samples, out_dim)`.
        """
        x = self.mlp(x)  # (n_samples, bottleneck_dim)
        x = nn.functional.normalize(x, dim=-1, p=2)  # (n_samples, bottleneck_dim)
        x = self.last_layer(x)  # (n_samples, out_dim)

        return x
