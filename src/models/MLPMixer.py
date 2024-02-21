# python train.py -gpu=3 -dataset=Parkland -train_mode=supervised -model=TransformerV4
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from input_utils.padding_utils import get_padded_size

import logging

from models.SwinModules import (
    PatchEmbed,
)

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.05):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            MLP(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            MLP(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x

class MLPMixer(nn.Module):
    """
    SWIN Transformer model

    Parameters
    ----------

    Returns
    -------
    int or float
        The square of `x`
    """

    def __init__(self, args) -> None:
        """
        SWIN Transformer model constructor

        Parameters
        ----------
        args:
            list of configuration
        """
        super().__init__()
        self.args = args
        self.config = args.dataset_config["MLPMixer"]
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.num_segments = args.dataset_config["num_segments"]
        self.norm_layer = nn.LayerNorm
        self.layer_norm = nn.LayerNorm(self.config["time_freq_out_channels"])
        self.mod_layer_norm = nn.LayerNorm(self.config["time_freq_out_channels"] * len(self.modalities))

        self.patch_embed = nn.ModuleDict()
        self.mixer_blocks = nn.ModuleDict()
        

        for loc in self.locations:
            self.patch_embed[loc] = nn.ModuleDict()
            self.mixer_blocks[loc] = nn.ModuleDict()
            for mod in self.modalities:
                # Decide the spatial size for "image"
                stride = self.config["in_stride"][mod]
                spectrum_len = args.dataset_config["loc_mod_spectrum_len"][loc][mod]
                img_size = (self.num_segments, spectrum_len // stride)

                # get the padded image size
                padded_img_size = get_padded_size(
                    img_size,
                    [1, 1],
                    self.config["patch_size"]["freq"][mod],
                    1,
                )
                logging.info(f"=\tPadded image size for {mod}: {padded_img_size}")

                # Patch embedding and Linear embedding (H, W, in_channel) -> (H / p_size, W / p_size, C)
                self.patch_embed[loc][mod] = PatchEmbed(
                    img_size=padded_img_size,
                    patch_size=self.config["patch_size"]["freq"][mod],
                    in_chans=args.dataset_config["loc_mod_in_freq_channels"][loc][mod] * stride,
                    embed_dim=self.config["time_freq_out_channels"],
                    norm_layer=self.norm_layer,
                )
                
                self.mixer_blocks[loc][mod] = nn.ModuleList([])

                for _ in range(self.config["depth"]):
                    self.mixer_blocks[loc][mod].append(MixerBlock(self.config["time_freq_out_channels"], self.patch_embed[loc][mod].num_patches, self.config["token_dim"], self.config["channel_dim"]))

        # Sample embedding layer
        self.sample_embd_layer = nn.Sequential(
            nn.Linear(self.config["time_freq_out_channels"] * len(self.modalities), self.config["fc_dim"]),
            nn.GELU(),
        )
        # Classification layer
        self.class_layer = nn.Sequential(
            nn.Linear(self.config["fc_dim"], args.dataset_config[args.task]["num_classes"]),
            # nn.Sigmoid() if args.multi_class else nn.Softmax(dim=1),
        )
        
    def forward(self, freq_x, class_head=True):
        args = self.args

        # Step 1: Feature extractions on time interval (i) and spectrum (s) domains
        mod_features = []
        for loc in self.locations:
            for mod in self.modalities:
                stride = self.config["in_stride"][mod]
                spectrum_len = args.dataset_config["loc_mod_spectrum_len"][loc][mod]
                img_size = (self.num_segments, spectrum_len // stride)
                freq_input = freq_x[loc][mod]

                # [b, c, i, spectrum] -- > [b, i, spectrum, c]
                freq_input = torch.permute(freq_input, [0, 2, 3, 1])
                b, i, s, c = freq_input.shape

                # Forces both audio and seismic to have the same "img" size
                freq_input = torch.reshape(freq_input, (b, i, s // stride, c * stride))

                # Repermute back to [b, c, i, spectrum], (b, c, h, w) required in PatchEmbed
                freq_input = torch.permute(freq_input, [0, 3, 1, 2])

                # Pad [i, spectrum] to the required padding size
                padded_img_size = self.patch_embed[loc][mod].img_size
                padded_height = padded_img_size[0] - img_size[0]
                padded_width = padded_img_size[1] - img_size[1]

                # test different padding
                freq_input = F.pad(input=freq_input, pad=(0, padded_width, 0, padded_height), mode="constant", value=0)

                # Patch Partition and Linear Embedding
                embeded_input = self.patch_embed[loc][mod](freq_input)
                
                for mblock in self.mixer_blocks[loc][mod]:
                    embeded_input = mblock(embeded_input)
                
                embeded_input = self.layer_norm(embeded_input)
                embeded_input = embeded_input.mean(dim=1)
                mod_features.append(embeded_input)
        
        mod_features = torch.cat(mod_features, dim=1)
        mod_features = self.mod_layer_norm(mod_features)
        logits = self.sample_embd_layer(mod_features)
        logits = self.class_layer(logits)
        return logits
            
