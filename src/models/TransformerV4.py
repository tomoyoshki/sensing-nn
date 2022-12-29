# python3 train.py -gpu=0 -dataset=Parkland -stage=pretrain_classifier -model="TransformerV4"
import torch
import torch.nn as nn

from input_utils.fft_utils import fft_preprocess

from models.SwinModules import (
    BasicLayer,
    PatchEmbed,
    PatchMerging,
)


class TransformerV4(nn.Module):
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
        self.config = args.dataset_config["TransformerV4"]
        self.device = args.device
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.num_segments = args.dataset_config["num_segments"]

        # Transformer Variables
        self.embed_dim = self.config["embed_dim"]
        self.depths = self.config["depths"]
        self.num_heads = self.config["num_heads"]
        self.window_size = self.config["window_size"]
        self.drop_rate = self.config["dropout_ratio"]
        self.norm_layer = nn.LayerNorm
        self.num_features = int(self.embed_dim * 2 ** (len(self.depths) - 1))
        self.norm = self.norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Layers
        self.freq_interval_layers = nn.ModuleDict()  # Frequency & Interval layers
        self.sensor_layers = nn.ModuleDict()  # Modality Layers

        # split image into non-overlapping patches (? linear embedding)
        self.patch_embed = nn.ModuleDict()
        self.mod_patch_embed = nn.ModuleDict()
        for loc in self.locations:
            self.freq_interval_layers[loc] = nn.ModuleDict()
            self.patch_embed[loc] = nn.ModuleDict()
            for mod in self.modalities:
                stride = self.config["in_stride"][mod]
                # Patch embedding and Linear embedding (H, W, in_channel) -> (H / p_size, W / p_size, C)
                self.patch_embed[loc][mod] = PatchEmbed(
                    img_size=(10, 20),
                    patch_size=self.config["patch_size"]["freq"][loc][mod],
                    in_chans=args.dataset_config["loc_mod_in_freq_channels"][loc][mod] * stride,
                    embed_dim=self.embed_dim,
                    norm_layer=self.norm_layer,
                )
                patches_resolution = self.patch_embed[loc][mod].patches_resolution

                # 1 layers Transformer Blocks
                self.freq_interval_layers[loc][mod] = nn.ModuleList()
                for i_layer in range(len(self.depths)):
                    # Swin Transformer Block
                    layer = BasicLayer(
                        dim=int(self.embed_dim * 2**i_layer),  # C in SWIN
                        input_resolution=(
                            patches_resolution[0] // (2**i_layer),  # Patch resolution = (H/4, W/4)
                            patches_resolution[1] // (2**i_layer),
                        ),
                        num_heads=self.num_heads[i_layer],
                        window_size=self.window_size,
                        depth=self.depths[i_layer],
                        drop=self.drop_rate,
                        norm_layer=self.norm_layer,
                        downsample=PatchMerging
                        if (i_layer < len(self.depths) - 1)
                        else None,  # Patch merging before transformer blocks
                    )
                    self.freq_interval_layers[loc][mod].append(layer)

            self.mod_patch_embed[loc] = PatchEmbed(
                img_size=(50, 2),
                patch_size=self.config["patch_size"]["mod"][loc][mod],
                in_chans=self.embed_dim,
                embed_dim=self.embed_dim,
                norm_layer=self.norm_layer,
            )
            patches_resolution = self.mod_patch_embed[loc].patches_resolution
            # Two layers Transformer Blocks
            self.sensor_layers[loc] = nn.ModuleList()
            for i_layer in range(len(self.depths)):
                # Swin Transformer Block
                layer = BasicLayer(
                    dim=int(self.embed_dim * 2**i_layer),  # C in SWIN
                    input_resolution=(
                        patches_resolution[0] // (2**i_layer),  # Patch resolution = (H/4, W/4)
                        patches_resolution[1] // (2**i_layer),
                    ),
                    num_heads=self.num_heads[i_layer],
                    window_size=self.window_size,
                    depth=self.depths[i_layer],
                    drop=self.drop_rate,
                    norm_layer=self.norm_layer,
                    downsample=PatchMerging
                    if (i_layer < len(self.depths) - 1)
                    else None,  # Patch merging before transformer blocks
                )
                self.sensor_layers[loc].append(layer)

        self.class_layer = nn.Sequential(
            nn.Linear(self.embed_dim, self.config["fc_dim"]),
            nn.GELU(),
            nn.Linear(self.config["fc_dim"], args.dataset_config["num_classes"]),
            nn.Sigmoid() if args.multi_class else nn.Softmax(dim=1),
        )

    def forward(self, org_time_x, augmenter):
        args = self.args
        # Step 0: Move data to target device
        for loc in org_time_x:
            for mod in org_time_x[loc]:
                org_time_x[loc][mod] = org_time_x[loc][mod].to(self.device)

        # Step 1 Optional data augmentation
        augmented_time_x = augmenter.augment_forward(org_time_x)

        # Step 2: FFT on the time domain data; c: 1 -> 2
        freq_x = fft_preprocess(augmented_time_x, args)

        # Step 3: Feature extractions on time interval (i) and spectrum (s) domains
        loc_mod_features = dict()
        for loc in self.locations:
            loc_mod_features[loc] = []
            for mod in self.modalities:
                stride = self.config["in_stride"][mod]

                # Frequency spectrum feature extraction
                freq_input = freq_x[loc][mod]

                # [b, c, i, spectrum] -- > [b, i, spectrum, c]
                freq_input = torch.permute(freq_x[loc][mod], [0, 2, 3, 1])
                b, i, s, c = freq_input.shape

                # Forces both audio and seismic to have the same "img" size
                freq_input = torch.reshape(freq_input, (b, i, s // stride, c * stride))

                # Repermute back to [b, c, i, spectrum] required in PatchEmbed (b, h, w, c)
                freq_input = torch.permute(freq_input, [0, 3, 1, 2])

                # Patch Partition and Linear Embedding
                embeded_input = self.patch_embed[loc][mod](freq_input)

                # SwinTransformer Layer block
                for layer in self.freq_interval_layers[loc][mod]:
                    freq_interval_output = layer(embeded_input)
                    embeded_input = freq_interval_output

                # Trying to increase depths
                for layer in self.freq_interval_layers[loc][mod]:
                    freq_interval_output = layer(embeded_input)
                    embeded_input = freq_interval_output

                # Append the result
                loc_mod_features[loc].append(freq_interval_output)
            # Stack results from different modalities
            loc_mod_features[loc] = torch.stack(loc_mod_features[loc], dim=2)

        # Step 4: Feature extractions on the modality/sensors
        loc_features = []
        for loc in loc_mod_features:
            # Repermute [b, L (H * W), mod, c] -> [b, c, L, mod] for Patch Embedding
            mod_input = torch.permute(loc_mod_features[loc], [0, 3, 1, 2])

            # Patch Partition and Linear Embedding on L (time & spectrum) and Modality domains -> [B, M, C]
            embeded_mod_input = self.mod_patch_embed[loc](mod_input)

            # SwinTransformer Blocks -> [B, M, C]
            for layer in self.sensor_layers[loc]:
                mod_output = layer(embeded_mod_input)
                embeded_mod_input = mod_output

            # Normalize and append to result
            mod_output = self.norm(mod_output)
            mod_output = self.avgpool(mod_output.transpose(1, 2))
            loc_features.append(mod_output)

        # Stack results from different location
        loc_features = torch.stack(loc_features, dim=2)

        # Classification Layers
        sample_features = loc_features.squeeze(dim=2)
        sample_features = sample_features.flatten(start_dim=1)
        logit = self.class_layer(sample_features)

        return logit
