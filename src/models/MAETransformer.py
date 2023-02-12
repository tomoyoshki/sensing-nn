# python train.py -gpu=3 -dataset=Parkland -train_mode=supervised -model=TransformerV4
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoderLayer
from input_utils.padding_utils import get_padded_size
from models.FusionModules import TransformerFusionBlock


from einops import rearrange

from timm.models.layers import trunc_normal_

import numpy as np

import logging

from models.SwinModules import (
    BasicLayer,
    PatchEmbed,
    PatchExpanding,
    PatchMerging,
)


class MAETransformer(nn.Module):
    """
    Masked auto encoder with SWIN Transformer

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
        self.config = args.dataset_config["MAETransformer"]
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.num_segments = args.dataset_config["num_segments"]

        # Transformer Variables
        self.drop_rate = self.config["dropout_ratio"]
        self.norm_layer = nn.LayerNorm
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Single sensor
        self.freq_interval_layers = nn.ModuleDict()  # Frequency & Interval layers
        self.patch_embed = nn.ModuleDict()
        self.absolute_pos_embed = nn.ModuleDict()
        self.mod_patch_embed = nn.ModuleDict()
        self.mod_in_layers = nn.ModuleDict()

        self.encoder_norm = nn.LayerNorm(self.config["time_freq_out_channels"] * 2)
        self.mask_token = nn.ModuleDict()

        """
        MAE decoder specifics
        """

        self.patch_expand = nn.ModuleDict()
        self.decoder_blocks = nn.ModuleDict()
        self.decoder_pred = nn.ModuleDict()
        self.decoder_norm = nn.LayerNorm(self.config["decoder_channels"])

        # Patch expanding

        # masked ratio for mae
        self.masked_ratio = self.config["masked_ratio"]

        """
        MAE encoder specifics
        """
        # Patchembed
        for loc in self.locations:
            self.freq_interval_layers[loc] = nn.ModuleDict()
            self.patch_embed[loc] = nn.ModuleDict()
            self.absolute_pos_embed[loc] = nn.ParameterDict()
            self.mod_in_layers[loc] = nn.ModuleDict()

            self.patch_expand[loc] = nn.ModuleDict()
            self.mask_token[loc] = nn.ParameterDict()
            self.decoder_blocks[loc] = nn.ModuleDict()
            self.decoder_pred[loc] = nn.ModuleDict()
            for mod in self.modalities:
                # Decide the spatial size for "image"
                stride = self.config["in_stride"][mod]
                spectrum_len = args.dataset_config["loc_mod_spectrum_len"][loc][mod]
                img_size = (self.num_segments, spectrum_len // stride)

                # get the padded image size
                padded_img_size = get_padded_size(
                    img_size,
                    self.config["window_size"][mod],
                    self.config["patch_size"]["freq"][mod],
                    len(self.config["time_freq_block_num"][mod]),
                )
                logging.info(f"=\tPadded image size for {mod}: {padded_img_size}")

                """
                Encoders
                """

                # Patch embedding and Linear embedding (H, W, in_channel) -> (H / p_size, W / p_size, C)
                self.patch_embed[loc][mod] = PatchEmbed(
                    img_size=padded_img_size,
                    patch_size=self.config["patch_size"]["freq"][mod],
                    in_chans=args.dataset_config["loc_mod_in_freq_channels"][loc][mod] * stride,
                    embed_dim=self.config["time_freq_out_channels"],
                    norm_layer=self.norm_layer,
                )
                patches_resolution = self.patch_embed[loc][mod].patches_resolution
                logging.info(f"=\tPatch resolution for {mod}: {patches_resolution}")

                self.mask_token[loc][mod] = nn.Parameter(torch.zeros(1, 1, self.config["time_freq_out_channels"]))

                self.patch_expand[loc][mod] = PatchExpanding(
                    embed_dim=self.config["time_freq_out_channels"]
                    * 2 ** (len(self.config["time_freq_block_num"][mod]) - 1),
                    norm_layer=self.norm_layer,
                )

                # Absolute positional embedding (optional)
                self.absolute_pos_embed[loc][mod] = nn.Parameter(
                    torch.zeros(1, self.patch_embed[loc][mod].num_patches, self.config["time_freq_out_channels"])
                )
                trunc_normal_(self.absolute_pos_embed[loc][mod], std=0.02)

                # Swin Transformer Block
                self.freq_interval_layers[loc][mod] = nn.ModuleList()

                # Drop path rate
                dpr = [
                    x.item()
                    for x in torch.linspace(
                        0, self.config["drop_path_rate"], sum(self.config["time_freq_block_num"][mod])
                    )
                ]  # stochastic depth decay rule

                for i_layer, block_num in enumerate(
                    self.config["time_freq_block_num"][mod]
                ):  # different downsample ratios
                    down_ratio = 2**i_layer
                    layer_dim = int(self.config["time_freq_out_channels"] * down_ratio)
                    print("Down layer dim: ", layer_dim)
                    layer = BasicLayer(
                        dim=layer_dim,  # C in SWIN
                        input_resolution=(
                            patches_resolution[0] // down_ratio,  # Patch resolution = (H/4, W/4)
                            patches_resolution[1] // down_ratio,
                        ),
                        num_heads=self.config["time_freq_head_num"],
                        window_size=self.config["window_size"][mod].copy(),
                        depth=block_num,
                        drop=self.drop_rate,
                        attn_drop=self.config["attn_drop_rate"],
                        drop_path=dpr[
                            sum(self.config["time_freq_block_num"][mod][:i_layer]) : sum(
                                self.config["time_freq_block_num"][mod][: i_layer + 1]
                            )
                        ],
                        norm_layer=self.norm_layer,
                        downsample=PatchMerging
                        if (i_layer < len(self.config["time_freq_block_num"][mod]) - 1)
                        else None,
                    )
                    self.freq_interval_layers[loc][mod].append(layer)
                # Unify the input channels for each modality
                self.mod_in_layers[loc][mod] = nn.Linear(
                    (patches_resolution[0] // down_ratio) * (patches_resolution[1] // down_ratio) * layer_dim,
                    self.config["mod_out_channels"],
                )

                self.decoder_blocks[loc][mod] = nn.ModuleList()

                for i_layer, block_num in enumerate(
                    self.config["decoder_time_freq_block_num"][mod][:-1]
                ):  # different downsample ratios
                    inverse_i_layer = len(self.config["time_freq_block_num"][mod]) - i_layer - 2
                    down_ratio = 2**inverse_i_layer
                    layer_dim = int(self.config["decoder_channels"] * down_ratio)

                    print("Up level layer dim: ", layer_dim)
                    layer = BasicLayer(
                        dim=layer_dim,  # C in SWIN
                        input_resolution=(
                            patches_resolution[0] // down_ratio,  # Patch resolution = (H/4, W/4)
                            patches_resolution[1] // down_ratio,
                        ),
                        num_heads=self.config["time_freq_head_num"],
                        window_size=self.config["window_size"][mod].copy(),
                        depth=block_num,
                        drop=self.drop_rate,
                        attn_drop=self.config["attn_drop_rate"],
                        drop_path=dpr[
                            sum(self.config["time_freq_block_num"][mod][:inverse_i_layer]) : sum(
                                self.config["time_freq_block_num"][mod][: inverse_i_layer + 1]
                            )
                        ],
                        norm_layer=self.norm_layer,
                        patch_expanding=PatchExpanding
                        if (i_layer < len(self.config["time_freq_block_num"][mod]) - 2)
                        else None,
                    )
                    self.decoder_blocks[loc][mod].append(layer)
                patch_area = self.config["patch_size"]["freq"][mod][0] * self.config["patch_size"]["freq"][mod][1]
                self.decoder_pred[loc][mod] = nn.Linear(
                    self.config["decoder_channels"],
                    patch_area * args.dataset_config["loc_mod_in_freq_channels"][loc][mod],
                    bias=True,
                )  # decoder to patch

        # Sample embedding layer
        self.sample_embd_layer = nn.Sequential(
            nn.Linear(self.config["loc_out_channels"], self.config["fc_dim"]),
            nn.GELU(),
        )

        # Classification layer
        if args.train_mode == "supervised" or self.config["pretrained_head"] == "linear":
            """Linear classification layers for supervised learning or finetuning."""
            self.class_layer = nn.Sequential(
                nn.Linear(self.config["fc_dim"], args.dataset_config[args.task]["num_classes"]),
                nn.Sigmoid() if args.multi_class else nn.Softmax(dim=1),
            )
        else:
            """Non-linear classification layers for self-supervised learning."""
            self.class_layer = nn.Sequential(
                nn.Linear(self.config["fc_dim"], self.config["fc_dim"] // 2),
                nn.GELU(),
                nn.Linear(self.config["fc_dim"] // 2, args.dataset_config[args.task]["num_classes"]),
                nn.Sigmoid() if args.multi_class else nn.Softmax(dim=1),
            )

    def random_masking(self, embed_input):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        mask_ratio = self.config["masked_ratio"]

        N, L, D = embed_input.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=self.args.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        masked_embed_input = torch.gather(embed_input, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=self.args.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return masked_embed_input, mask, ids_restore

    def window_masking(
        self,
        x: torch.Tensor,
        input_resolution,
        patch_resolution,
        window_size,
        mask_token,
        remove=False,
        mask_len_sparse: bool = False,
    ):
        """
        The new masking method, masking the adjacent r*r number of patches together
        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x
        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image
        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        B, L, D = x.shape

        h, w = input_resolution[0], input_resolution[1]  # padded image h and w
        ph, pw = patch_resolution[0], patch_resolution[1]  # num patches h and w
        dh, dw = int(ph // window_size[0]), int(pw // window_size[1])  # window_resolution h and w

        rh, rw = window_size[0], window_size[1]

        print(f"input_resolution: {h}, {w}")
        print(f"patch_resolution: {ph}, {pw}")
        print(f"window_resolution: {dh}, {dw}")
        noise = torch.rand(B, (dh * dw), device=x.device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        sparse_keep = sparse_shuffle[:, : int(dh * dw * (1 - self.config["masked_ratio"]))]

        print(f"Sparse keep dimension: {sparse_keep.shape}")

        index_keep_part = torch.div(sparse_keep, dh, rounding_mode="floor") * dh * (rh * rw) + sparse_keep % dw * rw
        index_keep = index_keep_part
        for i in range(rh):
            for j in range(rw):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat([index_keep, index_keep_part + dh * i + j], dim=1)

        print(int(L - index_keep.shape[-1]))

        print(index_keep.shape)

        index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0)
        index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=np.int)
        for i in range(B):
            index_mask[i] = np.setdiff1d(index_all[i], index_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)

        if mask_len_sparse:
            mask = torch.ones([B, dh * dw], device=x.device)
            mask[:, : sparse_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=sparse_restore)
        else:
            mask = torch.ones([B, L], device=x.device)
            mask[:, : index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)

        if remove:
            x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
            # x_masked = rearrange(x_masked, "B (H W) C -> B H W C", H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask, sparse_restore
        else:
            x_masked = torch.clone(x)
            for i in range(B):
                x_masked[i, index_mask.cpu().numpy()[i, :], :] = mask_token
            # x_masked = rearrange(x_masked, "B (H W) C -> B H W C", H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask

    def patchify(self, imgs, patch_size):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """

        patch_height = patch_size[0]
        patch_width = patch_size[1]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        height = imgs.shape[2] // patch_height
        width = imgs.shape[3] // patch_width

        print(f"Image shape: {imgs.shape}")
        print(f"Height: {height}")
        print(f"Width: {width}")
        print(f"Patch height: {patch_height}")
        print(f"Patch width: {patch_width}")
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], height, patch_height, width, patch_width))
        x = torch.einsum("nchpwq->nhwpqc", x)
        print(x.shape)
        x = x.reshape(shape=(imgs.shape[0], height * width, patch_height * patch_width * imgs.shape[1]))
        return x

    def forward_loss(self, target, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """

        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.0e-6) ** 0.5

        print(f"target shape: {target.shape}")
        print(f"prediction shape: {pred.shape}")

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, freq_x, class_head=True):
        args = self.args

        total_loss = 0

        # Step 1: Feature extractions on time interval (i) and spectrum (s) domains
        loc_mod_features = dict()
        for loc in self.locations:
            loc_mod_features[loc] = []
            for mod in self.modalities:
                if mod == "seismic":
                    continue
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
                # patch_resolution = self.config["window_size"][mod]
                patch_resolution = self.patch_embed[loc][mod].patches_resolution

                # test different padding
                freq_input = F.pad(input=freq_input, pad=(0, padded_width, 0, padded_height), mode="constant", value=0)

                print(f"Original input shape: {freq_input.shape}")
                # Patch Partition and Linear Embedding
                embeded_input = self.patch_embed[loc][mod](freq_input)

                print(f"Patch embeded shape: {embeded_input.shape}")

                # Masking the input
                masked_cls_embed_input, mask = self.window_masking(
                    embeded_input,
                    padded_img_size,
                    patch_resolution,
                    self.config["window_size"][mod],
                    self.mask_token[loc][mod],
                    remove=False,
                    mask_len_sparse=False,
                )

                print(f"Masked input shape: {masked_cls_embed_input.shape}")
                # SwinTransformer Layer block
                for layer in self.freq_interval_layers[loc][mod]:
                    encoder_tokens = layer(masked_cls_embed_input)
                    masked_cls_embed_input = encoder_tokens
                    print(f"Latent variable shape: {encoder_tokens.shape}")

                encoder_tokens = self.patch_expand[loc][mod](encoder_tokens)
                print(f"Expanded token shape: {encoder_tokens.shape}")

                # SwinTransformer Layer block
                for layer in self.decoder_blocks[loc][mod]:
                    decoded_tokens = layer(encoder_tokens)
                    print(f"Decoded token shape: {decoded_tokens.shape}")
                    encoder_tokens = decoded_tokens

                decoded_tokens = self.decoder_norm(decoded_tokens)
                # decoded_toke = rearrange(decoded_tokens, "B H W C -> B (H W) C")
                # predictor projection
                decoded_tokens = self.decoder_pred[loc][mod](decoded_tokens)
                target = self.patchify(freq_input, self.config["patch_size"]["freq"][mod])
                loss = self.forward_loss(target, decoded_tokens, mask)

                total_loss += loss

        return total_loss
        # # Unify the input channels for each modality
        # freq_interval_output = self.mod_in_layers[loc][mod](freq_interval_output.reshape([b, -1]))
        # freq_interval_output = freq_interval_output.reshape(b, 1, -1)

        # # Append the result
        # loc_mod_features[loc].append(freq_interval_output)

        # Stack results from different modalities, [b, 1, s, c]
        # loc_mod_features[loc] = torch.stack(loc_mod_features[loc], dim=2)

        # Step 4: Classification Layers
        # sample_features = sample_features.flatten(start_dim=1)
        # sample_features = self.sample_embd_layer(sample_features)

        # if class_head:
        #     logits = self.class_layer(sample_features)
        #     return logits
        # else:
        #     """Self-supervised pre-training"""
        #     return sample_features
