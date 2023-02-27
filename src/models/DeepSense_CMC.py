import os
import time
import torch
import torch.nn as nn

from models.ConvModules import ConvBlock, DeConvBlock
from models.FusionModules import MeanFusionBlock, SelfAttentionFusionBlock
from models.RecurrentModule import RecurrentBlock, DecRecurrentBlock
from input_utils.normalize import normalize_input
from input_utils.padding_utils import get_padded_size
from models.FusionModules import TransformerFusionBlock


from models.SwinModules import PatchEmbed
from models.MAEModule import get_2d_sincos_pos_embed


class DeepSense_CMC(nn.Module):
    def __init__(self, args, self_attention=False) -> None:
        """The initialization for the DeepSense class.
        Design: Single (interval, loc, mod) feature -->
                Single (interval, loc) feature -->
                Single interval feature -->
                GRU -->
                Logits
        Args:
            num_classes (_type_): _description_
        """
        super().__init__()
        self.args = args
        self.self_attention = self_attention
        self.config = args.dataset_config["DeepSense"]
        self.device = args.device
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.multi_location_flag = len(self.locations) > 1

        """define the architecture"""
        self.init_encoder(args)
        if args.train_mode == "MAE":
            self.init_patch_embedding(args)
            self.init_feature_encoding(args)
            self.init_decoder(args)

    def init_encoder(self, args):
        # Step 1: Single (loc, mod) feature
        self.loc_mod_extractors = nn.ModuleDict()
        for loc in self.locations:
            self.loc_mod_extractors[loc] = nn.ModuleDict()
            for mod in self.modalities:
                if type(self.config["loc_mod_conv_lens"]) is dict:
                    """for acoustic processing in Parkland data"""
                    conv_lens = self.config["loc_mod_conv_lens"][mod]
                    in_stride = self.config["loc_mod_in_conv_stride"][mod]
                else:
                    conv_lens = self.config["loc_mod_conv_lens"]
                    in_stride = 1

                # define the extractor
                self.loc_mod_extractors[loc][mod] = ConvBlock(
                    in_channels=args.dataset_config["loc_mod_in_freq_channels"][loc][mod],
                    out_channels=self.config["loc_mod_out_channels"],
                    in_spectrum_len=args.dataset_config["loc_mod_spectrum_len"][loc][mod],
                    conv_lens=conv_lens,
                    dropout_ratio=self.config["dropout_ratio"],
                    num_inter_layers=self.config["loc_mod_conv_inter_layers"],
                    in_stride=in_stride,
                )

        # Step 3: Loc fusion
        self.loc_fusion_layers = nn.ModuleDict()
        self.mod_extractors = nn.ModuleDict()
        for mod in self.modalities:
            if self.self_attention:
                self.loc_fusion_layers[mod] = SelfAttentionFusionBlock()
            else:
                self.loc_fusion_layers[mod] = MeanFusionBlock()

            self.mod_extractors[mod] = ConvBlock(
                in_channels=1,
                out_channels=self.config["loc_out_channels"],
                in_spectrum_len=self.config["loc_mod_out_channels"],
                conv_lens=self.config["loc_conv_lens"],
                dropout_ratio=self.config["dropout_ratio"],
                num_inter_layers=self.config["loc_conv_inter_layers"],
            )

        # Step 5: GRU
        self.recurrent_layers = nn.ModuleDict()
        for mod in self.modalities:
            self.recurrent_layers[mod] = RecurrentBlock(
                in_channel=self.config["loc_out_channels"],
                out_channel=self.config["recurrent_dim"],
                num_layers=self.config["recurrent_layers"],
                dropout_ratio=self.config["dropout_ratio"],
            )

        # mod fusion layer
        if args.contrastive_framework == "Cosmo":
            "Attention fusion for Cosmo"
            self.mod_fusion_layer = TransformerFusionBlock(
                self.config["recurrent_dim"] * 2,
                4,
                self.config["dropout_ratio"],
                self.config["dropout_ratio"],
            )
            self.sample_dim = self.config["recurrent_dim"] * 2
        else:
            self.sample_dim = self.config["recurrent_dim"] * 2 * len(self.modalities)

        # Classification layer
        if args.train_mode == "supervised" or self.config["pretrained_head"] == "linear":
            """Linear classification layers for supervised learning or finetuning."""
            self.class_layer = nn.Sequential(
                nn.Linear(self.sample_dim, args.dataset_config[args.task]["num_classes"]),
                # nn.Sigmoid() if args.multi_class else nn.Softmax(dim=1),
            )
        else:
            """Non-linear classification layers for self-supervised learning."""
            self.class_layer = nn.Sequential(
                nn.Linear(self.sample_dim, self.config["fc_dim"]),
                nn.GELU(),
                nn.Linear(self.config["fc_dim"], args.dataset_config[args.task]["num_classes"]),
                # nn.Sigmoid() if args.multi_class else nn.Softmax(dim=1),
            )

    def init_patch_embedding(self, args):
        self.patch_embed = nn.ModuleDict()
        self.pos_embed = nn.ModuleDict()
        self.decoder_pos_embed = nn.ModuleDict()
        self.mask_token = nn.ModuleDict()

        for loc in self.locations:
            self.patch_embed[loc] = nn.ModuleDict()
            self.pos_embed[loc] = nn.ParameterDict()
            self.decoder_pos_embed[loc] = nn.ParameterDict()
            self.mask_token[loc] = nn.ParameterDict()
            for mod in self.modalities:
                # get the padded image size
                stride = self.config["in_stride"][mod]
                spectrum_len = self.args.dataset_config["loc_mod_spectrum_len"][loc][mod]

                img_size = (self.num_segments, spectrum_len // stride)
                padded_img_size = get_padded_size(
                    img_size,
                    self.config["window_size"][mod],
                    self.config["patch_size"]["freq"][mod],
                    len(self.config["time_freq_block_num"][mod]),
                )
                self.patch_embed[loc][mod] = PatchEmbed(
                    img_size=padded_img_size,
                    patch_size=self.config["patch_size"]["freq"][mod],
                    in_chans=self.args.dataset_config["loc_mod_in_freq_channels"][loc][mod] * stride,
                    embed_dim=self.config["loc_mod_out_channels"],
                    norm_layer=nn.LayerNorm,
                )

                num_patches = self.patch_embed[loc][mod].num_patches
                self.pos_embed[loc][mod] = nn.Parameter(
                    torch.zeros(1, num_patches + 1, self.config["loc_mod_out_channels"]), requires_grad=False
                )  # fixed sin-cos embedding

                self.decoder_pos_embed = nn.Parameter(
                    torch.zeros(1, num_patches + 1, self.config["loc_mod_out_channels"]), requires_grad=False
                )  # fixed sin-cos embedding

                self.mask_token[loc][mod] = nn.Parameter(torch.zeros(1, 1, self.config["loc_mod_out_channels"]))
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        for loc in self.locations:
            for mod in self.modalities:
                pos_embed = get_2d_sincos_pos_embed(
                    self.pos_embed[loc][mod].shape[-1], int(self.patch_embed[loc][mod].num_patches), cls_token=True
                )
                self.pos_embed[loc][mod].data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

                decoder_pos_embed = get_2d_sincos_pos_embed(
                    self.decoder_pos_embed[loc][mod].shape[-1],
                    int(self.patch_embed[loc][mod].num_patches),
                    cls_token=True,
                )
                self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

                # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
                w = self.patch_embed[loc][mod].proj.weight.data
                torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

                # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
                torch.nn.init.normal_(self.cls_token[loc][mod], std=0.02)
                torch.nn.init.normal_(self.mask_token[loc][mod], std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def init_feature_encoding(self, args):
        self.encoded_features_layer = nn.Sequential(
            nn.Linear(self.sample_dim, self.config["fc_dim"]),
            nn.GELU(),
            nn.Linear(self.config["fc_dim"], self.config["fc_dim"]),
        )

        # [fc_dim layer] -> [mod1_feature, mod2_feature, mod3_feature]
        self.encoded_mod_extract_layer = nn.ModuleDict()
        for loc in self.locations:
            self.encoded_mod_extract_layer[loc] = nn.ModuleDict()
            for mod in self.modalities:
                self.encoded_mod_extract_layer[loc][mod] = nn.Sequential(
                    nn.Linear(self.config["fc_dim"], self.config["fc_dim"]),
                    nn.GELU(),
                    nn.Linear(self.config["fc_dim"], self.config["loc_out_channels"]),
                )

    def init_decoder(self, args):
        # step 1: GRU decoder
        self.dec_recurrent_layers = nn.ModuleDict()
        for mod in self.modalities:
            self.dec_recurrent_layers[mod] = DecRecurrentBlock(
                mod_interval=self.args.dataset_config["num_segments"],
                in_channel=self.config["loc_out_channels"],
                out_channel=self.config["recurrent_dim"],
                num_layers=self.config["recurrent_layers"],
                dropout_ratio=self.config["dropout_ratio"],
            )
        # step 2: Loc fusion
        self.dec_loc_fusion_layers = nn.ModuleDict()
        self.dec_mod_extractors = nn.ModuleDict()
        for mod in self.modalities:
            if self.self_attention:
                self.dec_loc_fusion_layers[mod] = SelfAttentionFusionBlock()
            else:
                self.dec_loc_fusion_layers[mod] = MeanFusionBlock()

            self.dec_mod_extractors[mod] = DeConvBlock(
                num_segments=self.args.dataset_config["num_segments"],
                in_channels=1,
                out_channels=self.config["loc_out_channels"],
                in_spectrum_len=self.config["loc_mod_out_channels"],
                conv_lens=self.config["loc_conv_lens"],
                dropout_ratio=self.config["dropout_ratio"],
                num_inter_layers=self.config["loc_conv_inter_layers"],
            )
        # step 3: Single (loc, mod) feature decoder - DeConv Blocks
        # Step 1: Single (loc, mod) feature
        self.dec_loc_mod_extractors = nn.ModuleDict()
        for loc in self.locations:
            self.dec_loc_mod_extractors[loc] = nn.ModuleDict()
            for mod in self.modalities:
                if type(self.config["loc_mod_conv_lens"]) is dict:
                    """for acoustic processing in Parkland data"""
                    conv_lens = self.config["loc_mod_conv_lens"][mod]
                    in_stride = self.config["loc_mod_in_conv_stride"][mod]
                else:
                    conv_lens = self.config["loc_mod_conv_lens"]
                    in_stride = 1

                # define the extractor
                self.dec_loc_mod_extractors[loc][mod] = DeConvBlock(
                    num_segments=self.args.dataset_config["num_segments"],
                    in_channels=args.dataset_config["loc_mod_in_freq_channels"][loc][mod],
                    out_channels=self.config["loc_mod_out_channels"],
                    in_spectrum_len=args.dataset_config["loc_mod_spectrum_len"][loc][mod],
                    conv_lens=conv_lens,
                    dropout_ratio=self.config["dropout_ratio"],
                    num_inter_layers=self.config["loc_mod_conv_inter_layers"],
                    in_stride=in_stride,
                )
        pass

    def mask_input(self, freq_x):
        for loc in self.locations:
            for mod in self.modalities:
                embeded_x = self.patch_embed[loc][mod](freq_x[loc][mod])
                embeded_x = embeded_x + self.pos_embed[:, 1:, :]

    def forward_encoder(self, freq_x, class_head=True):
        """The encoder function of DeepSense.
        Args:
            time_x (_type_): time_x is a dictionary consisting of the Tensor input of each input modality.
                        For each modality, the data is in (b, c (2 * 3 or 1), i (intervals), s (spectrum)) format.
        """
        masked_x, mask, ids_restore = self.mask_input()
        # Step 1: Single (loc, mod) feature extraction, (b, c, i)
        loc_mod_features = {mod: [] for mod in self.modalities}
        for loc in self.locations:
            for mod in self.modalities:
                loc_mod_feature = self.loc_mod_extractors[loc][mod](freq_x[loc][mod])
                loc_mod_features[mod].append(loc_mod_feature)

        for mod in loc_mod_features:
            loc_mod_features[mod] = torch.stack(loc_mod_features[mod], dim=3)

        # Step 2: Location fusion, (b, c, i)
        mod_interval_features = {mod: [] for mod in self.modalities}
        for mod in self.modalities:
            if not self.multi_location_flag:
                mod_interval_features[mod] = loc_mod_features[mod].squeeze(3)
            else:
                fused_mod_feature = self.loc_fusion_layers[mod](loc_mod_features[mod])
                extracted_mod_features = self.mod_extractors[mod](fused_mod_feature)
                mod_interval_features[mod] = extracted_mod_features

        # Step 3: Interval Fusion for each modality, [b, c, i]
        mod_features = []
        hidden_features = []
        for mod in self.modalities:
            mod_feature, hidden_feature = self.recurrent_layers[mod](mod_interval_features[mod])
            mod_features.append(mod_feature.flatten(start_dim=1))
            hidden_features.append(hidden_feature)

        # Step 4: Mod concatenation, [b, 1, mod, c]
        if not class_head:
            return dict(zip(self.modalities, mod_features)), dict(zip(self.modalities, hidden_features))
        else:
            if self.args.contrastive_framework == "Cosmo":
                """Attention-based Fusion"""
                mod_features = torch.stack(mod_features, dim=1)
                mod_features = mod_features.unsqueeze(dim=1)
                sample_features = self.mod_fusion_layer(mod_features).flatten(start_dim=1)
            else:
                """Concatenation-based Fusion"""
                sample_features = torch.cat(mod_features, dim=1)

            logits = self.class_layer(sample_features)
            return logits

    def forward_feature_encoder(self, mod_features):
        """Encode the features by merging and then demerge

        Args:
            mod_features (dict): mod_features[mod] = features extracted for mod
        """
        mod_features = [mod_features[mod] for mod in mod_features]

        # fusion basede on attention or concatnation
        if self.config["fusion"] == "attention":
            """Attention-based fusion."""
            mod_features = torch.stack(mod_features, dim=1)
            mod_features = mod_features.unsqueeze(dim=1)
            sample_features = self.mod_fusion_layer(mod_features).flatten(start_dim=1)
        else:
            """Concatenation-based fusion."""
            sample_features = torch.cat(mod_features, dim=1)

        # fully connected layer
        encoded_sample_features = self.encoded_features_layer(sample_features)

        # extract feature for each modality
        encoded_mod_features = []
        for loc in self.locations:
            for mod in self.modalities:
                encoded_mod_feature = self.encoded_mod_extract_layer[loc][mod](encoded_sample_features)
                encoded_mod_features.append(encoded_mod_feature)

        return dict(zip(self.modalities, mod_features))

    def forward_decoder(self, mod_features, hidden_features):
        print("Forward decoder")
        # Step 1: Interval Fusion Decoder for each modality, [b, c, i]
        dec_mod_features = {}
        for mod in self.modalities:
            dec_mod_feature = self.dec_recurrent_layers[mod](mod_features[mod], hidden_features[mod])
            dec_mod_features[mod] = dec_mod_feature

        # Step 2: Location fusion decoder, (b, c, i)
        dec_mod_interval_features = {}
        for mod in self.modalities:
            if not self.multi_location_flag:
                dec_mod_interval_features[mod] = dec_mod_features[mod].unsqueeze(3)
            else:
                # TODO: Test
                fused_mod_feature = self.dec_loc_fusion_layers[mod](dec_mod_features[mod])
                extracted_mod_features = self.dec_mod_extractors[mod](fused_mod_feature)
                dec_mod_interval_features[mod] = extracted_mod_features

        for mod in self.modalities:
            # TODO: Stack -> UnStack
            dec_mod_interval_features[mod] = dec_mod_interval_features[mod].squeeze(3)

        # Step 1: Single (loc, mod) feature extraction, (b, c, i)
        dec_loc_mod_input = {}
        for loc in self.locations:
            dec_loc_mod_input[loc] = {}
            for mod in self.modalities:
                decoded_input = self.dec_loc_mod_extractors[loc][mod](dec_mod_interval_features[mod])
                dec_loc_mod_input[loc][mod] = decoded_input

        return dec_loc_mod_input

    def forward(self, freq_x, class_head=True):
        mod_features, hidden_features = self.forward_encoder(freq_x, class_head)
        if self.args.train_mode != "MAE" or class_head:
            return mod_features
        encoded_mod_features = self.forward_feature_encoder(mod_features)
        decoded_output = self.forward_decoder(encoded_mod_features, hidden_features)
        # return decoded_output
