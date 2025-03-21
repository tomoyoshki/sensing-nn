from matplotlib.pyplot import axis
import torch
import torch.nn as nn
from general_utils.tensor_utils import select_with_rescale_factors_bcis, select_with_rescale_factors_bisc


class MeanFusionBlock(nn.Module):
    def __init__(self) -> None:
        '''The initialization of the mean fusion block.
        Stureture:
        '''
        super().__init__()

    def forward(self, input_feature, rescale_factors=None):
        '''The forward function of the MeanFusionBlock.

        Args:
            input_feature (_type_): list of [b, c, intervals, sensors]
            rescale_factors: [b, sensors]
        Return:
            [b, 1, i, c]
        NOTE: The batch dimension should be 1, or the samples must have the same # of missing sensors.
        '''
        # mean out
        if rescale_factors is None:
            # if True:
            # breakpoint()
            batch_mean_out = torch.mean(input_feature, dim=3, keepdim=False)
        else:
            b, c, i, s = input_feature.shape
            expand_recale_factors = rescale_factors.reshape([b, 1, 1, s]).tile([1, c, i, 1])

            # split features, rescale factors by sample
            split_expand_scale_factors = torch.split(expand_recale_factors, 1, dim=0)
            split_rescale_factors = torch.split(rescale_factors, 1, dim=0)
            split_input_features = torch.split(input_feature, 1, dim=0)

            # calculate the merged features sequentially for each sample
            batch_mean_out = []
            for idx in range(b):
                '''feature: [1, c , i, s], rescale_factors: [1, s]'''
                selected_rescale_factors = select_with_rescale_factors_bcis(
                    split_expand_scale_factors[idx],
                    split_rescale_factors[idx],
                )
                selected_input_feature = select_with_rescale_factors_bcis(
                    split_input_features[idx],
                    split_rescale_factors[idx],
                )
                norm_rescale_factors = torch.softmax(selected_rescale_factors, dim=3)
                batch_mean_out.append(torch.sum(selected_input_feature * norm_rescale_factors, dim=3, keepdim=False))

            # merge
            batch_mean_out = torch.cat(batch_mean_out, dim=0)

        # flatten and move c to spectral samples, [b, c, i] --> [b, 1, i, s]
        batch_mean_out = batch_mean_out.permute(0, 2, 1)
        batch_mean_out = torch.unsqueeze(batch_mean_out, dim=1)

        return batch_mean_out


class SelfAttentionFusionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0, bias=True) -> None:
        '''The initialization of the self-attention fusion block.
        Structure:
        '''
        super().__init__()

        # define the self attention head
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias, batch_first=True)

    def forward(self, x):
        '''The forward function of the SelfAttentionBlock.

        Args:
            x (_type_): list of [b, c, intervals, sensors]
        Output:
            [b, c, i] or [b, 1, i, c]
        '''
        # concat and exchange dimension: [b, c, i, s] --> [b, i, s, c] --> [b * i, s, c]
        x = x.permute(0, 2, 3, 1)
        b, i, s, c = x.shape
        x = torch.reshape(x, (b * i, s, c))

        # Step 1: Calculate the mean query, shape: [b * i, 1 (one query only), c]
        mean_query = torch.mean(x, dim=1, keepdim=True)

        # Step 2: Attention, attention out: [b, c, i]
        attn_out, attn_weights = self.attention(mean_query, x, x, need_weights=True)
        attn_out = attn_out.reshape(attn_out, (b, i, 1, c))
        attn_out = torch.squeeze(attn_out, dim=2)
        attn_out = torch.reshape(attn_out, (b, i, c))
        attn_out = attn_out.permute(0, 2, 1)

        # flatten and move c to spectral samples, [b, c, i] --> [b, 1, i, s (=c)]
        attn_out = attn_out.permute(0, 2, 1)
        attn_out = torch.unsqueeze(attn_out, dim=1)

        return attn_out


class TransformerFusionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate, attention_dropout_rate):
        ''' Normalization + Attention + Dropout

        Args:
            embed_dim (_type_): _description_
            num_heads (_type_): _description_
            dropout_rate (_type_): _description_
            attention_dropout_rate (_type_): _description_
        '''
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=attention_dropout_rate, batch_first=True)

    def forward(self, input_feature, rescale_factors=None):
        '''Sensor fusion by attention

        Args:
            inputs (_type_): [batch_size, time_interval, mod_num/loc_num/interval, feature_channels]
            rescale_factors: [b, sensors]

        Returns:
            _type_: [batch_size, time_interval, feature_channels]
        NOTE: The batch dimension should be 1, or the samples must have the same # of missing sensors.
        '''
        if rescale_factors is not None:
            # hard selection on modality
            b, i, s, c = input_feature.shape
            expand_recale_factors = rescale_factors.reshape([b, 1, s, 1]).tile([1, i, 1, c])

            # split features, rescale factors by sample
            split_expand_scale_factors = torch.split(expand_recale_factors, 1, dim=0)
            split_rescale_factors = torch.split(rescale_factors, 1, dim=0)
            split_input_features = torch.split(input_feature, 1, dim=0)

            # calculate the merged features sequentially for each sample
            batch_mean_out = []
            for idx in range(b):

                # select the rescale factors by reusing the function select_with_rescale_factors_bisc
                selected_rescale_factors = select_with_rescale_factors_bisc(
                    split_expand_scale_factors[idx],
                    split_rescale_factors[idx],
                )
                selected_input_feature = select_with_rescale_factors_bisc(
                    split_input_features[idx],
                    split_rescale_factors[idx],
                )

                # [1, s_selected]
                selected_rescale_factors = selected_rescale_factors.mean(dim=[1, 3])

                # [1, i, s_selected, c] -- > [i, s_selected, c]
                _, i, s_select, c = selected_input_feature.shape
                selected_input_feature = torch.reshape(selected_input_feature, (i, s_select, c))

                # norm
                x = self.norm1(selected_input_feature)

                # fusion
                mean_query = torch.mean(x, dim=1, keepdim=True)
                x, attn_weights = self.mha(mean_query, x, x, selected_rescale_factors.tile([i, 1]))
                batch_mean_out.append(x)

            fusion_out = torch.cat(batch_mean_out, dim=0)
        else:
            # [b, i, s, c] -- > [b * i, s, c]
            b, i, s, c = input_feature.shape
            input_feature = torch.reshape(input_feature, (b * i, s, c))

            # norm
            x = self.norm1(input_feature)

            # fusion
            mean_query = torch.mean(x, dim=1, keepdim=True)
            fusion_out, attn_weights = self.mha(mean_query, x, x)

        # [b * i, 1, c] --> [b, i, c]
        fusion_out = torch.reshape(fusion_out, (b, i, c))

        return fusion_out
