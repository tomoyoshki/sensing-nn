import torch
import torch.nn as nn


class RecurrentBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_layers=2, dropout_ratio=0) -> None:
        """The initialization of the recurrent block."""
        super().__init__()

        self.gru = nn.GRU(
            in_channel, out_channel, num_layers, bias=True, batch_first=True, dropout=dropout_ratio, bidirectional=True
        )

    def forward(self, x):
        """The forward function of the recurrent block.
        TODO: Add mask such that only valid intervals are considered in taking the mean.
        Args:
            x (_type_): [b, c_in, intervals]
        Output:
            [b, c_out]
        """
        # [b, c, i] --> [b, i, c]
        x = x.permute(0, 2, 1)

        # GRU --> mean
        # [b, i, c] --> [b, i, c]
        print("X has shape: ", x.shape)
        output, hidden_output = self.gru(x)
        print("Encoder output has shape: ", output.shape)
        print("Encoder hidden output has shape: ", hidden_output.shape)
        print()
        # [b, i, c] --> [b, c]
        output = torch.mean(output, dim=1)

        return output, hidden_output


class DecRecurrentBlock(nn.Module):
    def __init__(self, mod_interval, in_channel, out_channel, num_layers=2, dropout_ratio=0) -> None:
        """The initialization of the decoder recurrent block."""
        super().__init__()
        self.mod_interval = mod_interval
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.embed_dim = out_channel * num_layers

        self.mean_decoder = nn.Linear(self.embed_dim, self.embed_dim * mod_interval)
        self.gru = nn.GRU(
            in_channel * 2 + out_channel,
            out_channel,
            num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_ratio,
            bidirectional=True,
        )

        self.dec_rnn_layer = nn.Linear(out_channel * 2, in_channel)

    def forward(self, x, hidden_state):
        """The forward function of the recurrent block.
        TODO: Add mask such that only valid intervals are considered in taking the mean.
        Args:
            x (_type_): [b, c_in, intervals]
        Output:
            [b, c_out]
        """
        print("Recurrent shape before permute: ", x.shape)

        # [b, c] -> [b, i, c]
        # corresponds to the torch.mean
        # embeded_input = embeded_input.reshape(embeded_input.shape[0], -1, self.out_channel)

        print("X before embedding: ", x.shape)
        x = self.mean_decoder(x)
        x = x.reshape(x.shape[0], -1, self.embed_dim)
        print("X after embedding: ", x.shape)

        # last_layer_state = hidden_state[-1]

        # context = last_layer_state.repeat(x.shape[1], 1, 1).permute(1, 0, 2)

        # print("X has shape: ", x.shape)
        # print("Context has shape: ", context.shape)
        # X_and_context = torch.cat((x, context), 2)

        print("Decoder RNN final input: ", x.shape)
        dec_rnn_features, state = self.gru(x, hidden_state)

        dec_features = self.dec_rnn_layer(dec_rnn_features)

        dec_features = dec_features.permute(0, 2, 1)

        return dec_features
