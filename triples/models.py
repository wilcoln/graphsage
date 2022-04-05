import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """ Single Layer Perceptron for regression. """

    def __init__(self, in_channels, hidden_channels=100, out_channels=None, num_layers=2):
        assert num_layers >= 2, "At least 2 layers are required."

        super(MLP, self).__init__()

        self.name = f'{num_layers}-MLP'

        if out_channels is None:
            out_channels = hidden_channels

        self.layers = nn.ModuleList(
            [nn.Linear(in_channels, hidden_channels)] +
            [
                nn.Linear(hidden_channels, hidden_channels)
                for _ in range(num_layers - 2)
            ] +
            [nn.Linear(hidden_channels, out_channels)]
        )

        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_channels) for _ in range(num_layers-1)] +
            [nn.BatchNorm1d(out_channels)]
        )

    def forward(self, x):
        for i, (layer, batch_norm) in enumerate(zip(self.layers, self.batch_norms)):
            x = layer(x)
            x = batch_norm(x)
            if i < len(self.layers) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def reset_parameters(self):
        self.in_layer.reset_parameters()
        for layer in self.hidden_layers:
            layer.reset_parameters()
        self.output_layer.reset_parameters()

