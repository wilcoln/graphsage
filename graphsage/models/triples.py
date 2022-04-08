import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """ Multi-layer perceptron. """

    def __init__(self, in_channels, hidden_channels=100, out_channels=None, num_layers=2):
        assert num_layers >= 1, "num_layers must be at least 1"
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
        for layer in self.layers:
            layer.reset_parameters()


class TriplesMLP(nn.Module):
    """ Multi-layer perceptron for triples. """

    def __init__(self, in_channels, hidden_channels=100, out_channels=None, num_layers=2):
        super().__init__()
        self.phi = MLP(in_channels//2, hidden_channels, out_channels, num_layers)  # node encoder

    def forward(self, x):
        # compute the representation for each data point
        u, v = torch.split(x, x.shape[1] // 2, dim=1)
        u_phi = self.phi(u)  # representation of source node u
        v_phi = self.phi(v)  # representation of target node v

        return torch.cat([u_phi, v_phi], dim=1)

    def reset_parameters(self):
        self.phi.reset_parameters()
