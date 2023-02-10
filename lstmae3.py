import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, z_dim: int, hidden_layers: int = 1
    ):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, hidden_layers)
        self.bottleneck = nn.LSTM(hidden_dim, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x.to(torch.float32))
        _, (z, _) = self.bottleneck(h)
        return z[0]


class Decoder(nn.Module):
    def __init__(
        self, z_dim: int, hidden_dim: int, output_dim: int, hidden_layers: int = 1
    ):
        super().__init__()
        self.rnn = nn.LSTM(z_dim, hidden_dim, hidden_layers)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor, seq_len: int = 10) -> torch.Tensor:
        z = z.unsqueeze(0)
        h, _ = self.rnn(z)
        x = self.output(h)
        return torch.flatten(x)


class Autoencoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, z_dim: int, hidden_layers: int = 1
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, z_dim, hidden_layers)
        self.decoder = Decoder(z_dim, hidden_dim, input_dim, hidden_layers)

    def forward(self, x: torch.Tensor, return_z: bool = False) -> torch.Tensor:
        seq_len, *_ = x.shape
        z = self.encoder(x)
        x = self.decoder(z, seq_len)
        return (x, z) if return_z else x
