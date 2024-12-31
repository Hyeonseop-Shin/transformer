from torch import mean, std, nn


class AddLayerNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def layer_norm(self, x):
        mean = mean(x, dim=-1, keepdim=True)
        std = std(x, dim=-1, keepdim=True)

        return (x - mean) / std

    def forward(self, x, residual):
        return residual + self.layer_norm(x)