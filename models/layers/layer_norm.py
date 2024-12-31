from torch import nn, ones, zeros, sqrt


class LayerNorm(nn.Module):
    def __init__(self, d_model=512, epsilon=1e-12):
        super().__init__()
        
        self.gamma = nn.Parameter(ones(d_model))
        self.beta = nn.Parameter(zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=True, keepdim=True)
        # -1 means the last dimension

        x = (x - mean) / sqrt(var + self.epsilon)
        x = self.gamma * x + self.beta

        return x