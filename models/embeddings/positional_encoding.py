import torch
from torch import nn

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_length, device):
        super().__init__()

        self.encoding = torch.zeros(max_length, d_model, device=device)
        self.encoding.requires_grad = False # we dont need to compute gradient

        pos = torch.arange(0, max_length, device=device)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        batch_size, seq_len = x.size()

        return self.encoding[:seq_len, :]


