from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionWiseFeedForward

class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, d_hidden=4*512, n_head=8, drop_prob=0.1):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.feedforward = PositionWiseFeedForward(d_model=d_model, d_hidden=d_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, mask):
        _x = x
        x = self.multi_head_attention(q=x, k=x, v=x, mask=mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.feedforward(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        return x