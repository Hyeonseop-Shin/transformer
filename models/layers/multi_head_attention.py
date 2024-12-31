from torch import nn, transpose
from models.layers.scale_dot_product_attention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_num=512, head_num=8):
        super().__init__()
        self.dim_num = dim_num
        self.head_num = head_num

        self.query_embed = nn.Linear(dim_num, dim_num)
        self.key_embed = nn.Linear(dim_num, dim_num)
        self.value_embed = nn.Linear(dim_num, dim_num)
        self.output_embed = nn.Linear(dim_num, dim_num)

        self.scaled_dot_product_attention = ScaleDotProductAttention()

    def forward(self, q, k, v, mask=False):
        batch_size = q.size()[0]

        q = self.query_embed(q).view(batch_size, -1, self.head_num, self.dim_num // self.head_num).transpose(1, 2)
        k = self.key_embed(k).view(batch_size, -1, self.head_num, self.dim_num // self.head_num).transpose(1, 2)
        v = self.value_embed(v).view(batch_size, -1, self.head_num, self.dim_num // self.head_num).transpose(1, 2)

        output = self.scaled_dot_product_attention(q, k, v, mask)
        batch_num, head_num, seq_num, hidden_num = output.size()
        output = transpose(output, 1, 2).contiguous.view((batch_size, -1, hidden_num*self.head_num))

        return output