import math
from torch import nn, matmul

class ScaleDotProductAttention(nn.Module):

    def __inti__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=False, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_k]

        batch_size, head, length, d_k = k.size()

        k_T = k.transpose(2, 3)
        output = matmul(q, k_T) / math.sqrt(d_k)

        if mask:
            output = output.masked_fill(mask.unsqueeze(1).unsqueeze(-1), 0)

        output = self.softmax(output)
        output = matmul(output, v)

        return output
