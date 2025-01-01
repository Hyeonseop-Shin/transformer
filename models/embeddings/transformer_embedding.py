from torch import nn

from models.embeddings.positional_encoding import PositionalEncoding
from models.embeddings.token_embedding import TokenEmbedding

class TransformerEmebdding(nn.Module):
    
    def __init__(self, vocab_size, d_model, max_length, drop_prob, device):
        super(TransformerEmebdding, self).__init__()

        self.tok_embed = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_length=max_length, device=device)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_embed = self.tok_embed(x)
        pos_enc = self.pos_enc(x)
        return self.dropout(tok_embed + pos_enc)