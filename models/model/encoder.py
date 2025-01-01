from torch import nn

from models.blocks.encoder_block import EncoderBlock
from models.embeddings.transformer_embedding import TransformerEmebdding

class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_length, d_model, d_hidden, n_head, n_blocks, drop_prob, device):
        super().__init__()

        self.emb = TransformerEmebdding(vocab_size=enc_voc_size, 
                                        d_model=d_model, 
                                        max_length=max_length,
                                        drop_prob=drop_prob,
                                        device=device)
        
        self.blocks = nn.ModuleList([EncoderBlock(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                            for _ in range(n_blocks)])
        
    def forward(self, x, src_mask):
        x = self.emb

        for block in self.blocks:
            x = block(x, src_mask)
        
        return x
            