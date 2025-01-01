from torch import nn

from models.blocks.decoder_block import DecoderBlock
from models.embeddings.transformer_embedding import TransformerEmebdding

class Decoder(nn.Module):

    def __init__(self, dec_voc_size, max_length, d_model, d_hidden, n_head, n_blocks, drop_prob, device):
        super().__init__()

        self.emb = TransformerEmebdding(vocab_size=dec_voc_size, 
                                        d_model=d_model,
                                        max_length=max_length,
                                        drop_prob=drop_prob,
                                        device=device)
        
        self.blocks = nn.ModuleList([DecoderBlock(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                            for _ in range(n_blocks)])
        
        self.linear = nn.Linear(d_model, dec_voc_size)
    
    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)

        for block in self.blocks:
            trg = block(trg, enc_src, trg_mask, src_mask)
        
        output = self.linear(trg)

        return output