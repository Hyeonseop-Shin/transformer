import torch
from torch import nn

from models.model.encoder import Encoder
from models.model.decoder import Decoder

class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_index,
                 enc_voc_size, dec_voc_size,
                 d_model, n_head, max_length, d_hidden, n_blocks, drop_prop, device):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_index = trg_sos_index
        self.device = device

        self.encoder = Encoder(enc_voc_size=enc_voc_size,
                               max_length=max_length,
                               d_model=d_model,
                               d_hidden=d_hidden,
                               n_head=n_head,
                               n_blocks=n_blocks,
                               drop_prob=drop_prop,
                               device=device)
        
        self.decoder = Decoder(dec_voc_size=dec_voc_size,
                               max_length=max_length,
                               d_model=d_model,
                               d_hidden=d_hidden,
                               n_head=n_head,
                               n_blocks=n_blocks,
                               drop_prob=drop_prop,
                               device=device)
        
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src=src)
        trg_mask = self.make_trg_mask(trg=trg)

        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output
