import torch.nn as nn
import torch
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
import math

__all__ = [
    'PositionalEncoding',
    'SimpleTransformer'
]


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SimpleTransformer(nn.Module):

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _generate_square_context_mask(self, sz, k=8):
        mask = torch.zeros(sz, sz)
        rg = torch.arange(k)
        for i in range(sz):
            mask[i, torch.clamp(rg + i - (2 // k), min=0, max=sz - 1)] = 1.
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super(SimpleTransformer, self).__init__()
        layer = TransformerEncoderLayer(d_model=d_model,
                                        nhead=nhead,
                                        dim_feedforward=dim_feedforward,
                                        dropout=0.0,
                                        activation="gelu")
        norm = nn.LayerNorm(d_model)
        self.model = TransformerEncoder(layer, num_layers=num_layers, norm=None)
        self.src_mask = None

    def forward(self, src):
        src = src.permute(1, 0, 2).contiguous()
        device = src.device
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_context_mask(len(src)).to(device)
            self.src_mask = mask
        return self.model(src, self.src_mask).permute(1, 0, 2).contiguous()