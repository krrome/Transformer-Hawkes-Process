import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        # The attention is (n_batch, n_heads, seqlen, seqlen). Masking is applied by masking out "future" keys -
        # masking is on the final dimension.

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))

        # the attention (((q_1, k_1), 0 .. 0), ((q_2, k_1), (q_2, k_2), 0 .. 0)) etc. is then used as weights on
        # the values for every position to produce an output
        output = torch.matmul(attn, v)

        return output, attn
