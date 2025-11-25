import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRU(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layer,
        num_class,
        device,
        dropout: float = 0.3,
        use_attention: bool = True,
        use_pack: bool = True,
    ) -> None:
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layer,
            batch_first=True,
            dropout=dropout if num_layer > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_class)
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.device = device
        self.use_attention = use_attention
        self.use_pack = use_pack
        if use_attention:
            self.attn_score = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, lengths=None):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layer, batch_size, self.hidden_size, device=self.device)

        if self.use_pack and lengths is not None:
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.gru(packed, h0)
            out, out_lengths = pad_packed_sequence(packed_out, batch_first=True)
            out_lengths = out_lengths.to(out.device)
        else:
            out, _ = self.gru(x, h0)
            out_lengths = lengths if lengths is not None else torch.full(
                (batch_size,), out.size(1), device=x.device
            )

        if self.use_attention:
            max_len = out.size(1)
            mask = (
                torch.arange(max_len, device=out.device)[None, :] < out_lengths[:, None]
            )  # [B, T]
            scores = self.attn_score(out).squeeze(-1)  # [B, T]
            scores = scores.masked_fill(~mask, float("-inf"))
            attn_weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B,T,1]
            context = torch.sum(out * attn_weights, dim=1)  # [B, H]
        else:
            # fallback: last valid timestep
            indices = (out_lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
            context = out.gather(1, indices).squeeze(1)

        context = self.dropout(context)
        logits = self.fc(context)
        return logits

