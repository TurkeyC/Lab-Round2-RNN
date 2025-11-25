import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
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
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
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
        c0 = torch.zeros(self.num_layer, batch_size, self.hidden_size, device=self.device)

        if self.use_pack and lengths is not None:
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed, (h0, c0))
            out, out_lengths = pad_packed_sequence(packed_out, batch_first=True)
            out_lengths = out_lengths.to(out.device)
        else:
            out, _ = self.lstm(x, (h0, c0))
            out_lengths = lengths if lengths is not None else torch.full(
                (batch_size,), out.size(1), device=x.device
            )

        if self.use_attention:
            max_len = out.size(1)
            mask = (
                torch.arange(max_len, device=out.device)[None, :] < out_lengths[:, None]
            )
            scores = self.attn_score(out).squeeze(-1)
            scores = scores.masked_fill(~mask, float("-inf"))
            attn_weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            context = torch.sum(out * attn_weights, dim=1)
        else:
            indices = (out_lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
            context = out.gather(1, indices).squeeze(1)

        context = self.dropout(context)
        logits = self.fc(context)
        return logits

