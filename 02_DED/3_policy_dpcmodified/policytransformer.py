import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyTransformer(nn.Module):
    def __init__(self,
                 past_input_dim=6,
                 future_input_dim=6,
                 output_dim=1,
                 window=50,
                 p=50,
                 d_model=128,
                 n_heads=4,
                 d_ff=512,
                 n_layers=4,
                 dropout=0.1):
        super().__init__()
        self.window = window
        self.p = p

        # total sequence length = past + future
        self.seq_len = window + p
        # all tokens have same feature dim = past_input_dim (== future_input_dim)
        token_dim = past_input_dim

        # embed token_dim → d_model
        self.token_embed = nn.Linear(token_dim, d_model)
        # learnable positional embeddings
        self.pos_embed = nn.Embedding(self.seq_len, d_model)

        # encoder-only Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # projection head: d_model → output_dim
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x_in):
        past, future = x_in
        # past: (B, window, token_dim)
        # future: (B, p, token_dim)
        x = torch.cat([past, future], dim=1)            # → (B, seq_len, token_dim)
        x = self.token_embed(x)                         # → (B, seq_len, d_model)

        # add positional embeddings
        positions = torch.arange(self.seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_embed(positions)               # → (B, seq_len, d_model)

        # encode
        enc_out = self.encoder(x)                       # → (B, seq_len, d_model)

        # take only future positions for control
        tail = enc_out[:, self.window:, :]              # → (B, p, d_model)
        u = self.head(tail).squeeze(-1)                 # → (B, p)
        return u
