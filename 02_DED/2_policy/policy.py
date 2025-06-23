import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class PolicyNN(nn.Module):
    def __init__(self, 
                 past_input_dim=6, 
                 future_input_dim=4, 
                 output_dim=2, 
                 output_chunk_length=50,
                 window=50,
                 hidden_dim=128,
                 n_layers=4,
                 dropout_p=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.output_chunk_length = output_chunk_length

        input_dim_total = past_input_dim * window + future_input_dim * output_chunk_length

        self.input_layer = nn.Linear(input_dim_total, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)

        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)
        ])

        self.output_layer = nn.Linear(hidden_dim, output_dim * output_chunk_length * 3)

    def forward(self, x_in: Tuple[torch.Tensor, torch.Tensor]):
        past, future = x_in
        x = torch.cat([past.flatten(start_dim=1), future.flatten(start_dim=1)], dim=1)

        x = F.relu(self.input_layer(x))
        x = self.dropout(x)

        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)

        x = self.output_layer(x)
        x = x.view(x.shape[0], self.output_chunk_length, self.output_dim, 3)
        return x
