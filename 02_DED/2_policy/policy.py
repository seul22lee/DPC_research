import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class PolicyNN(nn.Module):
    def __init__(self, 
                 past_input_dim=6, 
                 future_input_dim=6, 
                 output_dim=1, 
                 p=50,
                 window=50,
                 hidden_dim=1024,
                 n_layers=3,
                 dropout_p=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.p = p

        input_dim_total = past_input_dim * window + future_input_dim * p

        self.input_layer = nn.Linear(input_dim_total, hidden_dim)
        self.input_ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)

        self.hidden_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for _ in range(n_layers - 2):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.norm_layers.append(nn.LayerNorm(hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, output_dim * p)

    def forward(self, x_in: Tuple[torch.Tensor, torch.Tensor]):
        past, future = x_in
        x = torch.cat([past.flatten(start_dim=1), future.flatten(start_dim=1)], dim=1)

        x = self.input_layer(x)
        # x = self.input_ln(x)
        x = F.leaky_relu(x, negative_slope=0.01) # leaky relu 
        x = self.dropout(x)

        for layer, norm in zip(self.hidden_layers, self.norm_layers):
            residual = x
            x = layer(x)
            # x = norm(x)
            x = F.leaky_relu(x, negative_slope=0.01) # leaky relu 
            x = self.dropout(x)
            x = x + residual  # residual connection

        x = self.output_layer(x)
        x = x.view(x.shape[0], self.p, self.output_dim)
        return x