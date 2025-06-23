import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
import pickle
from pickle import dump

def DPC_x1_loss(model_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    model_output: (B, T, D, 3) → quantiles
    target:       (B, T, D)    → true x1 values
    """
    # 1. median 
    pred = model_output.median(dim=-1).values  # (B, T, D)

    # 2. x1 extract
    pred_x1 = pred[:, :, 0]
    target_x1 = target[:, :, 0]

    # 3. MSE loss
    loss = F.mse_loss(pred_x1, target_x1)
    return loss

class DPC_Policy_x1(nn.Module):
    def __init__(self, input_dim=4, output_dim=1, output_chunk_length=10, window=5, hidden_dim=32, n_layers=3, dropout_p=0.2):
        super().__init__()
        self.output_dim = output_dim
        self.output_chunk_length = output_chunk_length

        self.input_layer = nn.Linear(input_dim * window, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)

        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)
        ])

        # output_dim × output_chunk_length × quantile (3)
        self.output_layer = nn.Linear(hidden_dim, output_dim * output_chunk_length * 3)

    def forward(self, x_in: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        x_in: tuple of 4 tensors
          - x1_past:     [B, w, 1]  # 과거 window 길이
          - x2_past:     [B, w, 1]
          - u_past:      [B, w, 1]
          - x1_future:   [B, p, 1]  # 미래 prediction 길이
        """
        x1_past, x2_past, u_past, x1_future = x_in  # 각 shape: (B, w, 1) 또는 (B, p, 1)
        min_len = x1_past.shape[1]
        x1_future_cut = x1_future[:, :min_len]  # (B, w, 1)로 맞춤
        x = torch.cat([x1_past, x2_past, u_past, x1_future_cut], dim=-1)  # (B, w, 4)
        x = x.flatten(start_dim=1)  # (B, w*4)

        x = F.relu(self.input_layer(x))
        x = self.dropout(x)

        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)

        x = self.output_layer(x)  # shape: (B, output_dim * w * 3)
        x = x.view(x.shape[0], self.output_chunk_length, self.output_dim, 3)
        return x  # quantile predictions
