import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
from pickle import dump

window = 10
p = 10


def DPC_loss(
    model_output: torch.Tensor,
    target: torch.Tensor,
    u_output: torch.Tensor,
    c_fut: torch.Tensor,
    smoothness_weight: Union[float, torch.Tensor] = 1.0, ## initially 0.1
    constraint_weight: Union[float, torch.Tensor] = 3.0
) -> torch.Tensor:

    model_output = model_output.median(dim=-1).values
    errors = (target[:, :, 0] - model_output[:, :, 0]) ** 2

    u_diff = u_output[:, 1:, :] - u_output[:, :-1, :]
    smooth_term = (u_diff ** 2).mean(dim=-1)

    low_v = F.relu(c_fut[:, :, 0] - model_output[:, :, 1]) ** 2
    up_v = F.relu(model_output[:, :, 1] - c_fut[:, :, 1]) ** 2
    constr_term = low_v + up_v

    sw = torch.as_tensor(smoothness_weight, dtype=errors.dtype, device=errors.device)
    cw = torch.as_tensor(constraint_weight, dtype=errors.dtype, device=errors.device)

    tracking_loss = torch.sqrt(errors.mean())
    smoothness_loss = torch.sqrt((sw * smooth_term).mean())
    constraint_loss = torch.sqrt((cw * constr_term).mean())

    return tracking_loss + smoothness_loss + constraint_loss


class DPC_PolicyNN(nn.Module):
    def __init__(
        self,
        input_dim: int,  
        output_dim: int,
        future_cov_dim: int,
        static_cov_dim: int,
        input_chunk_length: int,
        output_chunk_length: int,
        hidden_dim: int,
        dropout_prob: float = 0.1  # dropout 
    ):
        super(DPC_PolicyNN, self).__init__()

        self.input_dim = input_dim  
        self.output_dim = output_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length

        self.fc1 = nn.Linear(60, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, output_dim * output_chunk_length)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob) 

    def forward(
        self, x_in: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]
    ) -> torch.Tensor:
        x, x_future_covariates, c_fut, x_static_covariates = x_in

        if len(x.shape) == 3:
            x = x.flatten(start_dim=1) 

        if c_fut is not None:
            c_fut = c_fut.flatten(start_dim=1)
            x = torch.cat([x, c_fut], dim=1)

        if x_future_covariates is not None:
            x_future_covariates = x_future_covariates.flatten(start_dim=1)
            x = torch.cat([x, x_future_covariates], dim=1)

        if x_static_covariates is not None:
            x_static_covariates = x_static_covariates.flatten(start_dim=1)
            x = torch.cat([x, x_static_covariates], dim=1)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))
        x = self.dropout(self.relu(self.fc5(x)))
        x = self.dropout(self.relu(self.fc6(x)))
        x = self.fc7(x)

        batch_size = x.shape[0]
        x = x.view(batch_size, self.output_chunk_length, self.output_dim, 1)

        return x
