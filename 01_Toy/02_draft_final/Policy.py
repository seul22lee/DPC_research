import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
import pickle
from pickle import dump

window = 10

def DPC_loss(
    model_output: torch.Tensor,
    target: torch.Tensor,
    u_output: torch.Tensor,
    c_fut: torch.Tensor,
    return_all: bool = False
):

    model_output = model_output.median(dim=-1).values
    model_output_x1 = model_output[:, :, 0]
    target_x1 = target[:, :, 0]
    errors = (target_x1 - model_output_x1) ** 2

    u_diff = u_output[:, 1:, :] - u_output[:, :-1, :]
    squared_diff = u_diff ** 2

    model_output_x2 = model_output[:, :, 1]
    low_violation = F.relu(c_fut[:, :, 0] - model_output_x2) ** 2
    up_violation = F.relu(model_output_x2 - c_fut[:, :, 1]) ** 2
    constraint_loss = low_violation + up_violation

    tracking_loss_sqrt = torch.sqrt(errors.mean())
    smoothness_loss_sqrt = 0.1 * torch.sqrt(squared_diff.mean())
    constraint_loss_sqrt = 3 * torch.sqrt(constraint_loss.mean())

    # Constraint violation per sample
    violated = (model_output_x2 < c_fut[:, :, 0]) | (model_output_x2 > c_fut[:, :, 1])
    #violation_per_sample = violated.any(dim=1)
    #constraint_violation_mean = violation_per_sample.float().mean().item()

    violation_ratio_per_sample = violated.float().mean(dim=1)
    constraint_violation_mean = violation_ratio_per_sample.mean().item()

    total_loss = tracking_loss_sqrt + smoothness_loss_sqrt + constraint_loss_sqrt

    if return_all:
        return total_loss, tracking_loss_sqrt.item(), smoothness_loss_sqrt.item(), constraint_loss_sqrt.item()
    else:
        return total_loss, constraint_violation_mean




class DPC_PolicyNN(nn.Module):
    def __init__(self, input_dim, output_dim, future_cov_dim, static_cov_dim,
                 input_chunk_length, output_chunk_length, hidden_dim, n_layers):
        super().__init__()
        self.output_dim = output_dim
        self.output_chunk_length = output_chunk_length

        self.input_layer = nn.Linear(60, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim * output_chunk_length)

    def forward(self, x_in):
        x, x_future_covariates, c_fut, x_static_covariates = x_in
        x = x.flatten(start_dim=1)
        if c_fut is not None:
            x = torch.cat([x, c_fut.flatten(start_dim=1)], dim=1)
        if x_future_covariates is not None:
            x = torch.cat([x, x_future_covariates.flatten(start_dim=1)], dim=1)
        if x_static_covariates is not None:
            x = torch.cat([x, x_static_covariates.flatten(start_dim=1)], dim=1)

        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)

        return x.view(x.shape[0], self.output_chunk_length, self.output_dim, 1)
