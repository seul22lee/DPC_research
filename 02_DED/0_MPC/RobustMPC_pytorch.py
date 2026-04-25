## Robust MPC main module with pytorch minimize implementation

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from scipy.optimize import minimize, Bounds

from scipy.stats import qmc

from torchmin import minimize

class RMPC:
    def __init__(self,
                 u_hat:torch.tensor,           # initial guess of u_hat for optimizer
                 x_past:torch.tensor,          # past history of x (melt pool temperature) every MPC time step
                 u_past:torch.tensor,          # past history of u (laser power) every MPC time step
                 fix_cov_past: torch.tensor,   # past history of fix covariates every MPC time step
                 window:torch.tensor,          # window length for x and u. if include_u_hist==True, window = 1 for x
                 P:int,                        # length of horizon
                 x_current:torch.tensor,       # current state x (melt pool temperature)
                 K_ac:torch.tensor,            # gain of ancillary controller
                 MPC_model,                    # Modulized ML class, should include functional scaler, NN_forward
                 optim_obj,                    # Objective function. the default setting is using SLSQP with jac=True
                 include_u_hist: False,        # True:include longer history of u but only one step of x; False: include all the past x and u within horizon
                 error_past=None,              # initial value of error history
                 constraint=None,              # constraint function
                 tube_model=None,
                 u_tighten=None,
                 u_bound = ([-5,5]),           # bound of u, should be a tuple where each elements is a list of (lb, ub)
                 state_dim=int,                # Number of states
                 input_dim=int,                # Number of inputs
                 ) -> None:
        
        # Initialization
        self.u_hat = u_hat
        self.x_past = x_past
        self.u_past = u_past
        self.x_current = x_current
        self.x_sys_current = x_current
        self.x_hat_current = x_current
        self.K_ac = K_ac
        self.MPC_model = MPC_model
        self.window = window
        self.P = P
        self.optim_obj = optim_obj
        self.include_u_hist = include_u_hist
        self.error_past = error_past
        self.constraint = constraint
        self.tube_model = tube_model
        self.u_tighten = u_tighten
        self.count = 0
        
        # Initialize saving tensors
        self.e_save = torch.empty((2, 0), dtype=torch.float32)
        self.real_output = torch.empty((2, 0), dtype=torch.float32)
        self.nominal_output = torch.empty((2, 0), dtype=torch.float32)
        self.nominal_u = torch.empty((0, 1), dtype=torch.float32)
        self.u_applied_save = torch.empty((0, 1), dtype=torch.float32)
        self.true_pred_output_x1_save = torch.empty((0, P), dtype=torch.float32)
        self.true_pred_output_x2_save = torch.empty((0, P), dtype=torch.float32)
        self.x_hat_horizon_save = torch.empty((0, P), dtype=torch.float32)
        self.error_past_x1_upper = torch.empty((0, P), dtype=torch.float32)
        self.error_past_x1_lower = torch.empty((0, P), dtype=torch.float32)
        self.error_past_x2_upper = torch.empty((0, P), dtype=torch.float32)
        self.error_past_x2_lower = torch.empty((0, P), dtype=torch.float32)
        
        return None
    
    