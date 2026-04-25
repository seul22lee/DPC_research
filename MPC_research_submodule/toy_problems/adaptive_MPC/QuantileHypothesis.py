import torch
import numpy as np
from scipy.stats import mannwhitneyu


class Quantile_hypothesis:
    def __init__(self, sample_size, quantile_level=[0.05, 0.5, 0.95], alpha=0.25):
        """
        :param sample_size: number of samples
        :param quantile_level: quantile levels to be tested
        :param alpha: significance level
        """
        self.sample_size = sample_size
        self.quantile_level = quantile_level
        self.alpha = alpha
        self.saved_quantile_prev = []
        self.saved_quantile_new = []
    
        self.saved_u_hat_in = []
        self.saved_past_cov = []
        self.saved_target = 0
        self.P = 10
        self.P_count = 0
        
    def collect_quantile_loss(self, prev_model, new_model, val_input, val_state):
        
        u_hat = torch.tensor(val_input["xBest"].reshape(-1,1), requires_grad=False, dtype=torch.float32)
        u_hat_in = u_hat.unsqueeze(0)
        past_cov = torch.tensor(np.concatenate((val_input["x_past_s"],val_input["u_past_s"]), axis = 0),dtype=torch.float32).transpose(1,0).unsqueeze(0)
        
        # TiDE prediction
        pred_prev = prev_model([past_cov, u_hat_in, None])[:,0:1,:,:]
        pred_new = new_model([past_cov, u_hat_in, None])[:,0:1,:,:]
        
        # print(f"pred_prev:{pred_prev[:,0:1,:,:].shape}")
        # print(f"actual states = {val_state}")
        # print(f"pred_new:{pred_new[:,0:1,:,:]}")
        
        quantiles_tensor = torch.tensor(self.quantile_level).to(pred_prev.device)
        target = val_state.reshape(1,1,-1)
        
        # Quantile loss for the previous model
        error_prev = target.unsqueeze(-1) - pred_prev
        q_loss_prev = torch.max(
            (quantiles_tensor - 1) * error_prev, quantiles_tensor * error_prev
        ).sum(dim=3).mean()
        
        # Quantile loss for the new model
        error_new = target.unsqueeze(-1) - pred_new
        q_loss_new = torch.max(
            (quantiles_tensor - 1) * error_new, quantiles_tensor * error_new
        ).sum(dim=3).mean()
        
        self.saved_quantile_prev.append(q_loss_prev.item())
        self.saved_quantile_new.append(q_loss_new.item())
        
        # print(f"prev q loss = {q_loss_prev.item()}, new q loss = {q_loss_new.item()}")
        
        
    def collect_quantile_loss_horizon(self, prev_model, new_model, val_input, val_state):
        
        u_hat = torch.tensor(val_input["xBest"].reshape(-1,1), requires_grad=False, dtype=torch.float32)
        u_hat_in = u_hat.unsqueeze(0)
        past_cov = torch.tensor(np.concatenate((val_input["x_past_s"],val_input["u_past_s"]), axis = 0),dtype=torch.float32).transpose(1,0).unsqueeze(0)
        target = val_state.reshape(1,1,-1)
        
        self.saved_u_hat_in.append(u_hat_in)
        self.saved_past_cov.append(past_cov)
        try:
            if self.saved_target == 0:
                self.saved_target = target
        except:    
            self.saved_target = torch.concat((self.saved_target, target),axis=1)
        
        self.P_count += 1
        
        
        if self.P_count >= self.P:
            # TiDE prediction
            pred_prev = prev_model([self.saved_past_cov[-self.P], self.saved_u_hat_in[-self.P], None])
            pred_new = new_model([self.saved_past_cov[-self.P], self.saved_u_hat_in[-self.P], None])
            
            # print(f"pred_prev:{pred_prev[:,0:1,:,:].shape}")
            # print(f"actual states = {val_state}")
            # print(f"pred_new:{pred_new[:,0:1,:,:]}")

            
            quantiles_tensor = torch.tensor(self.quantile_level).to(pred_prev.device)
            target = self.saved_target[:,-self.P:,:].reshape(1,self.P,-1)
            
            # print(f"========== current step = {self.P_count} ===========")
            # print(f"target = {target}")
            # print(self.saved_past_cov[-self.P])
            
            
            
            # Quantile loss for the previous model
            error_prev = target.unsqueeze(-1) - pred_prev            
            q_loss_prev = torch.max(
                (quantiles_tensor - 1) * error_prev, quantiles_tensor * error_prev
            ).sum(dim=3).mean()
            
            
            # Quantile loss for the new model
            error_new = target.unsqueeze(-1) - pred_new
            q_loss_new = torch.max(
                (quantiles_tensor - 1) * error_new, quantiles_tensor * error_new
            ).sum(dim=3).mean()
            
            self.saved_quantile_prev.append(q_loss_prev.item())
            self.saved_quantile_new.append(q_loss_new.item())
        
            #print(f"prev q loss = {q_loss_prev.item()}, new q loss = {q_loss_new.item()}")    
        
        
    def Hypothesis_testing(self, method = 'mannwhitneyu'):
        if method == 'mannwhitneyu':
            stat, p = mannwhitneyu(self.saved_quantile_new, self.saved_quantile_prev, alternative='less')
            if p < 0.3:
                print("Successful fine-tuning")
                self.saved_quantile_prev = []
                self.saved_quantile_new = []
                self.saved_u_hat_in = []
                self.saved_past_cov = []
                self.saved_target = 0
                self.P_count = 0
                
                return "H1"
            else:
                print("Unsuccessful fine-tuning")
                self.saved_quantile_prev = []
                self.saved_quantile_new = []
                self.saved_u_hat_in = []
                self.saved_past_cov = []
                self.saved_target = 0
                self.P_count = 0
                return "H0"
            
        else:
            raise TypeError("Unknown drift detector type")
        
        
    