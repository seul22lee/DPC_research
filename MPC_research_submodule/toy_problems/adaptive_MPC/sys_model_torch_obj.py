import torch
from scipy.stats import truncnorm

class sys_model:
  def __init__(self,
               A = torch.tensor([[0.3, 0.1], [0.1, 0.2]], dtype=torch.float32),
               B = torch.tensor([[0.5], [1.0]], dtype=torch.float32),
               mu = torch.tensor([[0.0], [0.0]], dtype=torch.float32),
               w = torch.tensor([[0.0],[0.1]], dtype=torch.float32)):
    
    self.A = A
    self.B = B
    self.mu = mu       
    self.w = w
    self.system_ID = 0
  
  def nominal_model(self,x_k, u_k):
      ''' 
      Nominal model (noise clean), an unstable LTI model from literature.
      
      Arguments:
        - x_k: current state (N by 1)
        - u_k: applied input (1 by 1)

      Output:
        - x_next: output state (N by 1)
      '''

      # calculate derivative
      if self.system_ID == 0:
        return self.A @ x_k + self.B * u_k + self.mu
      elif self.system_ID == 1:
        nonl_term_1 = torch.tensor([[0.1, 0], [0, 0.1]], dtype=torch.float32)
        nonl_term_2 = torch.tensor([[0, 0.1], [0.1, 0]], dtype=torch.float32)
        nonl_term_3 = torch.tensor([[0.5, 0], [0, 0]], dtype=torch.float32)
        nonl_term_4 = torch.tensor([[0, 0], [0, 0.5]], dtype=torch.float32)
        perturb = 0.3 * torch.tanh(x_k) + 0.1 * torch.tanh(u_k) 
        # print((nonl_term_1@torch.sin(x_k) + nonl_term_2@torch.cos(x_k)) * x_k + (nonl_term_3@torch.sin(x_k) + nonl_term_4@torch.cos(x_k)) * u_k)
        # return self.A @ x_k + self.B * u_k + self.mu + ((nonl_term_2@torch.cos(0.5*x_k)) * x_k + (nonl_term_3@torch.sin(x_k) + nonl_term_4@torch.cos(x_k)) * u_k)
        return self.A @ x_k + self.B * u_k + self.mu + perturb

  def nominal_model_recur(self,x_current, u_seq):
      ''' 
      Recurring nominal model.
      
      Arguments:
        - x_current: current state (N by 1) at the first step
        - u_seq: applied input sequence (1 by n_step)

      Output:
        - x_pred_seq: output sequence of future state within the horizon (N by n_step)
      '''
      seq_len = u_seq.shape[0]
      x_pred_seq = torch.empty((2, 0), dtype=torch.float32)
      for i in range(seq_len):
          x_pred_seq_temp = self.nominal_model(x_current, u_seq[i,:])
          x_current = x_pred_seq_temp
          x_pred_seq = torch.cat((x_pred_seq, x_pred_seq_temp), dim=1)
      return x_pred_seq


  def multivariate_uniform(low, high, size):
      """
      Generate random samples from a multivariate uniform distribution.

      Parameters:
      low (list or tensor): Lower bounds for each dimension.
      high (list or tensor): Upper bounds for each dimension.
      size (int): Number of samples to generate.

      Returns:
      tensor: A tensor of shape (size, len(low)) containing the generated samples.
      """
      low = torch.tensor(low)
      high = torch.tensor(high)
      return torch.rand(size,1) * (high - low) + low

  def system_model(self, x_k, u_k):
      ''' 
      System model with noise injected. 
      
      Arguments:
        - x_k: current state (N by 1)
        - u_k: applied input (1 by 1)

      Output:
        - x_next: output state (N by 1)
      '''
      
      noise = torch.normal(mean=self.mu, std=self.w)
      
      # noise = multivariate_uniform(lb, ub, 1).reshape(-1,1)
      if self.system_ID == 0:
        return self.A @ x_k + self.B * u_k + noise
      elif self.system_ID == 1:
        nonl_term_1 = torch.tensor([[0.1, 0], [0, 0.1]], dtype=torch.float32)
        nonl_term_2 = torch.tensor([[0, 0.1], [0.1, 0]], dtype=torch.float32)
        nonl_term_3 = torch.tensor([[0.5, 0], [0, 0]], dtype=torch.float32)
        nonl_term_4 = torch.tensor([[0, 0], [0, 0.5]], dtype=torch.float32)
        perturb = 0.3 * torch.tanh(x_k) + 0.1 * torch.tanh(u_k)
        # return self.A @ x_k + self.B * u_k + noise + ((nonl_term_2@torch.cos(0.5*x_k)) * x_k + (nonl_term_3@torch.sin(x_k) + nonl_term_4@torch.cos(x_k)) * u_k)
        return self.A @ x_k + self.B * u_k + noise + perturb 