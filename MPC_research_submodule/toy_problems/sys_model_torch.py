import torch
from scipy.stats import truncnorm

def nominal_model(x_k, u_k):
    ''' 
    Nominal model (noise clean), an unstable LTI model from literature.
    
    Arguments:
      - x_k: current state (N by 1)
      - u_k: applied input (1 by 1)

    Output:
      - x_next: output state (N by 1)
    '''
    A = torch.tensor([[0.3, 0.1], [0.1, 0.2]], dtype=torch.float32)
    B = torch.tensor([[0.5], [1.0]], dtype=torch.float32)
    # calculate derivative
    return A @ x_k + B * u_k

def nominal_model_recur(x_current, u_seq):
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
        x_pred_seq_temp = nominal_model(x_current, u_seq[i,:])
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

def system_model(x_k, u_k):
    ''' 
    System model with noise injected. 
    
    Arguments:
      - x_k: current state (N by 1)
      - u_k: applied input (1 by 1)

    Output:
      - x_next: output state (N by 1)
    '''
    A = torch.tensor([[0.3, 0.1], [0.1, 0.2]], dtype=torch.float32)
    B = torch.tensor([[0.5], [1.0]], dtype=torch.float32)
    mu = torch.tensor([[0.0], [0.0]], dtype=torch.float32)   
    w = torch.tensor([[0.1], [0.2]], dtype=torch.float32)
    
    ub = torch.tensor([0.025*1.645, 0.05*1.645])
    lb = ub*-1
    
    w = torch.tensor([[0.05],[0.1]])
    noise = torch.normal(mean=mu, std=w)
    
    # noise = multivariate_uniform(lb, ub, 1).reshape(-1,1)
    return A @ x_k + B * u_k + noise

# K_ac = torch.tensor([[-0.2068,-0.6756]])*0.3