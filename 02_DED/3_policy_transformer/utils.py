
import torch
from torch.nn import ReLU, Sigmoid



# =========== Sigmoid ===================
def sigmoid(x,u_ub, u_lb):
    sig = Sigmoid()
    return sig(x)*(u_ub-u_lb) + u_lb


class LossParameters():
    def __init__(self, alpha0, delta_alpha, lamda0):
        # initial values
        self.alpha0 = alpha0
        self.delta_alpha = delta_alpha
        self.lamda0 = lamda0
        # parameters that varies
        self.alpha_0 = torch.tensor(alpha0)
        self.alpha_1 = torch.tensor(alpha0)
        self.alpha_2 = torch.tensor(alpha0)
        self.alpha_3 = torch.tensor(alpha0)
        self.delta_alpha = delta_alpha
        self.lamda_0 = torch.tensor(lamda0)
        self.lamda_1 = torch.tensor(lamda0)
        self.lamda_2 = torch.tensor(lamda0)
        self.lamda_3 = torch.tensor(lamda0)
    
    def reset(self):
        self.alpha_0 = torch.tensor(self.alpha0)
        self.alpha_1 = torch.tensor(self.alpha0)
        self.alpha_2 = torch.tensor(self.alpha0)
        self.alpha_3 = torch.tensor(self.alpha0)
        self.delta_alpha = self.delta_alpha
        self.lamda_0 = torch.tensor(self.lamda0)
        self.lamda_1 = torch.tensor(self.lamda0)
        self.lamda_2 = torch.tensor(self.lamda0)
        self.lamda_3 = torch.tensor(self.lamda0)


class GlobalState():
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalState, cls).__new__(cls)
            cls._instance.f0 = None
            cls._instance.optim_iter_count = 0
        return cls._instance
    
    def update_f0(self, new_f0):
        self.f0 = new_f0
            
    def update_optim_iter_count(self):
        self.optim_iter_count += 1
    
    def get_f0(self):
        return self.f0
    
    def reset_global_state(self):
        self.f0 = None
        self.optim_iter_count = 0
        

class BarrierParam():
    _instance = None
    def __init__(self, t0, mu):
        self.t0 = torch.tensor(t0)
        self.t = torch.tensor(t0)
        self.mu = torch.tensor(mu)
        self.optim_iter_count = 0
    
    def update_t(self):
        self.t *= self.mu
            
    def update_optim_iter_count(self):
        self.optim_iter_count += 1
        
    def reset(self):
        self.t = self.t0
        self.optim_iter_count = 0
        
    
def softmax_max(u, alpha=5):
    # weights = torch.softmax(alpha * u, dim=0)
    # return (weights * u).sum(dim=0)
    return torch.logsumexp(alpha * u, dim=0) / alpha

def softmax_min(u, alpha=5):
    # weights = torch.softmax(-alpha * u, dim=0)
    # return (weights * u).sum(dim=0)
    return -torch.logsumexp(-alpha * u, dim=0) / alpha