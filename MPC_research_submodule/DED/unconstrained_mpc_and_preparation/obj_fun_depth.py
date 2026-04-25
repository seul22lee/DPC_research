import torch
from torch.nn import ReLU

def RMPC_obj_wo_constraint(u_hat0:torch.tensor, # future u values within the control horizon, length = M; this should be the warm start version
                u_future_fix:torch.tensor, # future u values within the control horizon but not the design variables
                u_past:torch.tensor, # past u
                x_past_fix: torch.tensor, # past state of x, size (N,window)
                x_past:torch.tensor, # past state that changes
                SP_hat:torch.tensor, # Reference trajectory, length = P
                P, # Predictive Horizon
                NN_Nominal # NN object we're using
):
    ''' 
    All the data that goes into this module should be normalized
    '''    
    
    u_hat = u_hat0.reshape(-1,1)
    u_hat_in = u_hat0.unsqueeze(0)
    x_hat_all = NN_Nominal.forward(u_hat, u_future_fix, u_past, x_past_fix, x_past)
    
    x_hat = x_hat_all[:,0]
    
    # compute objective value
    u_hat_temp = u_hat_in[0,:,0].reshape(-1,1) 

    #u = u_hat_temp[0].reshape(-1,1)
    u = u_past[-1].reshape(-1,1)
    u_hat1 = torch.concatenate((u,u_hat_temp)) # append the computed u with the history u 
    
    #Obj = 1000 * torch.sum((x_hat-torch.tensor(SP_hat,dtype=torch.float32))**2) + 100*torch.sum((u_hat0[1:]-u_hat0[0:-1])**2)
    
    Obj = 1 * torch.sum((x_hat-torch.tensor(SP_hat.transpose(1,0),dtype=torch.float32))**2) + 10*torch.sum((u_hat1[:-1]-u_hat_temp)**2) \
        # + (x_hat[-1] - torch.tensor(SP_hat[0],dtype=torch.float32))**2

    relu = ReLU()
    
    penalty = torch.sum(relu(u - torch.ones_like(u))**2) + torch.sum(relu(-torch.ones_like(u) - u)**2)
    

    return Obj + penalty


def RMPC_obj_wo_constraint_scipy(u_hat0:torch.tensor, # future u values within the control horizon, length = M; this should be the warm start version
                u_future_fix:torch.tensor, # future u values within the control horizon but not the design variables
                u_past:torch.tensor, # past u
                x_past_fix: torch.tensor, # past state of x, size (N,window)
                x_past:torch.tensor, # past state that changes
                SP_hat:torch.tensor, # Reference trajectory, length = P
                P, # Predictive Horizon
                NN_Nominal # NN object we're using
):
    ''' 
    All the data that goes into this model should be normalized
    '''    
    u_hat = torch.tensor(u_hat0.reshape(-1,1),requires_grad=True,dtype=torch.float32)
    u_hat_in = u_hat.unsqueeze(0)
    x_hat = NN_Nominal.forward(u_hat, u_future_fix, u_past, x_past_fix, x_past)

    # compute objective value

    u_hat_temp = u_hat_in[0,:,0].reshape(-1,1) 

    u = u_past[-1].reshape(-1,1)
    u_hat1 = torch.concatenate((u,u_hat_temp)) # append the computed u with the history u
    
    Obj = 10 * torch.sum((x_hat-torch.tensor(SP_hat,dtype=torch.float32))**2) + 1*torch.sum((u_hat1[:-1]-u_hat_temp)**2)
    Obj.backward()

    
    return Obj.item(), u_hat.grad.numpy().reshape(-1)