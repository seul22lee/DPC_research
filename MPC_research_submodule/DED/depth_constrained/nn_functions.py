# Surrogate
import torch
torch.set_default_dtype(torch.float32)

''' 
This class is a fast wrapper around the NN model that will be used. It should perform two main functions:

1. Scalers
    a. scalers_u: the scaler to scale u from the original scale to (-1,1)
    b. inv_scaler_u: the scaler to scale u_s from (-1,1) to the original scale
    c. scalers_x: the scaler to scale x from original scale to (-1,1); it should support the function when only the assigned states are getting scaled
    d. inv_scaler_u: the scaler to scale x_s from (-1,1) to the original scale
    
2. Forward prediction
    it takes the original quantities, make predictions, and returns whether scaled or unscaled values based on user's preference
'''

class surrogate():
    def __init__(self, sys_params: dict, NN_model, device=None) -> None:
        self.device = device if device is not None else torch.device("cpu")

        # Make sure they're tensors with correct device
        self.x_max = torch.tensor(sys_params["x_max"], dtype=torch.float32, device=self.device)
        self.x_min = torch.tensor(sys_params["x_min"], dtype=torch.float32, device=self.device)
        self.y_max = torch.tensor(sys_params["y_max"], dtype=torch.float32, device=self.device)
        self.y_min = torch.tensor(sys_params["y_min"], dtype=torch.float32, device=self.device)

        self.NN_model = NN_model.to(self.device)
    
# ===================== Scalers =======================

    def scaler_x(self, x_original, dim_id = -1):
        if dim_id == -1:
            x_s = -1 + 2 * ((x_original - self.x_min) / (self.x_max-self.x_min))
            return x_s
        else: 
            x_s = -1 + 2 * (x_original - self.x_min[0,dim_id]) / (self.x_max[0,dim_id] - self.x_min[0,dim_id])
            return x_s
    
    def inv_scaler_x(self, x_s, dim_id = -1):
        
        if dim_id == -1:
            x_original = (x_s + 1)*0.5*(self.x_max-self.x_min) + self.x_min
            return x_original
        else: 
            x_original = (x_s + 1)*0.5*(self.x_max[0,dim_id] - self.x_min[0,dim_id]) + self.x_min[0,dim_id]
            return x_original
        
    def scaler_y(self, y_original, dim_id = -1):
        # y_max  = [1,2]
        try:
            if dim_id == -1:
                return -1 + 2 * ((y_original.transpose(1,0) - self.y_min) / (self.y_max-self.y_min))
            else: 
                y_s = -1 + 2 * (y_original.transpose(1,0) - self.y_min[0,dim_id]) / (self.y_max[0,dim_id] - self.y_min[0,dim_id])
                return y_s
        except:
            #print("y_original device:", y_original.device)
            #print("y_min device:", self.y_min.device)
            return -1 + 2 * ((y_original - self.y_min) / (self.y_max-self.y_min))
    
    def inv_scaler_y(self, y_s, dim_id = -1):
            if dim_id == -1:
                return (y_s + 1)*0.5*(self.y_max-self.y_min) + self.y_min
            else:
                y_original = (y_s + 1)*0.5*(self.y_max[0,dim_id] - self.y_min[0,dim_id]) + self.y_min[0,dim_id]
                return y_original

    
    
# ========= Forward ============

    def forward(self, u_hat, u_future_fix, u_past, x_past_fix, x_past):
       
        # convert u_hat into tensor and set u_hat as variable
        u_future_fix = u_future_fix.squeeze(0)
        u_hat = u_hat.squeeze(0)
        #print(u_future_fix.shape, u_hat.shape)
        future_cov = torch.concat((u_future_fix,u_hat), dim = 1).unsqueeze(0)
        # knit past and future covariate into the input format for TiDE

        x_past_fix = x_past_fix.squeeze(0)
        #print(x_past.shape, x_past_fix.shape, u_past.shape)
        past_cov = torch.tensor(torch.concat((x_past,x_past_fix,u_past), dim = 1),dtype=torch.float32).unsqueeze(0)
        
        # TiDE prediction
        x_hat = self.NN_model([past_cov,future_cov,None])   
        
        return  x_hat[0,:,:,1]
        
        
