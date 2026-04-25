
import torch
from torchmin import minimize as pytorch_minimize
from scipy.optimize import minimize, Bounds
import time
import numpy as np
import copy




class GAMMA_MPC():
    def __init__(self,
                 GAMMA_class,                 # GAMMA class that has been initialized and run through the first sim_interval*window iterations
                 TiDE,                        # TiDE model that includes scalers and forward function
                 MPC_obj_fun,                 # Objective function of MPC. Should be compatible with TiDE
                 x_ref_all,                   # Reference trajectory for x (melt pool temperature)
                 window,                      # window size
                 P,                           # Horizon size
                 fix_cov_all,                 # fix covariates (original scale) from the beginning to the end. It should be fixed once the toolpath is given
                 x_past,                      # past melt pool temperature using MPC timesteps, in original scale [n_feature, W]
                 u_past,                      # past laser power using MPC timesteps.
                 ):
        
        x_past_in = x_past[-window:]
        u_past_in = u_past[-window:,0].reshape(-1,1)
        
        
        self.GAMMA = GAMMA_class
        self.TiDE = TiDE
        self.window = window
        self.obj = MPC_obj_fun
        self.ref = x_ref_all
        self.P = P
        self.fix_cov_all = fix_cov_all
        self.x_past = x_past_in
        self.u_past = u_past_in
        self.MPC_counter = GAMMA_class.init_runs
        self.x_hat_current = x_past[:,-1] #[2,1]
        self.x_sys_current = x_past[:,-1]
        self.instant_pred = None
        
        # save data
        self.x_past_save = x_past.transpose(1,0)
        self.u_past_save = u_past.reshape(-1,1)
        self.NN_pred_save = x_past.transpose(1,0)
        self.save_time = []
        
        # For PID specifically
        self.PID_error_past = 0
        self.PID_dt = 0.0355
        self.PID_error_integral = 0
        
        self.PID_Kp = 0.1
        self.PID_Ki = 0.01
        self.PID_Kd = 0#0.01
        
        return None
    
    def MPC_run_one_step_pytorch(self):
        
        # select the counter part of reference
        mp_temp_ref = self.ref[self.MPC_counter:self.MPC_counter + self.P] # shape = [50]
        # scale reference
        mp_temp_ref_part_s = self.TiDE.scaler_y(mp_temp_ref.reshape(1,-1),0) # shape = [50,1]
        # scale past features
        mp_temp_past_part_s = self.TiDE.scaler_y(self.x_past) # shape = [50,2]
  
        # select past fix covariate
        fix_cov_past = self.fix_cov_all[self.MPC_counter-self.window:self.MPC_counter,:]
        
        # scale past fix covariate
        # for all the covariates
        # fix_cov_past_s = TiDE.scaler_x(fix_cov_past,dim_id=[0,1,2,3,4,5,6])
        # for partial covariates
        fix_cov_past_s = self.TiDE.scaler_x(fix_cov_past,dim_id=[0,1,2])
        
        # select future fix covariate
        fix_cov_future = self.fix_cov_all[self.MPC_counter:self.MPC_counter+self.P,:]
        # scale future fix covariate
        # for all the covariates
        # fix_cov_future_s = TiDE.scaler_x(fix_cov_future,dim_id=[0,1,2,3,4,5,6])
        # for partial covariates
        fix_cov_future_s = self.TiDE.scaler_x(fix_cov_future,dim_id=[0,1,2])
        # scale past laser power input
        past_laser_power_s = self.TiDE.scaler_x(self.u_past, dim_id=[3])
        
        
        # Adjust reference if there is a layer switch
        # if torch.any(fix_cov_future[6,:]==0):
        #     is_zero = fix_cov_future[6,:] == 0
        #     is_zero_int = is_zero.int()
        #     cut_off_indicator = torch.argmax(is_zero_int)
        #     print(cut_off_indicator)
        #     mp_temp_ref_part_s[cut_off_indicator:] = mp_temp_ref_part_s[cut_off_indicator-1]

        # optimization
        time1 = time.time()
        solution_s = pytorch_minimize(lambda u:self.obj(u,fix_cov_future_s,past_laser_power_s,fix_cov_past_s, mp_temp_past_part_s,mp_temp_ref_part_s,self.P,self.TiDE),torch.zeros((self.P,1)),method="l-bfgs")                             
        time2 = time.time()
    
        solution_s_x = torch.clamp(solution_s.x,min=-1, max=1)
        
        # scale solution to original scale
        solution = self.TiDE.inv_scaler_x(solution_s_x, dim_id = [3])
        
        
        # predict MP temp
        mp_hat_opt_s = self.TiDE.forward(solution_s_x, fix_cov_future_s, past_laser_power_s, fix_cov_past_s, mp_temp_past_part_s) # predicted MP temp, [50,2]
        # scale MP temp to original scale
        mp_hat_opt = self.TiDE.inv_scaler_y(mp_hat_opt_s) # [50,2]
        self.instant_pred = mp_hat_opt
        
        # apply anciliary controller
        rmpc = 1
        if rmpc:
            K_ac = -0.05
            e = self.x_sys_current[0] - self.x_hat_current[0]
            u_applied = float(solution[0]) + float(K_ac*e)
            x_current, depth_current = self.GAMMA.run_sim_interval(u_applied)
        
        else:
        # simulate environment
            x_current, depth_current = self.GAMMA.run_sim_interval(float(solution[0]))
            u_applied = solution[0]
            
        # saturation:
        if u_applied >= self.TiDE.x_max[0][3]:
            u_applied = self.TiDE.x_max[0][3]
        if u_applied <= self.TiDE.x_min[0][3]:
            u_applied = self.TiDE.x_min[0][3]
                
   
        # update past
        self.x_past[:,0:-1] = copy.deepcopy(self.x_past[:,1:]) # shape [2,50] 
        self.x_past[0,-1] = torch.tensor(x_current,dtype=torch.float32)
        self.x_past[1,-1] = torch.tensor(depth_current,dtype=torch.float32)
   
        self.u_past[0:-1] = copy.deepcopy(self.u_past[1:])
        self.u_past[-1] = torch.tensor(u_applied)
        
        self.save_time.append(time1-time2)
        
        self.x_hat_current = mp_hat_opt[0,:] # [1,2]
        self.x_sys_current = torch.tensor([[x_current],[depth_current]])
        
        self.MPC_counter += 1
               
        # save data
        
        self.x_past_save = torch.concatenate((self.x_past_save,torch.tensor([x_current,depth_current]).reshape(1,-1)))
        self.u_past_save = torch.concatenate((self.u_past_save,copy.deepcopy(self.u_past[-1].reshape(-1,1))))
        self.NN_pred_save = torch.concatenate((self.NN_pred_save,mp_hat_opt[0,:].reshape(1,-1)))
        

        return None
    
    def MPC_run_one_step_scipy(self):
        
        # select the counter part of reference
        mp_temp_ref = self.ref[self.MPC_counter:self.MPC_counter + self.P] 
        # scale reference
        mp_temp_ref_part_s = self.TiDE.scaler_y(mp_temp_ref)
        # scale past temperature
        mp_temp_past_part_s = self.TiDE.scaler_y(self.x_past).transpose(1,0)
        # select past fix covariate
        fix_cov_past = self.fix_cov_all[self.MPC_counter-self.window:self.MPC_counter,:]
        # scale past fix covariate
        fix_cov_past_s = self.TiDE.scaler_x(fix_cov_past,dim_id=[0,1,2,3,4,5,6])
        # select future fix covariate
        fix_cov_future = self.fix_cov_all[self.MPC_counter:self.MPC_counter+self.P,:]
        # scale future fix covariate
        fix_cov_future_s = self.TiDE.scaler_x(fix_cov_future,dim_id=[0,1,2,3,4,5,6])
        # scale past laser power input
        past_laser_power_s = self.TiDE.scaler_x(self.u_past, dim_id = [6])
        
        # Adjust reference if there is a layer switch
        fix_cov_future_ref = self.fix_cov_all[self.MPC_counter:self.MPC_counter+self.P + 5,:]
       
        # optimization
        bounds = Bounds(lb=np.ones(self.P)*-1, ub=np.ones(self.P))
        solution_s = minimize(lambda u:self.obj(u,fix_cov_future_s,past_laser_power_s,fix_cov_past_s, mp_temp_past_part_s,mp_temp_ref_part_s,P,TiDE),np.zeros((P)),method="slsqp",jac=True, bounds=bounds)
        # scale solution to original scale
        if solution_s.success == False:
            print(f"not success on iteration {self.MPC_counter}")
            
        solution_s_torch = torch.tensor(solution_s.x.reshape(-1,1),dtype=torch.float32)
        solution = self.TiDE.inv_scaler_y(solution_s_torch)
        
        # predict MP temp
        mp_hat_opt_s = self.TiDE.forward(solution_s_torch, fix_cov_future_s, past_laser_power_s, fix_cov_past_s, mp_temp_past_part_s) # predicted MP temp
        # scale MP temp to original scale
        mp_hat_opt = self.TiDE.inv_scaler_y(mp_hat_opt_s)
        
        # simulate environment
        x_current = self.GAMMA.run_sim_interval(float(solution[0]))
        
        # update past
        self.x_past[0:-1] = copy.deepcopy(self.x_past[1:])
        self.x_past[-1] = x_current
        
        self.u_past[0:-1] = copy.deepcopy(self.u_past[1:])
        self.u_past[-1] = solution[0]
                    
        self.MPC_counter += 1

        # save data
        self.x_past_save = torch.concatenate((self.x_past_save,copy.deepcopy(self.x_past[-1].reshape(-1,1))))
        self.u_past_save = torch.concatenate((self.u_past_save,copy.deepcopy(self.u_past[-1].reshape(-1,1))))
        self.NN_pred_save = torch.concatenate((self.NN_pred_save,mp_hat_opt.squeeze()[0].reshape(-1,1)))
        # save data is problematic and need to be checked 

        return None
    
    
    
    def PID_run_one_step(self):
        
        # PID gain
        # self.PID_Kp = 0.1   #0.1
        # self.PID_Ki = 0.01  #0.01
        # self.PID_Kd = 0
        
        # assign PID in some initialized function
        MPC_counter = self.MPC_counter
        ref = self.ref
        x_sys_current = self.x_sys_current
        
        # compute error: e = r[i] - y_current        
        x_sys_current = self.x_sys_current     # take x_current
        ref_current = ref[MPC_counter]         # take current reference
        error_P = ref_current - x_sys_current    # compute the error
                    
        # compute integral of error in discrete time: integral += e * dt
        self.PID_error_integral += error_P*self.PID_dt    # the integral is saved in self
        error_I = self.PID_error_integral
            
        # compute derivative in discrete time: derivative = (e - e_prev) / dt
            # need to call e_prev from self
        error_D = (error_P - self.PID_error_past) / self.PID_dt
        
        
        # compute the output with PID gain
            # u = optimal_Kp * e + optimal_Ki * integral + optimal_Kd * derivative
        u_applied = (self.PID_Kp * error_P) + (self.PID_Ki * error_I) + (self.PID_Kd * error_D) + self.u_past_save[-1]
        # run GAMMA simulation
        x_current = self.GAMMA.run_sim_interval(float(u_applied))
        
        # print(f"step = {self.MPC_counter}, u = {u_applied}, e_P = {error_P}, e_I = {error_I}, e_D = {error_D}")
        # update saved value
        self.PID_error_integral = error_P
        self.x_sys_current = x_current
        self.u_past_save = torch.concatenate((self.u_past_save,copy.deepcopy(u_applied.reshape(-1,1))))
        self.x_past_save = torch.concatenate((self.x_past_save,torch.tensor(x_current.reshape(-1,1))))
        self.MPC_counter += 1
        return None