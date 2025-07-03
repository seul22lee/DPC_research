
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from scipy.optimize import minimize, Bounds
from sys_model import nominal_model, nominal_model_recur, system_model
from scipy.stats import qmc

class RMPC_iteration:
    
    def __init__(self,
                 u_hat,           # initial guess of u_hat
                 x_past,          # initial guess of x
                 u_past,          # initial guess of u_past
                 window,          # window length for x and u. if include_u_hist==True, window = 1 for x
                 P,               # length of horizon
                 x_current,       # initial condition of x
                 K_ac,            # gain of ancillary controller
                 sys_params,      # including min and max for x and u
                 optim_obj,       # Objective function. the default setting is using SLSQP with jac=True
                 include_u_hist,  # include longer history of u but only one step of x
                 NN_model,        # TiDE model
                 NN_forward,      # forward function for NN
                 error_past=None, # initial value of error history
                 constraint=None,
                 tube_model=None,
                 u_tighten=None
                 ) -> None:
        
        # Initialize values
        self.u_hat = u_hat
        self.x_past = x_past
        self.u_past = u_past
        self.x_current = x_current
        self.x_sys_current = x_current
        self.x_hat_current = x_current
        self.K_ac = K_ac
        self.sys_params = sys_params
        self.window = window
        self.P = P
        self.optim_obj = optim_obj
        self.include_u_hist = include_u_hist
        self.NN_model = NN_model
        self.NN_forward = NN_forward
        self.error_past = error_past
        self.constraint = constraint
        self.tube_model = tube_model
        self.u_tighten = u_tighten
        self.count = 0

        self.x_max = sys_params["x_max"]
        self.x_min = sys_params["x_min"]
        self.u_max = sys_params["u_max"]
        self.u_min = sys_params["u_min"]

        # Initialize saving vectors
        self.e_save = np.empty((2,0))
        self.real_output = np.empty((2,0))
        self.nominal_output = np.empty((2,0))
        self.nominal_u = np.empty((0,1))
        self.u_applied_save = np.empty((0,1))
        self.true_pred_output_x1_save = np.empty((0,P))
        self.true_pred_output_x2_save = np.empty((0,P))
        self.x_hat_horizon_save = np.empty((0,P))
        self.error_past_x1_upper = np.empty((0,P))
        self.error_past_x1_lower = np.empty((0,P))
        self.error_past_x2_lower = np.empty((0,P))
        self.error_past_x2_upper = np.empty((0,P))

        

        return None
    
    def run_one_step(self,
                     SP_hat,  # reference trajectory. Length should be P
                     sys_noise = True,
                     RMPC = True,
                     ):
        
        self.count = self.count + 1

        x_past_s = -1 + 2 * ((self.x_past - self.x_min.reshape(-1,1)) / (self.x_max-self.x_min).reshape(-1,1)) 
        u_past_s = -1 + 2 * ((self.u_past - self.u_min.reshape(-1,1)) / (self.u_max-self.u_min).reshape(-1,1))
        SP_hat_s = -1 + 2 * ((SP_hat - self.x_min[0].reshape(-1,1)) / (self.x_max[0]-self.x_min[0]).reshape(-1,1))

        if self.include_u_hist:
            x_past_s[:,:-1] = 0
        
        try:
            if self.error_past == None:
                self.error_past = np.zeros((2,self.window))
        except:
            pass
       
        if self.constraint == None:
            constraint = None
        else:
            constraint = [{'type':'ineq',
                           'fun':lambda u: self.constraint(u, self.u_past, self.tube_model,self.error_past, self.x_max, self.x_min, x_past_s, self.u_max, self.u_min, self.NN_model)[0],
                           'jac':lambda u: self.constraint(u, self.u_past, self.tube_model,self.error_past, self.x_max, self.x_min, x_past_s, self.u_max, self.u_min, self.NN_model)[1]}]
            # constraint = [{'type':'ineq',
            #                'fun':lambda u: self.constraint(u, self.u_past, self.tube_model,self.error_past, self.x_max, self.x_min, x_past_s, self.u_max, self.u_min, self.NN_model)[0]}]

        if self.u_tighten == None:
            bounds = Bounds(lb=np.ones(self.P)*-1, ub=np.ones(self.P)*1)
        else:
            lb, ub = self.u_tightening(lb=np.ones(self.P)*-1, ub=np.ones(self.P)*1)
            lb = -1 + 2 * ((lb - self.u_min.reshape(-1)) / (self.u_max-self.u_min).reshape(-1))
            ub = -1 + 2 * ((ub- self.u_min.reshape(-1)) / (self.u_max-self.u_min).reshape(-1))
            bounds = Bounds(lb,ub)
        
        solution = minimize(self.optim_obj, self.u_hat, method = "SLSQP", args=(u_past_s, x_past_s, SP_hat_s, self.P, self.NN_model),jac=True, bounds=bounds, constraints = constraint, options={'maxiter':600,'gtol':1e-3})
        
        if solution.success == False:
            solution = minimize(self.optim_obj, np.zeros((self.P)), method = "SLSQP", args=(u_past_s, x_past_s, SP_hat_s, self.P, self.NN_model),jac=True, bounds=bounds, constraints = constraint, options={'maxiter':600,'gtol':1e-3})
            print(f"cannot obtain a valid solution at step {self.count}, {solution.message}")
            if solution.success == False:
                # generate sobol sampling from -1 to 1
                n_sobol = 20
                sobol_sequence = qmc.Sobol(self.P).random(n=n_sobol)
                sobol_sequence_scaled = 2 * sobol_sequence - 1
                obj_value = []
                solution_save = []
                obj_value.append(solution.fun)
                solution_save.append(solution)
                for i in range(n_sobol):
                    solution = minimize(self.optim_obj, sobol_sequence[i,:], method = "SLSQP", args=(u_past_s, x_past_s, SP_hat_s, self.P, self.NN_model),jac=True, bounds=bounds, constraints = constraint, options={'maxiter':300,'gtol':1e-3})
                    print(f"============= backup optimization at step {self.count}, {solution.message}")
                    obj_value.append(solution.fun)
                    solution_save.append(solution)
                max_ind = np.argmin(obj_value)
                solution = solution_save[max_ind]
                  
        u_opt_original_scale = (solution.x + 1).reshape(-1,1)*0.5*(self.u_max-self.u_min).reshape(-1,1) + self.u_min.reshape(-1,1)
    
            
        # Nominal Model
        TiDE_output, _ = self.NN_forward(solution.x,u_past_s,x_past_s,SP_hat_s,self.P,self.NN_model)
        TiDE_output = TiDE_output.detach().numpy()[0,:].reshape(-1,1)  # only the first element 
        x_hat_next = (TiDE_output + 1).reshape(-1,1) * 0.5 * ((self.x_max-self.x_min).reshape(-1,1)) + self.x_min.reshape(-1,1) # include states for x1 and x2
        self.nominal_output = np.concatenate((self.nominal_output,x_hat_next),axis=1)
        self.nominal_u = np.concatenate((self.nominal_u,u_opt_original_scale[0].reshape(-1,1)))

        # Nominal prediction in horizon
        TiDE_output_for_save = self.NN_forward(solution.x,u_past_s,x_past_s,SP_hat_s,self.P,self.NN_model)[0].detach().numpy()
        x_hat_horizon = ((TiDE_output_for_save + 1).reshape(-1,2) * 0.5 * ((self.x_max-self.x_min).reshape(-1,2)) + self.x_min.reshape(-1,2))[:,0].reshape(1,-1)
        self.x_hat_horizon_save = np.concatenate((self.x_hat_horizon_save,x_hat_horizon),axis=0)

        # Correct solution from the state space model
        true_pred_output = nominal_model_recur(self.x_sys_current,u_opt_original_scale)
        self.true_pred_output_x1_save = np.concatenate((self.true_pred_output_x1_save,true_pred_output[0,:].reshape(1,-1)),axis=0)
        self.true_pred_output_x2_save = np.concatenate((self.true_pred_output_x2_save,true_pred_output[1,:].reshape(1,-1)),axis=0)

        
        # System Model
        if RMPC:
            e = self.x_sys_current - self.x_hat_current
            self.e_save = np.concatenate((self.e_save,e),axis=1)
            u_applied = u_opt_original_scale[0] + self.K_ac@e
           
            # Saturation
            if u_applied >= 5:
                u_applied = np.ones_like(u_applied)*5
            if u_applied <= -5:
                u_applied = np.ones_like(u_applied)*-5

        else:
            e = self.x_sys_current - self.x_hat_current
            self.e_save = np.concatenate((self.e_save,e),axis=1)
            u_applied = u_opt_original_scale[0].reshape(-1,1) 

        if sys_noise:
            x_sys_next = system_model(self.x_sys_current,u_applied)
        else:
            x_sys_next = nominal_model(self.x_sys_current,u_applied)
        
        # save data
        self.real_output = np.concatenate((self.real_output,x_sys_next),axis=1)
        self.u_applied_save = np.concatenate((self.u_applied_save,u_applied),axis=0)
        
        # update past covariates for TiDE
        self.x_past[:,:-1] = self.x_past[:,1:]        
        self.x_past[:,-1] = x_sys_next.reshape(-1)

        self.error_past[:,:-1] = self.error_past[:,1:]
        self.error_past[:,-1] = e.reshape(-1)

        if self.include_u_hist:
            self.x_past[:,:-1] = 0

        self.u_past[:,:-1] = self.u_past[:,1:]
        self.u_past[:,-1]  = u_applied

        self.u_hat = solution.x
        self.x_hat_current = x_hat_next
        self.x_sys_current = x_sys_next

        # save data for visualizing single step
        self.true_pred_output = true_pred_output
        self.NN_prediction = x_hat_horizon
        self.latest_solution = solution
        
        
    def run_one_step_w_penalty(self,
                     SP_hat,  # reference trajectory. Length should be P
                     sys_noise = True,
                     RMPC = True,
                     ):
        
        self.count = self.count + 1

        x_past_s = -1 + 2 * ((self.x_past - self.x_min.reshape(-1,1)) / (self.x_max-self.x_min).reshape(-1,1)) 
        u_past_s = -1 + 2 * ((self.u_past - self.u_min.reshape(-1,1)) / (self.u_max-self.u_min).reshape(-1,1))
        SP_hat_s = -1 + 2 * ((SP_hat - self.x_min[0].reshape(-1,1)) / (self.x_max[0]-self.x_min[0]).reshape(-1,1))

        if self.include_u_hist:
            x_past_s[:,:-1] = 0
        
        try:
            if self.error_past == None:
                self.error_past = np.zeros((2,self.window))
        except:
            pass
       
        if self.constraint == None:
            constraint = None
        else:
            constraint = [{'type':'ineq',
                           'fun':lambda u: self.constraint(u, self.u_past, self.tube_model,self.error_past, self.x_max, self.x_min, x_past_s, self.u_max, self.u_min, self.NN_model)[0],
                           'jac':lambda u: self.constraint(u, self.u_past, self.tube_model,self.error_past, self.x_max, self.x_min, x_past_s, self.u_max, self.u_min, self.NN_model)[1]}]
            # constraint = [{'type':'ineq',
            #                'fun':lambda u: self.constraint(u, self.u_past, self.tube_model,self.error_past, self.x_max, self.x_min, x_past_s, self.u_max, self.u_min, self.NN_model)[0]}]

        if self.u_tighten == None:
            bounds = Bounds(lb=np.ones(self.P)*-1, ub=np.ones(self.P)*1)
        else:
            lb, ub = self.u_tightening(lb=np.ones(self.P)*-1, ub=np.ones(self.P)*1)
            lb = -1 + 2 * ((lb - self.u_min.reshape(-1)) / (self.u_max-self.u_min).reshape(-1))
            ub = -1 + 2 * ((ub- self.u_min.reshape(-1)) / (self.u_max-self.u_min).reshape(-1))
            bounds = Bounds(lb,ub)
        
        solution = minimize(self.optim_obj, self.u_hat, method = "SLSQP", args=(u_past_s, x_past_s, SP_hat_s, self.P, self.NN_model, self.tube_model, self.error_past, self.x_max, self.x_min, self.u_max, self.u_min),jac=True, bounds=bounds, constraints = constraint, options={'maxiter':600,'gtol':1e-3})
        
        if solution.success == False:
            solution = minimize(self.optim_obj, np.zeros((self.P)), method = "SLSQP", args=(u_past_s, x_past_s, SP_hat_s, self.P, self.NN_model, self.tube_model, self.error_past, self.x_max, self.x_min, self.u_max, self.u_min),jac=True, bounds=bounds, constraints = constraint, options={'maxiter':600,'gtol':1e-3})
            print(f"cannot obtain a valid solution at step {self.count}, {solution.message}")
            if solution.success == False:
                # generate sobol sampling from -1 to 1
                n_sobol = 20
                sobol_sequence = qmc.Sobol(self.P).random(n=n_sobol)
                sobol_sequence_scaled = 2 * sobol_sequence - 1
                obj_value = []
                solution_save = []
                obj_value.append(solution.fun)
                solution_save.append(solution)
                for i in range(n_sobol):
                    solution = minimize(self.optim_obj, sobol_sequence[i,:], method = "SLSQP", args=(u_past_s, x_past_s, SP_hat_s, self.P, self.NN_model),jac=True, bounds=bounds, constraints = constraint, options={'maxiter':300,'gtol':1e-3})
                    print(f"============= backup optimization at step {self.count}, {solution.message}")
                    obj_value.append(solution.fun)
                    solution_save.append(solution)
                max_ind = np.argmin(obj_value)
                solution = solution_save[max_ind]
                
           
        u_opt_original_scale = (solution.x + 1).reshape(-1,1)*0.5*(self.u_max-self.u_min).reshape(-1,1) + self.u_min.reshape(-1,1)
        
        
        
        # Nominal Model
        TiDE_output = self.NN_forward(solution.x,u_past_s,x_past_s,SP_hat_s,self.P,self.NN_model).detach().numpy()[0,:].reshape(-1,1)  # only the first element 
        x_hat_next = (TiDE_output + 1).reshape(-1,1) * 0.5 * ((self.x_max-self.x_min).reshape(-1,1)) + self.x_min.reshape(-1,1) # include states for x1 and x2
        self.nominal_output = np.concatenate((self.nominal_output,x_hat_next),axis=1)
        self.nominal_u = np.concatenate((self.nominal_u,u_opt_original_scale[0].reshape(-1,1)))

        # Nominal prediction in horizon
        TiDE_output_for_save = self.NN_forward(solution.x,u_past_s,x_past_s,SP_hat_s,self.P,self.NN_model).detach().numpy()
        x_hat_horizon = ((TiDE_output_for_save + 1).reshape(-1,2) * 0.5 * ((self.x_max-self.x_min).reshape(-1,2)) + self.x_min.reshape(-1,2))[:,0].reshape(1,-1)
        self.x_hat_horizon_save = np.concatenate((self.x_hat_horizon_save,x_hat_horizon),axis=0)

        # Correct solution from the state space model
        true_pred_output = nominal_model_recur(self.x_sys_current,u_opt_original_scale)
        self.true_pred_output_x1_save = np.concatenate((self.true_pred_output_x1_save,true_pred_output[0,:].reshape(1,-1)),axis=0)
        self.true_pred_output_x2_save = np.concatenate((self.true_pred_output_x2_save,true_pred_output[1,:].reshape(1,-1)),axis=0)

        
        # System Model
        if RMPC:
            e = self.x_sys_current - self.x_hat_current
            self.e_save = np.concatenate((self.e_save,e),axis=1)
            u_applied = u_opt_original_scale[0] + self.K_ac@e
           
            # Saturation
            if u_applied >= 1:
                u_applied = np.ones_like(u_applied)
            if u_applied <= -1:
                u_applied = np.ones_like(u_applied)*-1

        else:
            e = self.x_sys_current - self.x_hat_current
            self.e_save = np.concatenate((self.e_save,e),axis=1)
            u_applied = u_opt_original_scale[0].reshape(-1,1) 

        if sys_noise:
            x_sys_next = system_model(self.x_sys_current,u_applied)
        else:
            x_sys_next = nominal_model(self.x_sys_current,u_applied)
        
        # save data
        self.real_output = np.concatenate((self.real_output,x_sys_next),axis=1)
        self.u_applied_save = np.concatenate((self.u_applied_save,u_applied),axis=0)
        
        # update past covariates for TiDE
        self.x_past[:,:-1] = self.x_past[:,1:]        
        self.x_past[:,-1] = x_sys_next.reshape(-1)

        self.error_past[:,:-1] = self.error_past[:,1:]
        self.error_past[:,-1] = e.reshape(-1)

        if self.include_u_hist:
            self.x_past[:,:-1] = 0

        self.u_past[:,:-1] = self.u_past[:,1:]
        self.u_past[:,-1]  = u_applied

        self.u_hat = solution.x
        self.x_hat_current = x_hat_next
        self.x_sys_current = x_sys_next

        # save data for visualizing single step
        self.true_pred_output = true_pred_output
        self.NN_prediction = x_hat_horizon
        self.latest_solution = solution

    def u_tightening(self, lb, ub):
        # all operated under original scale
        u_hat = torch.tensor(self.u_hat, dtype=torch.float32, requires_grad=False).reshape(1,-1,1)
        u_past = torch.tensor(self.u_past,dtype=torch.float32, requires_grad=False).reshape(1,-1,1)
        error_past = torch.tensor(self.error_past, dtype=torch.float32, requires_grad=False).reshape(-1,2).unsqueeze(0)
        

        past_series = torch.concat((error_past, u_past), dim=2)

        with torch.no_grad():
            error_hat = self.tube_model([past_series,u_hat,None])[0,:,:,:] # scaled pred median of x1 and x2, shape = [10,2,3]

        # upper bound 
        error_upper_x1 = error_hat[:,0,2].reshape(1,-1)
        error_upper_x2 = error_hat[:,1,2].reshape(1,-1)
        u_upper = (ub + self.K_ac@torch.concat((error_upper_x1,error_upper_x2),dim=0).numpy()).reshape(-1)

        # lower bound
        error_lower_x1 = error_hat[:,0,0].reshape(1,-1)
        error_lower_x2 = error_hat[:,1,0].reshape(1,-1)
        u_lower = (lb + self.K_ac@torch.concat((error_lower_x1,error_lower_x2),dim=0).numpy()).reshape(-1)

        # save error prediction
        self.error_past_x1_upper = np.concatenate((self.error_past_x1_upper,error_upper_x1),axis=0)
        self.error_past_x2_upper = np.concatenate((self.error_past_x2_upper,error_upper_x2),axis=0)
        self.error_past_x2_lower = np.concatenate((self.error_past_x2_lower,error_lower_x2),axis=0)
        self.error_past_x1_lower = np.concatenate((self.error_past_x1_lower,error_lower_x1),axis=0)

        return u_lower, u_upper


# ==================================================
# Plots
# ==================================================

    def Plot_standard(self,Ref_traj,n_steps,plot_traj_steps,show_constraints=False):
        # Creating subplots
        grid = plt.GridSpec(2,2,wspace=0.2,hspace=0.3)
        plt.figure(figsize=(10,7))

        # First subplot
        plt.subplot(grid[0,0:])
        plt.plot(Ref_traj[:plot_traj_steps], linewidth=3,label="Reference")
        plt.plot(self.real_output[0,:plot_traj_steps], label="Output trajectory")
        if show_constraints:
            plt.plot(np.ones(plot_traj_steps)*4,"--", label="ub")
            plt.plot(np.ones(plot_traj_steps)*-4,"--", label="lb")
        plt.xlabel("time")
        plt.ylabel("y")
        plt.legend()
        plt.suptitle(f"$\kappa$={self.K_ac}",fontsize=15)
        

        # pred error hist
        plt.subplot(grid[1,0])
        u_qt, l_qt = np.quantile(self.e_save[0,:],[0.99,0.01])
        plt.hist(self.e_save[0,:],50)
        plt.xlabel(r"$y-\bar{y}$")
        plt.ylabel("counts")
        plt.title(f"0.99 quantile = {u_qt:.4f}, 0.01 quantile = {l_qt:.4f}")


        # control error plot
        plt.subplot(grid[1,1])
        control_err = Ref_traj[:n_steps] - self.real_output[0,:]
        mse = ((Ref_traj[:n_steps] - self.real_output[0,:])**2).mean()
        plt.hist(control_err,50)
        plt.xlabel(r"control error")
        plt.ylabel("counts")
        plt.title(f"MSE = {mse:.4f}")
        

        plt.tight_layout()  # Adjust layout to prevent overlapping
        plt.show()


    def Plot_one_step(self, Ref_traj):
        self.run_one_step(Ref_traj)

        plt.plot(Ref_traj,label="reference")
        plt.plot(self.true_pred_output[0,:],"--.",markersize=15, label="true response")
        plt.plot(self.NN_prediction[0,:],label = "TiDE")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Response")
        plt.title("Predictive Response w.r.t Optimized Input")
        plt.show()


    def Plot_ref_only(self,Ref_traj,n_steps):
        plt.figure(figsize=(12,4))
        plt.plot(Ref_traj[:n_steps], linewidth=3,label="Reference")
        plt.plot(self.real_output[0,:], label="Output trajectory")
        plt.xlabel("time")
        plt.ylabel("y")
        plt.legend()
        plt.title("MPC under noiseless environment",fontsize=15)

    
    def Plot_input_output(self,show_constraint=False):
        # Create a new figure and subplot
        plt.figure(figsize=(12,12))

        # Plot the first figure
        plt.subplot(3, 1, 1)  # 3 rows, 1 column, position 1
        plt.plot(self.real_output[0,:], label="x1")
        if show_constraint:
            plt.plot(np.ones_like(self.real_output[0,:])*4,"--", label="x1 upper bound")
            plt.plot(np.ones_like(self.real_output[0,:])*-4,"--", label="x1 lower bound")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("x1")

        # Plot the second figure
        plt.subplot(3, 1, 2)  # 3 rows, 1 column, position 2
        plt.plot(self.real_output[1,:], label="x2")
        if show_constraint:
            plt.plot(np.ones_like(self.real_output[0,:])*3,"--", label="x1 upper bound")
            plt.plot(np.ones_like(self.real_output[0,:])*-3,"--", label="x1 lower bound")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("x2")

        # Plot the third figure
        plt.subplot(3, 1, 3)  # 3 rows, 1 column, position 3
        plt.plot(self.nominal_u,label="nominal u")
        plt.plot(self.u_applied_save,label="applied u")
        if show_constraint:
            plt.plot(np.ones_like(self.nominal_u)*1,"--", label="u upper bound")
            plt.plot(np.ones_like(self.nominal_u)*-1,"--", label="u lower bound")
        plt.xlabel("Time")
        plt.ylabel("u")
        plt.legend()

        # Show the plot
        plt.show()



        