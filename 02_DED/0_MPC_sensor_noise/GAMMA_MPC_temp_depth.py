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
                 LossParam = None,
                 global_state = None,
                 # ===== (changed) 측정 노이즈 인자 추가 — 전부 기본값이라 기존 호출 그대로 동작 =====
                 add_meas_noise = True,             # (changed) False면 노이즈 끔 → 원본과 동일 동작
                 meas_noise_mode = "absolute",      # (changed) "absolute" 또는 "relative"
                 meas_noise_std = (20.0, 0.005),    # (changed) (temp[K], depth[mm]) 절대 std
                 meas_noise_rel = (0.01, 0.02),     # (changed) (temp, depth) 상대 std(값의 비율)
                 noise_seed = 1234,                 # (changed) 재현성용 seed. None이면 매번 랜덤
                 # ===== (added) 입력(레이저파워) disturbance 인자 — 전부 기본값이라 기존 호출 그대로 동작 =====
                 add_input_disturbance = False,     # (added) True면 plant에 들어가는 입력에 disturbance 주입
                 disturbance_fn = None,             # (added) callable(step, u_cmd, rng) -> d  (원본 스케일[W])
                 disturbance_seed = None,           # (added) 랜덤 disturbance 재현용 seed
                 disturbance_in_history = False,    # (added) False=컨트롤러는 disturbance 모름(미측정),
                                                    #         True=교란된 실제 입력을 u_past에 기록(측정됨)
                 clip_plant_input = False           # (added) True면 교란된 입력을 물리 한계로 클리핑
                 ):
        
        self.iteration_log = []
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
        self.LossParam = LossParam
        self.global_state = global_state

        # ===== (changed) 측정 노이즈 설정 저장 + 전용 RNG 생성 =====
        self.add_meas_noise  = add_meas_noise                     # (changed)
        self.meas_noise_mode = meas_noise_mode                    # (changed)
        self.meas_noise_std  = meas_noise_std                     # (changed)
        self.meas_noise_rel  = meas_noise_rel                     # (changed)
        self.meas_rng        = np.random.default_rng(noise_seed)  # (changed) 전용 generator (global numpy 안 건드림)

        # ===== (added) 입력 disturbance 설정 + 전용 RNG =====
        self.add_input_disturbance  = add_input_disturbance          # (added)
        self.disturbance_fn         = disturbance_fn                 # (added)
        self.disturbance_in_history = disturbance_in_history         # (added)
        self.clip_plant_input       = clip_plant_input               # (added)
        self.dist_rng               = np.random.default_rng(disturbance_seed)  # (added)

        self.warm_start = torch.ones((P,1))*-0.8
        
        # save data
        self.x_past_save = x_past.transpose(1,0)
        self.x_past_save_actual = x_past.transpose(1,0)
        self.u_past_save = u_past.reshape(-1,1)
        # ===== (added) 입력 disturbance 분석용 저장 배열 =====
        # u_past_save        : 컨트롤러가 기록하는 입력(=명령값, 미측정 가정시) — 기존과 동일 의미
        # u_cmd_save         : 컨트롤러 명령값(saturation 후)
        # u_plant_save       : 실제 plant(GAMMA)에 들어간 입력 (= u_cmd + disturbance)
        # disturbance_save   : 주입된 disturbance d
        self.u_cmd_save       = u_past.reshape(-1,1)                 # (added)
        self.u_plant_save     = u_past.reshape(-1,1)                 # (added)
        self.disturbance_save = torch.zeros_like(u_past.reshape(-1,1))  # (added)
        self.NN_pred_save = x_past.transpose(1,0)
        self.NN_pred_save_temp_ub = torch.empty(1,P)
        self.NN_pred_save_temp_med = torch.empty(1,P)
        self.NN_pred_save_temp_lb = torch.empty(1,P)
        self.NN_pred_save_depth_ub = torch.empty(1,P)
        self.NN_pred_save_depth_med = torch.empty(1,P)
        self.NN_pred_save_depth_lb = torch.empty(1,P)
        self.save_time = []
        
        # For PID specifically
        self.PID_error_past = 0
        self.PID_dt = 0.0355
        self.PID_error_integral = 0
        
        self.PID_Kp = 0.1
        self.PID_Ki = 0.01
        self.PID_Kd = 0#0.01
        
        return None

    # ===== (added) 측정 노이즈 헬퍼 — add_meas_noise=True일 때만 호출됨 =====
    def _add_meas_noise(self, temp, depth):                              # (added)
        """GAMMA 참값(temp, depth)에 변수별 독립 가우시안 노이즈를 더해 반환."""    # (added)
        t, d = float(temp), float(depth)                                # (added)
        if self.meas_noise_mode == "relative":                          # (added)
            t_std, d_std = abs(t)*self.meas_noise_rel[0], abs(d)*self.meas_noise_rel[1]  # (added)
        else:                                                           # (added)
            t_std, d_std = self.meas_noise_std[0], self.meas_noise_std[1]                # (added)
        t_noisy = t + float(self.meas_rng.normal(0.0, t_std))           # (added)
        d_noisy = d + float(self.meas_rng.normal(0.0, d_std))           # (added)
        return t_noisy, d_noisy                                         # (added)

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
        solution_s = pytorch_minimize(lambda u:self.obj(u,fix_cov_future_s,past_laser_power_s,fix_cov_past_s, mp_temp_past_part_s,mp_temp_ref_part_s,self.P,self.TiDE),self.warm_start,method="l-bfgs")                           
                
        n_iter = solution_s.nit
        self.iteration_log.append(n_iter)

        #print(f"[Step {self.MPC_counter}] Warm-start: {n_iter} iterations")

        # CSV에 한 줄 추가
        with open("iteration_log.csv", mode='a', newline='') as file:
            import csv
            writer = csv.writer(file)
            writer.writerow([self.MPC_counter, n_iter, "warm"])

        time2 = time.time()
        
        self.global_state.reset_global_state()
        self.LossParam.reset()
            
        if solution_s.success != True or solution_s.x[0]==self.warm_start[0]:
            solution_s = pytorch_minimize(lambda u:self.obj(u,fix_cov_future_s,past_laser_power_s,fix_cov_past_s, mp_temp_past_part_s,mp_temp_ref_part_s,self.P,self.TiDE),torch.ones((self.P,1))*-0.8,method="l-bfgs")                           
            time2 = time.time()

            n_iter = solution_s.nit
            self.iteration_log.append(n_iter)

            #print(f"[Step {self.MPC_counter}] Cold-start fallback: {n_iter} iterations")

            # CSV에 한 줄 추가
            with open("iteration_log.csv", mode='a', newline='') as file:
                import csv
                writer = csv.writer(file)
                writer.writerow([self.MPC_counter, n_iter, "cold"])

            self.global_state.reset_global_state()
            self.LossParam.reset()

        # ========= use sin function for gradient projection =======
        # self.warm_start = solution_s.x
        # solution_s_x = torch.sin(solution_s.x)
    
        # ========= use clamping to bound input variables ===========
        solution_s_x = torch.clamp(solution_s.x,min=-1, max=1)
        self.warm_start = solution_s_x
        
        # scale solution to original scale
        solution = self.TiDE.inv_scaler_x(solution_s_x, dim_id = [3])
        
        
        # predict MP temp
        mp_hat_opt_s, mp_hat_opt_all_s = self.TiDE.forward(solution_s_x, fix_cov_future_s, past_laser_power_s, fix_cov_past_s, mp_temp_past_part_s) # predicted MP temp, [50,2]
        # scale MP temp to original scale
        mp_hat_opt = self.TiDE.inv_scaler_y(mp_hat_opt_s) # [50,2]
        mp_hat_opt_lb = self.TiDE.inv_scaler_y(mp_hat_opt_all_s[:,:,0])
        mp_hat_opt_ub = self.TiDE.inv_scaler_y(mp_hat_opt_all_s[:,:,2])
        self.instant_pred = mp_hat_opt
        
        # apply anciliary controller  → u_cmd (컨트롤러 명령값)
        rmpc = 1
        if rmpc:
            K_ac = -0.05*0
            e = self.x_sys_current[0] - self.x_hat_current[0]
            u_cmd = float(solution[0]) + float(K_ac*e)
        else:
            u_cmd = float(solution[0])

        # ===== (added) 입력 disturbance 주입 =====
        # 실제 plant에 들어가는 입력 = 명령값 + disturbance.
        # disturbance_fn(step, u_cmd, rng) -> d  (원본 스케일 [W]). 미지정/off면 d=0.
        d = 0.0
        if self.add_input_disturbance and (self.disturbance_fn is not None):
            d = float(self.disturbance_fn(self.MPC_counter, u_cmd, self.dist_rng))
        u_plant = u_cmd + d

        # (옵션) 교란된 입력을 물리 한계로 클리핑
        if self.clip_plant_input:
            u_plant = min(max(u_plant, float(self.TiDE.x_min[0][3])), float(self.TiDE.x_max[0][3]))

        # simulate environment — plant은 교란된 입력 u_plant 를 받는다
        x_current, depth_current = self.GAMMA.run_sim_interval(u_plant)

        # ===== (changed) 측정 노이즈 주입 =====
        x_true, depth_true = x_current, depth_current          # (changed) GAMMA 참값(ground truth) 보관
        if self.add_meas_noise:                                # (changed) OFF면 아래 건너뜀 → x_current/depth_current 원본 그대로
            x_current, depth_current = self._add_meas_noise(x_true, depth_true)  # (changed) 노이즈 낀 측정값으로 교체

        # saturation: 컨트롤러 명령값(u_cmd)을 액추에이터 한계로 포화 → 기록/이력에 사용
        u_applied = u_cmd
        if u_applied >= self.TiDE.x_max[0][3]:
            u_applied = self.TiDE.x_max[0][3]
        if u_applied <= self.TiDE.x_min[0][3]:
            u_applied = self.TiDE.x_min[0][3]

        # 컨트롤러 이력(u_past)에 기록할 값 선택:
        #  - disturbance_in_history=False(기본): 명령값(u_applied) → 컨트롤러는 disturbance를 모름(미측정)
        #  - disturbance_in_history=True       : 실제 입력(u_plant) → 교란을 측정해 모델에 반영
        u_hist = u_plant if self.disturbance_in_history else u_applied

        # update past
        self.x_past[:,0:-1] = copy.deepcopy(self.x_past[:,1:]) # shape [2,50] 
        self.x_past[0,-1] = torch.tensor(x_current,dtype=torch.float32)
        self.x_past[1,-1] = torch.tensor(depth_current,dtype=torch.float32)
        #self.x_past[1,-1] = torch.tensor(mp_hat_opt[0,1],dtype=torch.float32)
   
        self.u_past[0:-1] = copy.deepcopy(self.u_past[1:])
        self.u_past[-1] = torch.tensor(u_hist)
        
        self.save_time.append(time1-time2)
        #print(f"Iteration {self.MPC_counter} - Time taken for optimization: {time2-time1:.4f} seconds") 
        
        self.x_hat_current = mp_hat_opt[0,:] # [1,2]
        self.x_sys_current = torch.tensor([[x_current],[depth_current]])
        
        self.MPC_counter += 1
               
        # save data
        
        self.x_past_save = torch.concatenate((self.x_past_save,torch.tensor([x_current,depth_current]).reshape(1,-1)))
        #self.x_past_save = torch.concatenate((self.x_past_save,torch.tensor([x_current,mp_hat_opt[0,1]]).reshape(1,-1)))
        # (changed) 참값(노이즈 없는 ground truth) 저장 — 저장 전용이라 제어 동역학엔 영향 없음
        self.x_past_save_actual = torch.concatenate((self.x_past_save_actual,torch.tensor([x_true,depth_true]).reshape(1,-1)))  # (changed)
        self.u_past_save = torch.concatenate((self.u_past_save,copy.deepcopy(self.u_past[-1].reshape(-1,1))))
        # ===== (added) 입력 disturbance 분석용 기록 =====
        self.u_cmd_save       = torch.concatenate((self.u_cmd_save,       torch.tensor([[u_applied]],dtype=torch.float32)))  # (added)
        self.u_plant_save     = torch.concatenate((self.u_plant_save,     torch.tensor([[u_plant]],  dtype=torch.float32)))  # (added)
        self.disturbance_save = torch.concatenate((self.disturbance_save, torch.tensor([[d]],        dtype=torch.float32)))  # (added)
        self.NN_pred_save = torch.concatenate((self.NN_pred_save,mp_hat_opt[0,:].reshape(1,-1)))
        
        self.NN_pred_save_temp_ub = torch.concatenate((self.NN_pred_save_temp_ub,mp_hat_opt_ub[:,0].reshape(1,-1)))
        self.NN_pred_save_temp_med = torch.concatenate((self.NN_pred_save_temp_med,mp_hat_opt[:,0].reshape(1,-1)))
        self.NN_pred_save_temp_lb = torch.concatenate((self.NN_pred_save_temp_lb,mp_hat_opt_lb[:,0].reshape(1,-1)))
        
        self.NN_pred_save_depth_ub = torch.concatenate((self.NN_pred_save_depth_ub,mp_hat_opt_ub[:,1].reshape(1,-1)))
        self.NN_pred_save_depth_med = torch.concatenate((self.NN_pred_save_depth_med,mp_hat_opt[:,1].reshape(1,-1)))
        self.NN_pred_save_depth_lb = torch.concatenate((self.NN_pred_save_depth_lb,mp_hat_opt_lb[:,1].reshape(1,-1)))
        

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
        #self.PID_error_past = error_P
        self.x_sys_current = x_current
        self.u_past_save = torch.concatenate((self.u_past_save,copy.deepcopy(u_applied.reshape(-1,1))))
        self.x_past_save = torch.concatenate((self.x_past_save,torch.tensor(x_current.reshape(-1,1))))
        self.MPC_counter += 1
        return None