import numpy as np
import torch
import matplotlib.pyplot as plt
# from frouros.detectors.concept_drift import DDM, DDMConfig, CUSUM, CUSUMConfig
# from frouros.metrics import PrequentialError



# class DriftDetector():
#     def __init__(self, config, alpha=1.0):
#         self.config = config
#         self.detector = None
#         self.metric = PrequentialError(alpha)
        
#         if self.config["type"] == "ddm":
#             self.detector = DDM(DDMConfig(**self.config["params"]))
#         elif self.config["type"] == "cusum":
#             self.detector = CUSUM(config=CUSUMConfig(**self.config["params"]))
#         else:
#             raise ValueError("Unknown drift detector type")

#     def detect(self, pred_error):
#         pred_error = pred_error.detach().cpu().numpy()
#         metric_dev = self.metric(error_value=pred_error)
#         _ = self.detector.update(metric_dev)
#         status = self.detector.status
        
#         return status

#     def reset(self):
#         if self.detector is not None:
#             self.detector.reset()

class DriftDetector():
    def __init__(self, type = 'cusum', k = 0.05, h = 0.25, min_run_length = 50, mu=0):
        self.k = k
        self.h = h
        self.min_run_length = min_run_length
        self.C_plus = 0
        self.C_minus = 0
        self.C_plus_save = []
        self.C_minus_save = []
        self.mu = mu
        self.counter = 0
        self.counter_global = 0
        self.status = {'drift': False, 'run_length':0}
        self.type = type
        self.raise_alert = False
        
    def detect(self, pred_error):
        
        if self.type == 'cusum':
            pred_error = pred_error.detach().cpu().numpy()
            self.C_plus = max(0, self.C_plus + pred_error - self.k)
            self.C_minus = max(0, self.C_minus - pred_error - self.k)
            
            self.counter += 1
            self.counter_global += 1
            self.status['run_length'] = self.counter
            self.C_minus_save.append(self.C_minus)
            self.C_plus_save.append(self.C_plus)
            
            if self.counter >= self.min_run_length:
                if self.C_plus > self.h or self.C_minus > self.h:
                    self.status['drift'] = True
                else:
                    if self.status['drift'] != True:
                        self.status['drift'] = False
                    
                
            return self.status
        
        else: 
            raise TypeError("Unknown drift detector type")
            
        
    def reset(self):
        if self.type == 'cusum':
            self.C_plus = 0
            self.C_minus = 0
            self.counter = 0
            self.status = {'drift': False, 'run_length':0}
        else:
            raise TypeError("Unknown drift detector type")
        
    def plot_CUSUM(self):
        plt.figure(figsize=(10, 6), dpi=80)
        plt.plot(np.linspace(1,self.counter_global,self.counter_global), self.C_plus_save, label='C_plus')
        plt.plot(np.linspace(1,self.counter_global,self.counter_global), np.array(self.C_minus_save)*-1, label='C_minus')
        plt.plot(np.linspace(1,self.counter_global,self.counter_global), self.h * np.ones_like(self.C_plus_save), label='Upper Threshold')
        plt.plot(np.linspace(1,self.counter_global,self.counter_global),-self.h * np.ones_like(self.C_plus_save), label='Lower Threshold')
        plt.xlabel('Time')
        plt.legend()
        
        
        plt.show()
        
        
        