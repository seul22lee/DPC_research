import numpy as np


A = np.array([[0.3,0.1], [0.1, 0.2]])
B = np.array([[0.5],[1.0]])

def nominal_model(x_k,u_k):
    ''' 
    Nominal model (noise clean), an unstable LTI model from literature.
    
    Arguments:
      - x_k: current state (N by 1)
      - u_k: applied input (1 by 1)

    Output:
      - x_next: output state (N by 1)
    '''
    
    # calculate derivative
    return np.array(A@x_k+B*u_k)

def nominal_model_recur(x_current,u_seq):
    ''' 
    Recurring nominal model.
    
    Arguments:
      - x_current: current state (N by 1) at the first step
      - u_seq: applied input sequence (1 by n_step)

    Output:
      - x_pred_seq: output sequence of future state within the horizon (N by n_step)
    '''
    seq_len = len(u_seq)
    x_pred_seq = np.empty((2,0))
    for i in range(seq_len):
        x_pred_seq_temp = nominal_model(x_current,u_seq[i]).reshape(-1,1)
        x_current = x_pred_seq_temp
        x_pred_seq = np.concatenate((x_pred_seq,x_pred_seq_temp),axis=1)
    return x_pred_seq


def system_model(x_k,u_k):
    ''' 
    System model with noise injected. 
    
    Arguments:
      - x_k: current state (N by 1)
      - u_k: applied input (1 by 1)

    Output:
      - x_next: output state (N by 1)
    '''
    mu = np.array([[0],[0]])
    w = np.array([[0.05],[0.1]])
    #w = np.array([[0],[0]])

    # calculate derivative
    return A@x_k+B*u_k + np.random.normal(mu,w,size=(2,1))