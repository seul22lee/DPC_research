import torch
import torch.optim as optim

def pytorch_optim(obj_fn, u_init, lr=0.1, max_iter=200, method="LBFGS"):
    """
    Perform gradient descent using PyTorch's optimizers.
    
    Parameters:
        obj_fn (callable): The objective function to minimize.
        u_init (torch.Tensor): Initial guess for the optimization variable.
        lr (float): Learning rate for the optimizer.
        max_iter (int): Maximum number of iterations for optimization.
    
    Returns:
        torch.Tensor: The optimized variable.
    """
    
    if method == "LBFGS":
    
        def closure():
            lbfgs.zero_grad()
            objective = obj_fn(u)
            objective.backward()
            torch.nn.utils.clip_grad_norm(u, 1)
            print(u.grad)
            return objective
        
        
        # Ensure u_init requires gradient
        u = u_init.clone().detach().requires_grad_(True)
        
        lbfgs = optim.LBFGS([u],
                            history_size = 10,
                            max_iter = 4,
                            line_search_fn="strong_wolfe")
        
        for i in range(max_iter):
            lbfgs.step(closure)
        
        return u

    if method == "SGD":    
        # Ensure u_init requires gradient
        u = u_init.clone().detach().requires_grad_(True)

        # Define the SGD optimizer
        sgd = optim.SGD([u], lr=0.01)  # Adjust learning rate as needed

        # Training loop
        for i in range(max_iter):
            sgd.zero_grad()  # Clear gradients
            objective = obj_fn(u)  # Compute the objective function
            objective.backward()  # Backward pass (compute gradients)
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_([u], 1)
                       
            # Perform optimization step
            sgd.step()

        return u

    if method == "Adam":    
        # Ensure u_init requires gradient
        u = u_init.clone().detach().requires_grad_(True)

        # Define the Adam optimizer
        optimizer = optim.Adam([u], lr=0.01)  # Adjust learning rate as needed

        # Training loop
        for i in range(max_iter):
            optimizer.zero_grad()  # Clear gradients
            objective = obj_fn(u)  # Compute the objective function
            objective.backward()  # Backward pass (compute gradients)
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_([u], 1)
                       
            # Perform optimization step
            optimizer.step()

        return u