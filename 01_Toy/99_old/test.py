import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# System matrices
A = np.array([[1.0, 1.0],
              [0.0, 1.0]])
B = np.array([[0.0],
              [1.0]])

n = A.shape[0]
m = B.shape[1]
N = 5  # Horizon

# Initial state
x0 = np.array([[0.0], [0.0]])

# Reference trajectory for x1
x1_ref = np.array([1.0, 0.5, 0.0, -0.5, -1.0])

# Time-varying constraints for x and u (shape: [n, N])
x_lower = np.array([
    [-1.0, -1.2, -1.5, -1.2, -1.0],  # lower bound for x1
    [-1.0, -1.0, -1.0, -1.0, -1.0]   # lower bound for x2
])
x_upper = np.array([
    [ 1.0,  1.2,  1.5,  1.2,  1.0],  # upper bound for x1
    [ 1.0,  1.0,  1.0,  1.0,  1.0]   # upper bound for x2
])

u_lower = np.array([[-1.0, -0.8, -0.5, -0.8, -1.0]])
u_upper = np.array([[ 1.0,  0.8,  0.5,  0.8,  1.0]])

# Variables
x = cp.Variable((n, N+1))
u = cp.Variable((m, N))

# Objective
cost = cp.sum_squares(x[0, 1:] - x1_ref)

# Constraints
constraints = [x[:, 0] == x0.flatten()]
for t in range(N):
    constraints += [x[:, t+1] == A @ x[:, t] + B @ u[:, t]]
    constraints += [x_lower[:, t] <= x[:, t+1], x[:, t+1] <= x_upper[:, t]]
    constraints += [u_lower[:, t] <= u[:, t], u[:, t] <= u_upper[:, t]]

# Solve
prob = cp.Problem(cp.Minimize(cost), constraints)
prob.solve()

# Results
print("Optimal control input u:")
print(u.value)

print("\nOptimal state x:")
print(x.value)

# Plotting
t = np.arange(N+1)
plt.figure(figsize=(8,4))
plt.plot(t, x.value[0, :], label="x1")
plt.plot(t, x.value[1, :], label="x2")
plt.plot(np.arange(1, N+1), x1_ref, 'k--', label="x1_ref")
plt.fill_between(np.arange(1, N+1), x_lower[0], x_upper[0], color='gray', alpha=0.2, label="x1 bounds")
plt.xlabel("Time")
plt.ylabel("State")
plt.legend()
plt.grid(True)
plt.title("State Trajectory with Time-Varying Constraints")
plt.tight_layout()
plt.show()
