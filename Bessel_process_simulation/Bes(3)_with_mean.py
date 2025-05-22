import numpy as np
import matplotlib.pyplot as plt


# Simulation parameters
a = 1.0          # boundary radius
R0 = 0.2         # initial radius
dt = 1e-4        
max_steps = int(1e6)  # maximum number of steps per path
N_paths = 500    # number of paths to average for mean R(t)
N_plot = 5       # number of sample paths to plot

# Precompute time grid up to T_max = max_steps * dt
T_max = max_steps * dt
times = np.linspace(0, T_max, max_steps+1)

def simulate_path(R0, a, dt, max_steps):
    """Simulate one BES(3) path up to max_steps, return array of R values."""
    R = np.empty(max_steps+1)
    R[0] = R0
    for i in range(1, max_steps+1):
        if R[i-1] >= a:
            R[i:] = a
            break
        dW = np.sqrt(dt) * np.random.randn()
        R[i] = R[i-1] + (1.0 / R[i-1]) * dt + dW
        # enforce positivity
        if R[i] <= 0:
            R[i] = abs(R[i])
    return R

# 1) Simulate N_paths to compute mean trajectory
all_paths = np.zeros((N_paths, max_steps+1))
for n in range(N_paths):
    all_paths[n] = simulate_path(R0, a, dt, max_steps)
mean_R = all_paths.mean(axis=0)

# 2) Plot a few sample paths + mean
plt.figure(figsize=(8,5))

# Plot sample paths
for i in range(N_plot):
    plt.plot(times, all_paths[i], lw=1, alpha=0.7, label=f'Path {i+1}' if i<1 else "")

# Plot mean trajectory
plt.plot(times, mean_R, color='C3', lw=2, label=r'$\langle R(t)\rangle$')

# Boundary
plt.axhline(a, linestyle='--', color='k', label=r'$R=a$')

plt.xlim(0, 1)  # zoom into early times if it helps
plt.xlabel('Time $t$')
plt.ylabel('$R(t)$')
plt.title('BES(3): Sample Paths and Mean Trajectory')
plt.legend()
plt.tight_layout()
plt.show()
