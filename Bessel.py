# Reproduce BES(3) process applied to the narrow escape problem
# with full spherical simmetry (escape at a given radius)

# Theoretical mean for radial diffusion in 3D: E[tau] = (a^2 - R0^2) / 3*D
# Assume unit diffusion D=1 


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Simulation parameters
a = 1.0          # boundary radius
R0 = 0.2         # initial radius
dt = 1e-4        
max_steps = int(1e6)  # maximum number of steps
N = 1000         # number simulations (paths)

# Bessel(3) process until hiting the boundary
def simulate_hitting_time(R0, a, dt, max_steps):
    R = R0
    t = 0.0
    for _ in range(max_steps):
        if R >= a:
            return t
        # Euler-Maruyama step for BES(3): dR = 1/R dt + dW
        R += (1.0 / R) * dt + np.sqrt(dt) * np.random.randn()
        # Bessel must be positive
        
        if R <= 0:
            R = abs(R)
        t += dt
    return np.nan  # if the particle didn't reach the boundary we assume infinte time

# Run simulation for N paths
taus = np.array([simulate_hitting_time(R0, a, dt, max_steps) for _ in range(N)])
taus = taus[~np.isnan(taus)]
mean_tau_sim = np.mean(taus)

# Theoretical mean for radial diffusion in 3D: E[tau] = (a^2 - R0^2) / 3
mean_tau_theory = (a**2 - R0**2) / 3


### sample paths ###

# Plot sample paths
plt.figure()
for i in range(N):
    t_path = 0.0
    path = [R0]
    for _ in range(int(5*a**2/dt)):  # simulate for time ~5*a^2
        R = path[-1]
        if R >= a:
            break
        R += (1.0 / R) * dt + np.sqrt(dt) * np.random.randn()
        R = max(R, 1e-6)
        path.append(R)
        t_path += dt
    plt.plot(np.linspace(0, t_path, len(path)), path)
plt.axhline(a, linestyle='--')
plt.xlabel('Time')
plt.ylabel('R(t)')
plt.title('Sample Paths of BES(3) Until Hitting Radius a')
plt.show()



### only histogram ###

# Plot histogram of hitting times
plt.figure()
plt.hist(taus, bins=30, density=True)
plt.xlabel('Hitting Time τ')
plt.ylabel('Density')
plt.title(f'Hitting Time Distribution (Mean ~ {mean_tau_sim:.3f})')
plt.show()

# Print results
print(f"Simulated mean hitting time: {mean_tau_sim:.4f}")
print(f"Theoretical mean hitting time: {mean_tau_theory:.4f}")




### histogram and density curves ###

# Plot histogram + density curve
plt.figure(figsize=(8,5))
# 1) histogram (normalized)
counts, bins, patches = plt.hist(taus, bins=30, density=True, alpha=0.6,
                                 edgecolor='k', label='Empirical density (hist)')

# 2) kernel density estimate
kde = gaussian_kde(taus)
t_vals = np.linspace(bins[0], bins[-1], 200)
plt.plot(t_vals, kde(t_vals), lw=2, label='Empirical density (KDE)')

# 3) annotate means in the upper right
textstr = (
    f"Simulated mean: {mean_tau_sim:.4f}\n"
    f"Theoretical mean: {mean_tau_theory:.4f}"
)
# place a text box in upper right in axes coords
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
plt.text(0.95, 0.75, textstr, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=props)

plt.xlabel('Hitting Time τ')
plt.ylabel('Density')
plt.title('Hitting Time Distribution with Density Curve')
plt.legend()
plt.tight_layout()
plt.show()