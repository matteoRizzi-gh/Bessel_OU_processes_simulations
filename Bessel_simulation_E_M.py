# Bessel simulation via E-M, for delta up to 3

### IMPORTAN: ###
# it is not possible to use EM for Bessel SDE because there's a singularity in R=0.
# The implementation is a “symmetrized” or “reflected” Euler–Maruyama scheme 
# (sometimes also called the “full‐reflection” or “absolute‐value” Euler scheme)
# for a one‐dimensional positive SDE.
# It makes sense since R in [0, +inf).

# Considering there is bias near zero, which is controlled asymptotically

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

deltas   = np.arange(1.0, 3.1, 0.5) # delta grid
R0 = 0.2 # initial radius
T = 10.0 # total simulation time
dt = 1e-4
N_paths = 500 # simulation paths
N_steps = int(T/dt)
epsilon = np.sqrt(dt)

# Time grid
t = np.linspace(0, T, N_steps+1)


def bse_solver(delta):
    """
    Simulate N_paths of BES(delta) on [0, T] via Euler–Maruyama.
    Returns an array of shape (N_paths, N_steps+1).
    """
    traj = np.zeros((N_paths, N_steps+1))
    traj[:, 0] = R0

    for i in range(N_paths):
        R = R0
        counts=0
        for n in range(1,N_steps+1):
            # Bessel must be positive
            # if R <= 0:
            #     R = 1e-6
            dW = np.sqrt(dt) * np.random.randn()
            drift = (( delta - 1) / (2 * R)) * dt
            R += drift + dW
            if R<0:
                R = abs(R)
                counts +=1
            traj[i , n] = R
    print(counts)
    return traj


# example d
d=2
# Analytic expectation for BES(delta) started at 0
E_analytic = np.sqrt(2*t) * gamma((d+1)/2) / gamma(d/2)

# run simulation
paths = bse_solver(d)

plt.figure()
for i in range(5):
    plt.plot(t, paths[i], label =f"d={d}" if i == 0 else "", alpha=0.6)
plt.xlabel("time")
plt.ylabel("R(t)")
plt.legend()
plt.title("Sample paths of BES(delta) for variuos delta")
plt.show()


# Empirical statistics
E_emp = paths.mean(axis=0)
S_emp = paths.std(axis=0)
print(S_emp)

plt.figure(figsize=(8,5))
# shaded ±1 std band
plt.fill_between(t,
                 E_emp - S_emp,
                 E_emp + S_emp,
                 color='C0', alpha=0.3,
                 label='Empirical $\pm1$ std')
# empirical mean
plt.plot(t, E_emp,  color='C0', label='Empirical mean')
# analytic curve
plt.plot(t, E_analytic, 'k--', label='Analytic $E[R_t]$ (x0=0)')

plt.xlabel("time")
plt.ylabel("R(t)")
plt.title(f"BES({d}): Empirical vs Analytic Expectation")
plt.legend()
plt.tight_layout()
plt.show()

'''
for d in deltas:
    # Analytic expectation for BES(delta) started at 0
    E_analytic = np.sqrt(2*t) * gamma((d+1)/2) / gamma(d/2)

    # run simulation
    paths = bse_solver(d)
    
    # paths figure
    plt.figure()
    t = np.linspace(0, T, N_steps + 1)
    for i in range(5):
        plt.plot(t, paths[i], label =f"d={d}" if i == 0 else "", alpha=0.6)
    plt.xlabel("time")
    plt.ylabel("R(t)")
    plt.legend()
    plt.title("Sample paths of BES(delta) for variuos delta")
    plt.show()

    # Empirical statistics
    E_emp = paths.mean(axis=0)
    S_emp = paths.std(axis=0)
    print(d, " ", S_emp)

    plt.figure(figsize=(8,5))
    # shaded ±1 std band
    plt.fill_between(t,
                    E_emp - S_emp,
                    E_emp + S_emp,
                    color='C0', alpha=0.3,
                    label='Empirical $\pm1$ std')
    # empirical mean
    plt.plot(t, E_emp,  color='C0', label='Empirical mean')
    # analytic curve
    plt.plot(t, E_analytic, 'k--', label='Analytic $E[R_t]$ (x0=0)')

    plt.xlabel("time")
    plt.ylabel("R(t)")
    plt.title(f"BES({d}): Empirical vs Analytic Expectation")
    plt.legend()
    plt.tight_layout()
    plt.show()

'''    