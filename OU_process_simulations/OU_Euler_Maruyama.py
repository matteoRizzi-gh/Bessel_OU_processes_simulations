import numpy as np
import matplotlib.pyplot as plt

# OU parameters
alpha = 1.0      # mean reversion rate (b)
sigma = 1.0      # volatility (λ)
T = 20.0         # final time
M = 1000          # number of Monte Carlo paths

# 1) Fixed-step sampling
dt = 0.0001
N = int(T / dt)
t = np.linspace(0, T, N + 1)

# Generate Brownian increments for EM
dW_em = np.sqrt(dt) * np.random.randn(M, N)  # EM noise: sqrt(dt)*Z

# Initialize paths
X0 = 2 * np.random.rand(M)
X_em = np.zeros((M, N + 1))
X_ex = np.zeros((M, N + 1))
X_em[:, 0] = X0
X_ex[:, 0] = X0

# Precompute coefficients for exact solution
exp_at = np.exp(-alpha * dt)
var_ex = (sigma**2 / (2 * alpha)) * (1 - np.exp(-2 * alpha * dt))

for i in range(N):
    # Euler-Maruyama (uses dW_em)
    X_em[:, i + 1] = X_em[:, i] + (-alpha * X_em[:, i]) * dt + sigma * dW_em[:, i]
    
    # Exact solution (uses independent noise)
    Z_ex = np.random.randn(M)  # Correct noise: Z ~ N(0,1)
    X_ex[:, i + 1] = X_ex[:, i] * exp_at + np.sqrt(var_ex) * Z_ex

# Plot sample paths (unchanged)
# ... (same plotting code as before)

colors = ["blue", "green", "orange", "purple", "cyan", "magenta", "yellow", "black", "brown"]

plt.figure(figsize=(8,4))
for j in range(1):
    plt.plot(t, X_em[j], lw=1, label='EM' if j==0 else "", color=colors[j+2])
    plt.plot(t, X_ex[j], '--', lw=1, label='Exact' if j==0 else "", color=colors[j+3])
plt.title("Sample paths OU: Euler–Maruyama vs Exact")
plt.xlabel("t"); plt.ylabel("X(t)")
plt.legend()
plt.show()

# 2) Moments computation (unchanged)
# ... (same moments code)
mean_em = X_em.mean(axis=0)
mean_ex = X_ex.mean(axis=0)
second_mom_em = (X_em**2).mean(axis=0)
second_mom_ex = (X_ex**2).mean(axis=0)

plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(t, mean_em, label="EM mean")
plt.plot(t, mean_ex, "--", label="Exact mean")
plt.title("E[X(t)]")
plt.xlabel("t"); plt.legend()

plt.subplot(122)
plt.plot(t, second_mom_em, label="EM $E[X^2(t)]$")
plt.plot(t, second_mom_ex, "--", label="Exact $E[X^2(t)]$")
plt.title("$E[X_t^2]$")
plt.xlabel("t"); plt.legend()
plt.show()