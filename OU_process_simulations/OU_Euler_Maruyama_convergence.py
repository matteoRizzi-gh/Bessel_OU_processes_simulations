import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 1.0
sigma = 0.5
T = 1.0  # final time
M = 3000  # increased number of Monte Carlo paths for better statistics
dts = np.logspace(-4, -1, 10)  # extended range to smaller time steps
errors = []

np.random.seed(42)

for dt in dts:
    N = int(T / dt)
    t = np.linspace(0, T, N + 1)

    dW = np.sqrt(dt) * np.random.randn(M, N)
    X0 = np.random.randn(M)

    # Euler-Maruyama
    X_em = np.zeros((M, N + 1))
    X_em[:, 0] = X0
    for n in range(N):
        X_em[:, n+1] = X_em[:, n] - alpha * X_em[:, n] * dt + sigma * dW[:, n]

    # Exact solution using the same dW
    # This implementation is correct for strong convergence analysis,
    # as it aligns with the Ito integral's discrete approximation.
    X_exact = np.zeros((M, N + 1))
    X_exact[:, 0] = X0
    for n in range(1, N + 1):
        X_exact[:, n] = X0 * np.exp(-alpha * t[n])
        # Your original stochastic integral sum
        for k in range(n):
            X_exact[:, n] += sigma * np.exp(-alpha * (t[n] - t[k])) * dW[:, k]

    # Compute strong error at final time
    error = np.mean(np.abs(X_em[:, -1] - X_exact[:, -1]))
    errors.append(error) # This is the ONLY place to append 'error'

# Plot strong error convergence
plt.figure(figsize=(6, 5))
plt.loglog(dts, errors, 'b-', label='Strong error')
ref_line = errors[0] * (dts / dts[0])**0.5  # slope 0.5 reference line
plt.loglog(dts, ref_line, 'r--', label='Slope 0.5 reference')
plt.xlabel(r'$\Delta t$')
plt.ylabel(r'$\mathbb{E}[|X_{EM}(T) - X_{exact}(T)|]$')
plt.title('Strong convergence of Euler–Maruyama for OU process')
plt.legend()
plt.grid(True, which="both", ls=':')
plt.tight_layout()
plt.show()