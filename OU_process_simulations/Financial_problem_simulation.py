import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# ---
## 1. Parameters Setup
np.random.seed(42)

# OU Process Parameters (for the spread)
theta = 0.5        # Mean reversion speed
mu = 0             # Long-term mean
sigma = 0.2        # Volatility

# Trading Parameters
k = 1.5            # Threshold for trading (in standard deviations)
transaction_cost_rate = 0.001 # 0.1% transaction cost per trade (as a rate)

# Simulation Parameters
T_years = 1.0      # Total time period in years
trading_days_per_year = 252 # Number of trading days in a year
dt = T_years / trading_days_per_year # Daily time step (as fraction of a year)
N = int(T_years / dt) # Number of time steps (will be 252 for 1 year)

# ---
## 2. Simulate OU Process (Spread)
t = np.arange(0, T_years, dt) # Time points in years
X = np.zeros(N)    # Spread process
X[0] = mu          # Start at mean

for i in range(1, N):
    dW = np.random.normal(0, np.sqrt(dt)) # Brownian motion increment
    dX = theta * (mu - X[i-1]) * dt + sigma * dW
    X[i] = X[i-1] + dX

# ---
## 3. Simulate Asset Prices
# Create two correlated assets where spread is OU process
beta = 1.2  # Hedge ratio (often derived from historical regression)

# Asset A (e.g., KO) - simulated as a Geometric Brownian Motion
# Using a fixed initial price for P_A and P_B for consistent notional
initial_P_A = 100.0
P_A_log_returns = np.random.normal(0.0005, 0.01, N) # Daily log returns
P_A = initial_P_A * np.exp(np.cumsum(P_A_log_returns))
P_A[0] = initial_P_A # Ensure first price is exact initial_P_A

# Asset B (e.g., PEP) constructed to maintain the OU spread: X = ln(P_A) - beta * ln(P_B)
# Rearranging: ln(P_B) = (ln(P_A) - X) / beta
# So: P_B = exp((ln(P_A) - X) / beta)
P_B = np.exp((np.log(P_A) - X) / beta)

# ---
## 4. Test for Mean-Reversion
print("--- Mean-Reversion Test (ADF) ---")
adf_result = adfuller(X)
print(f"ADF Statistic: {adf_result[0]:.3f}")
print(f"p-value: {adf_result[1]:.3f}")
if adf_result[1] < 0.05:
    print("Conclusion: Spread is mean-reverting (p-value < 0.05, good for pairs trading)")
else:
    print("Conclusion: Spread is NOT mean-reverting")

# ---
## 5. Implement Trading Strategy
positions = np.zeros(N)      # 1 = long spread, -1 = short spread, 0 = flat
portfolio_pnl = np.zeros(N)  # Cumulative Profit and Loss

# Calculate mean and std from the simulated spread data (for strategy bands)
spread_mean = np.mean(X)
spread_std = np.std(X)

entry_upper = spread_mean + k * spread_std
entry_lower = spread_mean - k * spread_std
exit_level = spread_mean # Exit when spread returns to its mean

for i in range(1, N):
    # Carry over previous day's cumulative PnL
    portfolio_pnl[i] = portfolio_pnl[i-1]

    # Get the position from the previous day
    current_position = positions[i-1]

    # Calculate daily PnL from holding the position before checking for new trades
    if current_position == 1: # Long spread (Long Asset A, Short Asset B)
        daily_pnl = (P_A[i] - P_A[i-1]) - beta * (P_B[i] - P_B[i-1])
        portfolio_pnl[i] += daily_pnl
    elif current_position == -1: # Short spread (Short Asset A, Long Asset B)
        daily_pnl = - (P_A[i] - P_A[i-1]) + beta * (P_B[i] - P_B[i-1])
        portfolio_pnl[i] += daily_pnl

    # Determine the position for the current day
    new_position = current_position
    trade_executed = False # Flag to track if a trade happens today

    # --- Check for Closing Conditions ---
    if current_position == 1 and X[i] >= exit_level:
        new_position = 0 # Close long spread (go flat)
        trade_executed = True
    elif current_position == -1 and X[i] <= exit_level:
        new_position = 0 # Close short spread (go flat)
        trade_executed = True

    # --- Check for Opening Conditions (only if currently flat after checking for closes) ---
    # Prioritize closing existing positions before opening new ones on the same day.
    if new_position == 0:
        if X[i] > entry_upper:
            new_position = -1 # Open short spread (Sell A, Buy B)
            trade_executed = True
        elif X[i] < entry_lower:
            new_position = 1 # Open long spread (Buy A, Sell B)
            trade_executed = True

    # Apply transaction cost if a trade was executed
    if trade_executed:
        # Calculate the notional value of one unit of the spread for transaction cost
        # This assumes trading 1 share of A and beta shares of B at current prices
        notional_value_of_trade = P_A[i] + beta * P_B[i]
        cost = transaction_cost_rate * notional_value_of_trade
        portfolio_pnl[i] -= cost # Deduct cost from PnL

    positions[i] = new_position # Update position for the current day

# ---
## 6. Visualization
plt.figure(figsize=(15, 12))

# Plot 1: Asset Prices
plt.subplot(3, 1, 1)
plt.plot(t, P_A, label='Asset A (e.g., KO)')
plt.plot(t, P_B, label='Asset B (e.g., PEP)')
plt.title('Simulated Asset Prices')
plt.ylabel('Price')
plt.legend()
plt.grid(True, which="both", ls=':')

# Plot 2: OU Spread with Trading Bands
plt.subplot(3, 1, 2)
plt.plot(t, X, label='Spread (X)')
plt.axhline(spread_mean, color='k', linestyle='--', label='Mean')
plt.axhline(entry_upper, color='r', linestyle=':', label=f'Entry Upper (+{k}$\sigma$)')
plt.axhline(entry_lower, color='r', linestyle=':', label=f'Entry Lower (-{k}$\sigma$)')
plt.axhline(exit_level, color='g', linestyle='-.', label='Exit Level (Mean)')
plt.title('Ornstein-Uhlenbeck Spread Process with Trading Bands')
plt.ylabel('Spread Value')
plt.legend()
plt.grid(True, which="both", ls=':')

# Plot 3: Trading Positions and Cumulative PnL
plt.subplot(3, 1, 3)
plt.plot(t, positions, label='Position (1:Long, -1:Short, 0:Flat)', alpha=0.7)
plt.twinx() # Use a second y-axis for PnL
plt.plot(t, portfolio_pnl, color='purple', label='Cumulative PnL')
plt.title('Trading Positions and Portfolio Cumulative PnL')
plt.xlabel('Time (Years)')
plt.ylabel('Cumulative PnL')
plt.legend(loc='upper left')
plt.grid(True, which="both", ls=':')


plt.tight_layout()
plt.show()

# ---
## 7. Performance Metrics
# For total return, we need an initial capital. Let's assume it's the notional value of one spread unit at T=0.
initial_capital_for_spread = initial_P_A + beta * P_B[0] # Notional cost to open 1 unit of spread
if initial_capital_for_spread == 0: # Avoid division by zero if prices are zero
    initial_capital_for_spread = 1.0 # Placeholder for safety

total_return = portfolio_pnl[-1] / initial_capital_for_spread

returns_daily = np.diff(portfolio_pnl)
# Filter out non-trading days' returns if you want to be strict, but diff accounts for 0 changes
# Or, if only interested in returns on trading days, use returns only where positions change or PnL is non-zero
# For simplicity, using all daily diffs, excluding the first element (which is 0)
trading_returns = returns_daily[np.where(np.diff(positions) != 0)] # Returns from actual trades or changes

print("\n--- Strategy Performance ---")
print(f"Total Cumulative PnL: ${portfolio_pnl[-1]:.2f}")
print(f"Total Return (based on initial notional): {total_return*100:.2f}%")

# Calculate Sharpe Ratio
# Only consider returns where the PnL actually changed (i.e., when a position was held or trade occurred)
# Using all daily PnL differences as 'returns' for simplicity for Sharpe.
valid_returns = returns_daily[np.where(returns_daily != 0)] # Only consider days with PnL change
if len(valid_returns) > 1 and np.std(valid_returns) > 0:
    daily_sharpe = np.mean(valid_returns) / np.std(valid_returns)
    annualized_sharpe = daily_sharpe * np.sqrt(trading_days_per_year)
    print(f"Annualized Sharpe Ratio: {annualized_sharpe:.2f}")
else:
    print("Not enough data or no variability in returns to calculate Sharpe Ratio.")


# Max Drawdown (Absolute)
peak_pnl = np.maximum.accumulate(portfolio_pnl)
drawdown = peak_pnl - portfolio_pnl
max_drawdown = np.max(drawdown)
print(f"Max Drawdown (Absolute): ${max_drawdown:.2f}")

# Max Drawdown (as a percentage of peak PnL, if peak is positive)
if np.max(peak_pnl) > 0:
    max_drawdown_percent = max_drawdown / np.max(peak_pnl) * 100
    print(f"Max Drawdown (Percentage of Peak PnL): {max_drawdown_percent:.2f}%")
else:
    print("Max Drawdown (Percentage): Not applicable (no positive peak PnL)")