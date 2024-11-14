import numpy as np
from scipy.stats import norm

# Set random seed for reproducibility
np.random.seed(110124)

# Parameters
S0 = 100.0     # Spot price
K = 100.0      # Strike price
T = 1.0        # Time to maturity in years
r = 0.06       # Risk-free interest rate
q = 0.06       # Continuous dividend yield
sigma = 0.35   # Volatility
N = 4000       # Number of simulated paths
M = 100        # Number of time steps
dt = T / M     # Time increment

# Discount factor
discount_factor = np.exp(-r * T)

# Black-Scholes-Merton (BSM) Price Calculation
d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
call_bsm = S0 * np.exp(-q * T) * norm.cdf(d1) - K * discount_factor * norm.cdf(d2)
print(f"Black-Scholes-Merton (BSM) Price: ${call_bsm:.2f}\n")

# Precompute constants for simulation
nudt = (r - q - 0.5 * sigma**2) * dt
sigmadt = sigma * np.sqrt(dt)

# Arrays to store payoffs
payoffs_plain = np.zeros(N)
payoffs_antithetic = np.zeros(N // 2)
payoffs_cv = np.zeros(N)
ST_cv = np.zeros(N)

# Plain Monte Carlo Simulation
for i in range(N):
    S = S0
    for _ in range(M):
        epsilon = np.random.normal()
        S *= np.exp(nudt + sigmadt * epsilon)
    payoffs_plain[i] = max(S - K, 0)

# Calculate plain MC price and standard error
C_MC = discount_factor * np.mean(payoffs_plain)
SE_MC = discount_factor * np.std(payoffs_plain, ddof=1) / np.sqrt(N)
print(f"Plain Monte Carlo Simulation:")
print(f"Price: ${C_MC:.2f}")
print(f"Standard Error: ${SE_MC:.2f}\n")

# Antithetic Variates Method
for i in range(N // 2):
    S_plus = S0
    S_minus = S0
    for _ in range(M):
        epsilon = np.random.normal()
        Wt = epsilon * np.sqrt(dt)
        S_plus *= np.exp(nudt + sigma * Wt)
        S_minus *= np.exp(nudt - sigma * Wt)
    payoff_plus = max(S_plus - K, 0)
    payoff_minus = max(S_minus - K, 0)
    payoffs_antithetic[i] = (payoff_plus + payoff_minus) / 2

# Calculate antithetic variates price and standard error
C_AV = discount_factor * np.mean(payoffs_antithetic)
SE_AV = discount_factor * np.std(payoffs_antithetic, ddof=1) / np.sqrt(N // 2)
print(f"Antithetic Variates Method:")
print(f"Price: ${C_AV:.2f}")
print(f"Standard Error: ${SE_AV:.2f}\n")

# Control Variate Method
E_ST = S0 * np.exp((r - q) * T)  # Expected value of S_T

for i in range(N):
    S = S0
    for _ in range(M):
        epsilon = np.random.normal()
        S *= np.exp(nudt + sigmadt * epsilon)
    payoffs_cv[i] = max(S - K, 0)
    ST_cv[i] = S

# Estimate the coefficient b
Covariance = np.cov(payoffs_cv, ST_cv, ddof=1)[0, 1]
Variance_ST = np.var(ST_cv, ddof=1)
b = Covariance / Variance_ST

# Adjust payoffs using control variate
adjusted_payoffs_cv = payoffs_cv - b * (ST_cv - E_ST)

# Calculate control variate price and standard error
C_CV = discount_factor * np.mean(adjusted_payoffs_cv)
SE_CV = discount_factor * np.std(adjusted_payoffs_cv, ddof=1) / np.sqrt(N)
print(f"Control Variate Method:")
print(f"Price: ${C_CV:.2f}")
print(f"Standard Error: ${SE_CV:.2f}")
