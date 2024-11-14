import numpy as np

# Parameters
S0 = 100.0     # Spot price
K = 100.0      # Strike price
T = 1.0        # Time to maturity in years
r = 0.06       # Risk-free interest rate
q = 0.06       # Continuous dividend yield
sigma = 0.35   # Volatility
N = 100        # Number of time steps
dt = T / N     # Time increment

# Compute up and down factors
u = np.exp(sigma * np.sqrt(dt))
d = 1 / u

# Risk-neutral probability
p = (np.exp((r - q) * dt) - d) / (u - d)

# Initialize asset price tree
asset_prices = np.zeros((N + 1, N + 1))
asset_prices[0, 0] = S0

for i in range(1, N + 1):
    asset_prices[i, 0] = asset_prices[i - 1, 0] * u
    for j in range(1, i + 1):
        asset_prices[i, j] = asset_prices[i - 1, j - 1] * d

# Initialize option price tree
option_values = np.zeros((N + 1, N + 1))

# Compute option value at maturity
for j in range(N + 1):
    option_values[N, j] = max(asset_prices[N, j] - K, 0)

# Backward induction for option price tree
for i in range(N - 1, -1, -1):
    for j in range(i + 1):
        continuation_value = (p * option_values[i + 1, j] + (1 - p) * option_values[i + 1, j + 1]) * np.exp(- (r - q) * dt)
        exercise_value = asset_prices[i, j] - K
        option_values[i, j] = max(continuation_value, exercise_value)

american_call_price = option_values[0, 0]
print(f"American Call Option Price: ${american_call_price:.4f}")


def american_call_binomial(S0, K, T, r, q, sigma, N):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    asset_prices = np.zeros((N + 1, N + 1))
    asset_prices[0, 0] = S0

    for i in range(1, N + 1):
        asset_prices[i, 0] = asset_prices[i - 1, 0] * u
        for j in range(1, i + 1):
            asset_prices[i, j] = asset_prices[i - 1, j - 1] * d

    option_values = np.zeros((N + 1, N + 1))

    for j in range(N + 1):
        option_values[N, j] = max(asset_prices[N, j] - K, 0)

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            continuation_value = (p * option_values[i + 1, j] + (1 - p) * option_values[i + 1, j + 1]) * np.exp(- (r - q) * dt)
            exercise_value = asset_prices[i, j] - K
            option_values[i, j] = max(continuation_value, exercise_value)

    return option_values[0, 0]

# Base option price
C0 = american_call_binomial(S0, K, T, r, q, sigma, N)

# Delta
dS = 0.01 * S0
C_up = american_call_binomial(S0 + dS, K, T, r, q, sigma, N)
C_down = american_call_binomial(S0 - dS, K, T, r, q, sigma, N)
delta = (C_up - C_down) / (2 * dS)

# Gamma
gamma = (C_up - 2 * C0 + C_down) / (dS ** 2)

# Theta
dT = 1e-4  # Small time change
C_T = american_call_binomial(S0, K, T - dT, r, q, sigma, N)
theta = (C_T - C0) / dT

# Vega
dSigma = 0.01
C_sigma_up = american_call_binomial(S0, K, T, r, q, sigma + dSigma, N)
C_sigma_down = american_call_binomial(S0, K, T, r, q, sigma - dSigma, N)
vega = (C_sigma_up - C_sigma_down) / (2 * dSigma)

# Rho
dr = 0.01
C_r_up = american_call_binomial(S0, K, T, r + dr, q, sigma, N)
C_r_down = american_call_binomial(S0, K, T, r - dr, q, sigma, N)
rho = (C_r_up - C_r_down) / (2 * dr)

print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.4f}")
print(f"Theta: {theta:.4f}")
print(f"Vega: {vega:.4f}")
print(f"Rho: {rho:.4f}")
