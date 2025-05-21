import numpy as np
from scipy.integrate import quad

# Model parameters
S0 = 100.0    # Initial stock price
K = 100.0     # Strike price
r = 0.05      # Risk-free rate
T = 1.0       # Time to maturity
kappa = 2.0   # Mean reversion rate
theta = 0.05  # Long-term average volatility
sigma = 0.3   # Volatility of volatility
rho = -0.5    # Correlation coefficient
v0 = 0.05     # Initial volatility

# Define characteristic functions
def heston_characteristic_function(u, S0, r, T, kappa, theta, sigma, rho, v0):
   xi = kappa - rho * sigma * 1j * u
   d = np.sqrt((rho * sigma * 1j * u - xi)**2 - sigma**2 * (-u * 1j - u**2))
   g = (xi - rho * sigma * 1j * u - d) / (xi - rho * sigma * 1j * u + d)
   C = r * 1j * u * T + (kappa * theta) / sigma**2 * ((xi - rho * sigma * 1j * u - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
   D = (xi - rho * sigma * 1j * u - d) / sigma**2 * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
   return np.exp(C + D * v0 + 1j * u * np.log(S0))

# Define functions to compute call and put options prices
def P1_integrand(u):
    phi = heston_characteristic_function(u - 1j, S0, r, T, kappa, theta, sigma, rho, v0)
    return np.real(np.exp(-1j * u * np.log(K)) * phi / (1j * u))

def P2_integrand(u):
    phi = heston_characteristic_function(u, S0, r, T, kappa, theta, sigma, rho, v0)
    return np.real(np.exp(-1j * u * np.log(K)) * phi / (1j * u))

# Final pricing function
def heston_call_price():
    integral_P1, _ = quad(P1_integrand, 0, np.inf, limit=1000)
    integral_P2, _ = quad(P2_integrand, 0, np.inf, limit=1000)

    P1 = 0.5 + integral_P1 / np.pi
    P2 = 0.5 + integral_P2 / np.pi

    return S0 * P1 - K * np.exp(-r * T) * P2

def heston_put_price():
    call = heston_call_price()
    return call - S0 + K * np.exp(-r * T)