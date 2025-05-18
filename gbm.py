import numpy as np
import matplotlib.pyplot as plt

def gbm():
    # Parameters of both equations
    T = 1.0 # Total time
    N = 1000 # Number of time steps
    dt = T / N # Time step size
    t = np.linspace(0.0, T, N+1) # Time vector
    mu = 0.1 # Expected return
    v0 = 0.1 # Initial volatility
    kappa = 3.0 # Mean reversion rate
    theta = 0.2 # Long-term average volatility
    sigma = 0.1 # Volatility

    # Generate random shocks - random fluctuations
    dW1 = np.random.randn(N) * np.sqrt(dt)
    dW2 = np.random.randn(N) * np.sqrt(dt)

    # Initialize arrays for stock price and volatility
    S = np.zeros(N+1)
    v = np.zeros(N+1)
    S[0] = 100.0 # Initial stock price
    v[0] = v0 # Initial volatility

    # Euler-Maruyama method to solve the stochastic differential equation for stock price dynamics
    for i in range(1, N+1):
        v[i] = v[i-1] + kappa * (theta - v[i-1]) * dt + sigma * np.sqrt(v[i-1]) * dW2[i-1]
        S[i] = S[i-1] * (1 + mu * dt + np.sqrt(v[i-1]) * dW1[i-1])

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, S)
    plt.title('Stock Price Dynamics')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(t, v)
    plt.title('Volatility Dynamics')
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    gbm()