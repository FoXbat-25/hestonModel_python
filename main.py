from gbm import gbm
from hcf import *
import time

def main():

    start = time.time()
    gbm()
    call_price = heston_call_price(S0, K, r, T, kappa, theta, sigma, rho, v0)
    put_price = heston_put_price(S0, K, r, T, kappa, theta, sigma, rho, v0)

    print("European Call Option Price:", np.round(call_price, 2))
    print("European Put Option Price:", np.round(put_price, 2))

    print(f"Process took {time.time()-start} seconds.")

if __name__ == "__main__":
    main()