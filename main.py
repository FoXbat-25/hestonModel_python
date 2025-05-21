from gbm import gbm
from hcf import *
import time

def main():

    start = time.time()

    call_price = heston_call_price()
    put_price = heston_put_price()

    print("European Call Option Price:", np.round(call_price, 2))
    print("European Put Option Price:", np.round(put_price, 2))

    print(f"Process took {time.time()-start} seconds.")

if __name__ == "__main__":
    main()