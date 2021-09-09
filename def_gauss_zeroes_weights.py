import time
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
#
@jit(nopython=True, cache=True)
def gauss_zeroes_weights(x1, x2, n):
    '''
    Task:
        To compute 'n' Gauss nodes and weights within (x1, x2).
    In:
        x1, x2   d       interval
        n        i       number of nodes
    Out:
        x, w     d[n]    zeros and weights
    Tree:
        -
    Notes:
        Tested only for x1 < x2. To test run, e.g.,
        ng = 64
        x, w = gauszw(-1.0, 1.0, ng)
        and compare vs [1].
    Refs:
        1. https://pomax.github.io/bezierinfo/legendre-gauss.html
    '''
    const_yeps = 3.0e-14
    x = np.zeros(n)
    w = np.zeros(n)
    m = int((n+1)/2)
    yxm = 0.5*(x2 + x1)
    yxl = 0.5*(x2 - x1)
    for i in range(m):
        yz = np.cos(np.pi*(i + 0.75)/(n + 0.5))
        while True:
            yp1 = 1.0
            yp2 = 0.0
            for j in range(n):
                yp3 = yp2
                yp2 = yp1
                yp1 = ((2.0*j + 1.0)*yz*yp2 - j*yp3 )/(j+1)
            ypp = n*(yz*yp1 - yp2)/(yz*yz - 1.0)
            yz1 = yz
            yz = yz1 - yp1/ypp
            if (np.abs(yz - yz1) < const_yeps):
                break # exit while loop
        x[i] = yxm - yz*yxl
        x[n-1-i] = yxm + yxl*yz
        w[i] = 2.0*yxl/((1.0 - yz*yz)*ypp*ypp)
        w[n-1-i] = w[i]
    return x, w
#==============================================================================
#
if __name__ == "__main__":
#
#
    size = 24
    n = 10
    n2 = n*2
    zs, ws = gauss_zeroes_weights(-1.0, 1.0, n)
    zd = np.zeros(n2)
    wd = np.zeros(n2)
    zd = (zs + 1.0)/2.0
    wd = ws/2.0
    zdzd = np.concatenate((np.flip(-zd), zd))
    wdwd = np.concatenate((wd, wd))
    z2, w2 = gauss_zeroes_weights(-1.0, 1.0, n2)
    fig = plt.figure()
    plt.plot(z2, w2, color=(0.5, 0.5, 0.5))
    plt.plot(-zdzd, wdwd, color=(0.75, 0.75, 0.75))
    plt.plot(z2, w2, 'or', -zdzd, wdwd, 'ob')
    plt.grid(True)
    plt.xlabel('Zeros', fontsize=18)
    plt.ylabel('Weights', fontsize=18)    
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
#    plt.title('Gaussian quadratures', size=16)
    fig.set_size_inches((12, 6))
    plt.tight_layout()
#    plt.legend(['Single', 'Double'], fontsize=16)
    plt.savefig('gauss.tiff', dpi=600)
    
    
    z, w = gauss_zeroes_weights(0.0, 1.0, n)
    for i in range(n):
        print(w[i] - wd[i], z[i] - zd[i])
#
    n = 1024
    time_start = time.time()
    zn, wn = gauss_zeroes_weights(-1.0, 1.0, n)
    time_end = time.time()
#
    print("gauszw runtime = %.3f sec."%(time_end-time_start))
#==============================================================================