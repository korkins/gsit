import time
import numpy as np
from numba import jit
#
@jit(nopython=True, cache=True)
def legendre_polynomial(x, kmax):
    '''
    Task:
        To compute the Legendre polynomials, Pk(x), for all orders k=0:kmax and a
        single point 'x' within [-1:+1]
    In:
        x      f   abscissa
        kmax   i   maximum order, k = 0,1,2...kmax
    Out:
        pk    [kmax+1]   Legendre polynomials
    Tree:
        -
    Notes:
        The Bonnet recursion formula [1, 2]:
        
        (k+1)P{k+1}(x) = (2k+1)*P{k}(x) - k*P{k-1}(x),                      (1)
        
        where k = 0:K, P{0}(x) = 1.0, P{1}(x) = x.
        For fast summation over k, this index changes first.
    Refs:
        1. https://en.wikipedia.org/wiki/Legendre_polynomials
        2. http://mathworld.wolfram.com/LegendrePolynomial.html
        3. https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.legendre.html
        4. https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_legendre.html
    '''
    nk = kmax+1
    pk = np.zeros(nk)
    if kmax == 0:
        pk[0] = 1.0
    elif kmax == 1:
        pk[0] = 1.0
        pk[1] = x
    else:
        pk[0] = 1.0
        pk[1] = x
        for ik in range(2, nk):
            pk[ik] = (2.0 - 1.0/ik)*x*pk[ik-1] - (1.0 - 1.0/ik)*pk[ik-2]
    return pk
#==============================================================================
#
if __name__ == "__main__":
#
#
    mu = np.linspace(-1.0, 1.0, 1001)
    nmu = len(mu)
    kmax = 2000
    pkmu = np.zeros((nmu, kmax+1))
    time_start = time.time()
    for imu in range(nmu):
        pkmu[imu, :] = legendre_polynomial(mu[imu], kmax)
    time_end = time.time()
#
    print("polleg = %.3f sec."%(time_end-time_start))
#==============================================================================