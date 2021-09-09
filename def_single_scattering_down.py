import time
import numpy as np
from numba import jit
from def_legendre_polynomial import legendre_polynomial
#
@jit(nopython=True, cache=True)
def single_scattering_down(mu, mu0, azr, tau0, xk):
    '''
    Task:
        To compute single scattering at bottom of a homogeneous atmosphere.
    In:
        mu     d        cos(vza_up) > 0
        mu0    d        cos(sza) > 0
        azr    d[naz]   relative azimuths in radians; naz = len(azr)
        tau0   d        total atmosphere optical thickness
        xk     d[nk]    expansion moments * ssa/2, (2k+1) included, nk=len(xk)
    Out:
        I11dn  d        Iboa=f(mu, mu0, azr)
    Tree:
        -
    Note:
        TOA scaling factor = 2pi;
    Refs:
        1. -
    '''
#
#   Parameters:
    nk = len(xk)
    tiny = 1.0e-8
#
    smu = np.sqrt(1.0 - mu*mu)
    smu0 = np.sqrt(1.0 - mu0*mu0)
    nu = mu*mu0 + smu*smu0*np.cos(azr)
    p = np.zeros_like(nu)
    for inu, nui in enumerate(nu):
        pk = legendre_polynomial(nui, nk-1)
        p[inu] = np.dot(xk, pk)
#  
    if np.abs(mu - mu0) < tiny:
        I11dn = p*tau0*np.exp(-tau0/mu0)/mu0
    else:
        I11dn = p*mu0/(mu0 - mu)*(np.exp(-tau0/mu0) - np.exp(-tau0/mu))
#
    return I11dn
#==============================================================================
#
if __name__ == "__main__":
#
    tau0 = 1.0/3.0
    xk = np.array([1.0, 0.0, 0.5])
    mu0 = np.linspace(0.1, 1.0, 91)
    mu =  mu0
    azr = np.linspace(0.0, np.pi, 1801)
    nmu0 = len(mu0)
    nmu = len(mu)
    naz = len(azr)
    I1dn = np.zeros((nmu0, nmu, naz))
    time_start = time.time()
    for imu0 in range(len(mu0)):
        for imu in range(len(mu)):
            I1dn[imu0, imu, :] = single_scattering_down(mu[imu], mu0[imu0], azr, tau0, xk)
    time_end = time.time()
#
    print("sglsup = %.3f sec."%(time_end-time_start))
#==============================================================================