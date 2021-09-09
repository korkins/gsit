import time
import numpy as np
from numba import jit
from def_legendre_polynomial import legendre_polynomial
#
@jit(nopython=True, cache=True)
def single_scattering_up(mu, mu0, azr, tau0, xk):
    '''
    Task:
        To compute single scattering at top of a homogeneous atmosphere.
    In:
        mu     d        cos(vza_up) < 0
        mu0    d        cos(sza) > 0
        azr    d[naz]   relative azimuths in radians; naz = len(azr)
        tau0   d        total atmosphere optical thickness
        xk     d[nk]    expansion moments * ssa/2, (2k+1) included, nk=len(xk)
    Out:
        I11up  d        Itoa=f(mu, mu0, azr)
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
#
    smu = np.sqrt(1.0 - mu*mu)
    smu0 = np.sqrt(1.0 - mu0*mu0)
    nu = mu*mu0 + smu*smu0*np.cos(azr)
    p = np.zeros_like(nu)
    for inu, nui in enumerate(nu):
        pk = legendre_polynomial(nui, nk-1)
        p[inu] = np.dot(xk, pk)
#  
    mup = -mu
    I11up = p*mu0/(mu0 + mup)*(1.0 - np.exp(-tau0/mup - tau0/mu0))
#
    return I11up
#==============================================================================
#
if __name__ == "__main__":
#
    tau0 = 1.0/3.0
    xk = np.array([1.0, 0.0, 0.5])
    mu0 = np.linspace(0.1, 1.0, 91)
    mu = -mu0
    azr = np.linspace(0.0, np.pi, 1801)
    nmu0 = len(mu0)
    nmu = len(mu)
    naz = len(azr)
    I1up = np.zeros((nmu0, nmu, naz))
    time_start = time.time()
    for imu0 in range(len(mu0)):
        for imu in range(len(mu)):
            I1up[imu0, imu, :] = single_scattering_up(mu[imu], mu0[imu0], azr, tau0, xk)
    time_end = time.time()
#
    print("sglsup = %.3f sec."%(time_end-time_start))
#==============================================================================