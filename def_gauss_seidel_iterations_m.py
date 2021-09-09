import time
import numpy as np
from numba import jit
from def_gauss_zeroes_weights import gauss_zeroes_weights
from def_legendre_polynomial import legendre_polynomial
from def_schmidt_polynomial import schmidt_polynomial
#
@jit(nopython=True, cache=True)
def gauss_seidel_iterations_m(m, mu0, srfa, nit, ng1, nlr, dtau, xk):
    '''
    Task:
        Solve RTE in a basic scenario using Gauss-Seidel (GS) iterations.
    In:
        m      i       Fourier moment: m = 0, 1, 2, ... len(xk)-1
        mu0    d       cos(SZA) > 0
        srfa   d       Lambertian surface albedo
        nit    i       number of iterations, nit > 0
        ng1    i       number of gauss nodes per hemisphere
        nlr    i       number of layer elements dtau, tau0 = dtau*nlr
        dtau   d       thickness of element layer (integration step over tau)
        xk     d[nk]   expansion moments*ssa/2, (2k+1) included, nk=len(xk)   
    Out:
        mug, wg    d[ng1*2]        Gauss nodes & weights
        Iup, Idn   d[nlr+1, ng1]   intensity, I = f(tau), at Gauss nodes
    Tree:
        gsit()
            > gauszw() - computes Gauss zeros and weights
            > polleg() - computes ordinary Legendre polynomilas Pk(x)
            > polqkm() - computes *renormalized* associated Legendre polynomilas Qkm(x)
    Note:
        TOA scaling factor = 2pi;
    Refs:
        1. -
    '''
#
#   Parameters:
    tiny = 1.0e-8
    nb = nlr+1
    nk = len(xk)
    ng2 = ng1*2
    tau0 = nlr*dtau
    tau = np.linspace(0.0, tau0, nb)
#
#   Gauss nodes and weights: mup - positive Gauss nodes; mug - all Gauss nodes:
    mup, w = gauss_zeroes_weights(0.0, 1.0, ng1)
    mug = np.concatenate((-mup, mup))
    wg = np.concatenate((w, w))
#
    pk = np.zeros((ng2, nk))
    p = np.zeros(ng2)
    if m == 0:
        pk0 = legendre_polynomial(mu0, nk-1)
        for ig in range(ng2):
            pk[ig, :] = legendre_polynomial(mug[ig], nk-1)
            p[ig] = np.dot(xk, pk[ig, :]*pk0)
    else:
        pk0 = schmidt_polynomial(m, mu0, nk-1)
        for ig in range(ng2):
            pk[ig, :] = schmidt_polynomial(m, mug[ig], nk-1)
            p[ig] = np.dot(xk, pk[ig, :]*pk0)     
#
#   SS down:
    I11dn = np.zeros(ng1)
    for ig in range(ng1):
        mu = mup[ig]
        if (np.abs(mu0 - mu) < tiny):
            I11dn[ig] = p[ng1+ig]*dtau*np.exp(-dtau/mu0)/mu0
        else:
            I11dn[ig] = p[ng1+ig]*mu0/(mu0 - mu)*(np.exp(-dtau/mu0) - np.exp(-dtau/mu))
    I1dn = np.zeros((nb, ng1))
    I1dn[1, :] = I11dn
    for ib in range(2, nb):
        I1dn[ib, :] = I1dn[ib-1, :]*np.exp(-dtau/mup) + I11dn*np.exp(-tau[ib-1]/mu0)
#
#   SS up:
    I11up = p[0:ng1]*mu0/(mu0 + mup)*(1.0 - np.exp(-dtau/mup - dtau/mu0))
    I1up = np.zeros_like(I1dn)
    if m == 0 and srfa > tiny:
        I1up[nb-1, :] = 2.0*srfa*mu0*np.exp(-tau0/mu0)
        I1up[nb-2, :] = I1up[nb-1, :]*np.exp(-dtau/mup) + I11up*np.exp(-tau[nb-2]/mu0)
    else:
        I1up[nb-2, :] = I11up*np.exp(-tau[nb-2]/mu0)
    for ib in range(nb-3, -1, -1):
        I1up[ib, :] = I1up[ib+1, :]*np.exp(-dtau/mup) + I11up*np.exp(-tau[ib]/mu0)
#
#   MS: only odd/even k are needed for up/down - not yet applied
    wpij = np.zeros((ng2, ng2)) # sum{xk*pk(mui)*pk(muj)*wj, k=0:nk}
    for ig in range(ng2):
        for jg in range(ng2):
            wpij[ig, jg] = wg[jg]*np.dot(xk, pk[ig, :]*pk[jg, :]) # thinkme: use matrix formalism?
#
#   MOM: [Jup; Jdn] = [[Tup Rup]; [Rdn Tdn]]*[Iup Idn]; Tup = Tdn = T; Rup = Rdn  = R   
    T = wpij[0:ng1, 0:ng1].copy()   # T.flags.c_contiguous = True
    R = wpij[0:ng1, ng1:ng2].copy() # R.flags.c_contiguous = True
#
    Iup = np.copy(I1up)
    Idn = np.copy(I1dn)
    for itr in range(nit):
#       Down:
        Iup05 = 0.5*(Iup[0, :] + Iup[1, :])
        Idn05 = 0.5*(Idn[0, :] + Idn[1, :]) # Idn[0, :] = 0.0        
        J = np.dot(R, Iup05) + np.dot(T, Idn05)
        Idn[1, :] = I11dn + (1.0 - np.exp(-dtau/mup))*J
        for ib in range(2, nb):
            Iup05 = 0.5*(Iup[ib-1, :] + Iup[ib, :]) 
            Idn05 = 0.5*(Idn[ib-1, :] + Idn[ib, :])
            J = np.dot(R, Iup05) + np.dot(T, Idn05)
            Idn[ib, :] = Idn[ib-1, :]*np.exp(-dtau/mup) + \
                             I11dn*np.exp(-tau[ib-1]/mu0) + \
                                 (1.0 - np.exp(-dtau/mup))*J
#       Up:
#       Lambertian surface
        if m == 0 and srfa > tiny:
            Iup[nb-1, :] = 2.0*srfa*np.dot(Idn[nb-1, :], mup*w) + 2.0*srfa*mu0*np.exp(-tau0/mu0)
        Iup05 = 0.5*(Iup[nb-2, :] + Iup[nb-1, :]) # Iup[nb-1, :] = 0.0
        Idn05 = 0.5*(Idn[nb-2, :] + Idn[nb-1, :])       
        J = np.dot(T, Iup05) + np.dot(R, Idn05)
        Iup[nb-2, :] = Iup[nb-1, :]*np.exp(-dtau/mup) + \
                           I11up*np.exp(-tau[nb-2]/mu0) + \
                               (1.0 - np.exp(-dtau/mup))*J
        for ib in range(nb-3, -1, -1): # going up, ib = 0 (TOA) must be included
            Iup05 = 0.5*(Iup[ib, :] + Iup[ib+1, :]) 
            Idn05 = 0.5*(Idn[ib, :] + Idn[ib+1, :])
            J = np.dot(T, Iup05) + np.dot(R, Idn05)
            Iup[ib, :] = Iup[ib+1, :]*np.exp(-dtau/mup) + \
                             I11up*np.exp(-tau[ib]/mu0) + \
                                 (1.0 - np.exp(-dtau/mup))*J
#       print("iter=%i"%itr)
    return mug, wg, Iup[:, :], Idn[:, :]
#==============================================================================
#
if __name__ == "__main__":
#
    m = 0
    mu0 = 0.6
    srfa = 0.3
    nit = 100
    ng1 = 32
    nlr = 100
    dtau = 0.01
    ssa = 1.0
    xk = 0.5*ssa*np.array([1.0, 0.0, 0.5])
#
    time_start = time.time()
    mug, wg, Iup, Idn = gsitm(m, mu0, srfa, nit, ng1, nlr, dtau, xk)
    time_end = time.time()
    print("1st call of gsitm: runtime = %.3f sec."%(time_end-time_start))   
#
    time_start = time.time()
    ng1 += 1
    mug, wg, Iup, Idn = gsitm(m, mu0, srfa, nit, ng1, nlr, dtau, xk)
    time_end = time.time()
    print("2nd call of gsitm: runtime = %.3f sec."%(time_end-time_start))
#
    time_start = time.time()
    ng1 -= 1
    mug, wg, Iup, Idn = gsitm(m, mu0, srfa, nit, ng1, nlr, dtau, xk)
    time_end = time.time()
    print("3rd call of gsitm: runtime = %.3f sec."%(time_end-time_start))
#
    time_start = time.time()
    ng1 += 1
    mug, wg, Iup, Idn = gsitm(m, mu0, srfa, nit, ng1, nlr, dtau, xk)
    time_end = time.time()
    print("4th call of gsitm: runtime = %.3f sec."%(time_end-time_start))
#
    np.savez('gsitm_output', mu0=mu0, srfa=srfa, ng1=ng1, dtau=dtau, ssa=ssa, xk=xk, \
             mug=mug, wg=wg, Iup=Iup, Idn=Idn)
    print('done: gsit output saved')
#
#==============================================================================