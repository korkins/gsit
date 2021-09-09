import numpy as np
from numba import jit
from def_legendre_polynomial import legendre_polynomial
from def_schmidt_polynomial import schmidt_polynomial
#
@jit(nopython=True, cache=True)
def source_function_integrate_up(m, mu, mu0, srfa, nlr, dtau, xk, mug, wg, Ig05, Igboa):
    '''
    Task:
        Source function integration: up.
    In:
        m      i            Fourier moment: m = 0, 1, 2, ....
        mu     d            upward LOS cos(VZA) < 0
        mu0    d            cos(SZA) > 0
        srfa   d            Lambertian surface albedo
        nlr    i            number of layer elements dtau, tau0 = dtau*nlr
        dtau   d            thickness of element layer (integration step over tau)
        ssa    d            single scattering albedo
        xk     d[nk]        expansion moments*ssa/2, (2k+1) included, nk=len(xk)
        mug    d[ng2]       Gauss nodes
        wg     d[ng2]       Gauss weights
        Ig05   d[nlr, ng2]  RTE solution at Gauss nodes & at midpoint of every layer dtau
        Igboa  d[ng1]       Same as Ig05, except for downward at BOA
    Out:
        Itoa   d            Itoa=f(mu)
    Tree:
        -
    Note:
        TOA scaling factor = 2pi;
    Refs:
        1. -
    Revision History:
        2020-06-29:
            New input parameter: m;
            Removed input paramter: ssa;
            Pk(x) or Qkm(x) is now called depending on m;
        2020-06-18:
            Changes similar to gsitm()
        2020-06-14:
            First created and tested vs IPOL for R&A
    '''
#
#   parameters:
    ng2 = len(wg)
    ng1 = ng2//2
    nk = len(xk)
    mup = -mu
    nb = nlr+1
    tau0 = nlr*dtau
    tau = np.linspace(0.0, tau0, nb)
#
    pk = np.zeros((ng2, nk))
    if m == 0:
        pk0 = legendre_polynomial(mu0, nk-1)
        pku = legendre_polynomial(mu, nk-1) 
        for ig in range(ng2):
            pk[ig, :] = legendre_polynomial(mug[ig], nk-1)   
    else:
        pk0 = schmidt_polynomial(m, mu0, nk-1)
        pku = schmidt_polynomial(m, mu, nk-1) 
        for ig in range(ng2):
            pk[ig, :] = schmidt_polynomial(m, mug[ig], nk-1) 
    p = np.dot(xk, pku*pk0)
#  
    I11up = p*mu0/(mu0 + mup)*(1.0 - np.exp(-dtau/mup - dtau/mu0))
#
    I1up = np.zeros(nb)
    if m == 0 and srfa > 0.0:
        I1up[nb-1] = 2.0*srfa*mu0*np.exp(-tau0/mu0)
        I1up[nb-2] = I1up[nb-1]*np.exp(-dtau/mup) + I11up*np.exp(-tau[nb-2]/mu0)
    else:
        I1up[nb-2] = I11up*np.exp(-tau[nb-2]/mu0)
    for ib in range(nb-3, -1, -1):
        I1up[ib] = I1up[ib+1]*np.exp(-dtau/mup) + I11up*np.exp(-tau[ib]/mu0)
#
    wpij = np.zeros(ng2) # sum{xk*pk(mu)*pk(muj)*wj, k=0:nk}
    for jg in range(ng2):
        wpij[jg] = wg[jg]*np.dot(xk, pku[:]*pk[jg, :])
#
    Iup = np.copy(I1up)
    if m == 0 and srfa > 0.0:
        Iup[nb-1] = 2.0*srfa*np.dot(Igboa, mug[ng1:ng2]*wg[ng1:ng2]) + \
                        2.0*srfa*mu0*np.exp(-tau0/mu0)
    J = np.dot(wpij, Ig05[nb-2, :])
    Iup[nb-2] = Iup[nb-1]*np.exp(-dtau/mup) + \
                       I11up*np.exp(-tau[nb-2]/mu0) + \
                               (1.0 - np.exp(-dtau/mup))*J
    for ib in range(nb-3, -1, -1):   
        J = np.dot(wpij, Ig05[ib, :])
        Iup[ib] = Iup[ib+1]*np.exp(-dtau/mup) + \
                         I11up*np.exp(-tau[ib]/mu0) + \
                             (1.0 - np.exp(-dtau/mup))*J
#
#   Subtract SS (including surface) & extract TOA value
    Ims = Iup - I1up
    Itoa = Ims[0]
    return Itoa
#==============================================================================