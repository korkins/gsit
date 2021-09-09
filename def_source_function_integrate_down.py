import numpy as np
from numba import jit
from def_legendre_polynomial import legendre_polynomial
from def_schmidt_polynomial import schmidt_polynomial
#
@jit(nopython=True, cache=True)
def source_function_integrate_down(m, mu, mu0, nlr, dtau, xk, mug, wg, Ig05):
    '''
    Task:
        Source function integration: down.
    In:
        m      i            Fourier moment: m = 0, 1, 2, ....
        mu     d            upward LOS cos(VZA) < 0
        nlr    i            number of layer elements dtau, tau0 = dtau*nlr
        dtau   d            thickness of element layer (integration step over tau)
        xk     d[nk]        expansion moments*ssa/2, (2k+1) included, nk=len(xk)
        mug    d[ng2]       Gauss nodes
        wg     d[ng2]       Gauss weights
        Ig05   d[nlr, ng2]  RTE solution at Gauss nodes & at midpoint of every layer dtau 
    Out:
        Iboa   d            Itoa=f(mu)
    Tree:
        -
    Note:
        TOA scaling factor = 2pi;
    Refs:
        1. -
    '''
#
#   Parameters:
    tiny = 1.0e-8
    ng2 = len(wg)
    nk = len(xk)
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
    if np.abs(mu - mu0) < tiny:
        I11dn = p*dtau*np.exp(-dtau/mu0)/mu0
    else:
        I11dn = p*mu0/(mu0 - mu)*(np.exp(-dtau/mu0) - np.exp(-dtau/mu))
#
    I1dn = np.zeros(nb)
    I1dn[1] = I11dn
    for ib in range(2, nb):
        I1dn[ib] = I1dn[ib-1]*np.exp(-dtau/mu) + I11dn*np.exp(-tau[ib-1]/mu0)
#
    wpij = np.zeros(ng2) # sum{xk*pk(mu)*pk(muj)*wj, k=0:nk}
    for jg in range(ng2):
        wpij[jg] = wg[jg]*np.dot(xk, pku[:]*pk[jg, :])
#
    Idn = np.copy(I1dn) # boundary condition: Idn[0, :] = 0.0
    J = np.dot(wpij, Ig05[0, :])
    Idn[1] = I11dn + (1.0 - np.exp(-dtau/mu))*J
    for ib in range(2, nb):   
        J = np.dot(wpij, Ig05[ib-1, :])
        Idn[ib] = Idn[ib-1]*np.exp(-dtau/mu) + \
                         I11dn*np.exp(-tau[ib-1]/mu0) + \
                             (1.0 - np.exp(-dtau/mu))*J
#
#   Subtract SS & extract BOA value
    Ims = Idn - I1dn
    Iboa = Ims[nb-1]
    return Iboa
#==============================================================================