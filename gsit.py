import time
import numpy as np
#------------------------------------------------------------------------------
#
def gauszw(x1, x2, n):
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
    _yeps = 3.0e-14
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
            if (np.abs(yz - yz1) < _yeps):
                break # exit while loop
        x[i] = yxm - yz*yxl
        x[n-1-i] = yxm + yxl*yz
        w[i] = 2.0*yxl/((1.0 - yz*yz)*ypp*ypp)
        w[n-1-i] = w[i]
    return x, w
#==============================================================================
#
def polleg(x, kmax):
    '''
    Task:
        To compute the Legendre polynomials, Pk(x), for all orders k=0:kmax and a
        single point 'x' within [-1:+1]
    In:
        x      d   abscissa
        kmax   i   maximum order, k = 0,1,2...kmax
    Out:
        pk    [kmax+1]   Legendre polynomials
    Tree:
        -
    Notes:
        The dafault argument, m, is used to match the list of arguments of
        polqkm(x).
        
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
def polqkm(m, x, kmax):
    '''
    Task:
        To compute the Qkm(x) plynomials for all k = m:kmax & Fourier order
        m > 0. Qkm(x) = 0 is returned for all k < m.
    In:
        m      i   Fourier order (as in theory cos(m*phi)): m = 1,2,3....
        x      d   abscissa
        kmax   i   maximum order, k = 0,1,2...kmax
    Out:
        pk    [kmax+1]   polynomials
    Tree:
        -
    Notes:
        Think me: provide only non-zero polynomials, k>=m, on output.
        Definition:
            
            Qkm(x) = sqrt[(k-m)!/(k+m)!]*Pkm,                               (1)
            Pkm(x) = (1-x2)^(m/2)*(dPk(x)/dx)^m,                            (2)

        where Pk(x) are the Legendre polynomials. Note, unlike in [2] (-1)^m is
        omitted in Qkm(x). Refer to [1-4] for details.

        Qkm(x) for a few initial values of m > 0 and k for testing:
        m = 1:
            Q01 = 0.0                                                // k = 0
            Q11 = sqrt( 0.5*(1.0 - x2) )                             // k = 1
            Q21 = 3.0*x*sqrt( (1.0 - x2)/6.0 )                       // k = 2
            Q31 = (3.0/4.0)*(5.0*x2 - 1.0)*sqrt( (1.0 - x2)/3.0 )    // k = 3
        m = 2:
            Q02 = 0.0                                                // k = 0
            Q12 = 0.0                                                // k = 1
            Q22 = 3.0/(2.0*sqrt(6.0))*(1.0 - x2);	                 // k = 2
            Q32 = 15.0/sqrt(120.0)*x*(1.0 - x2);                     // k = 3
            Q42 = 15.0/(2.0*sqrt(360.0))*(7.0*x2 - 1.0)*(1.0 - x2)   // k = 4
        m = 3:
            Q03 = 0.0                                                // k = 0
            Q13 = 0.0                                                // k = 1
            Q23 = 0.0                                                // k = 2
            Q33 = 15.0/sqrt(720.0)*(1.0 - x2)*sqrt(1.0 - x2);        // k = 3
            Q43 = 105.0/sqrt(5040.0)*(1.0 - x2)*x*sqrt(1.0 - x2)     // k = 4

       Data for stress test: POLQKM.f90 (agrees with polqkm.cpp)
            k = 512 (in Fortran 513), m = 256
		       x        POLQKM.f90               def polqkm              |err|
            -1.00       0.000000000000000E+000  -0.0000000000000000e+00   0.0
            -0.50 (!)  -2.601822304856592E-002  -2.6018223048565915e-02   3.5e-18
             0.00       3.786666189291950E-002   3.7866661892919498e-02   0.0
             0.25       9.592316443679009E-003   9.5923164436790085e-03   0.0
             0.50 (!)  -2.601822304856592E-002  -2.6018223048565915e-02   3.5e-18
             0.75      -2.785756308806302E-002  -2.7857563088063021e-02   0.0
             1.00       0.000000000000000E+000   0.0000000000000000e+00   0.0
    Refs:
        1. Gelfand IM et al., 1963: Representations of the rotation and Lorentz
           groups and their applications. Oxford: Pergamon Press.
        2. Hovenier JW et al., 2004: Transfer of Polarized Light in Planetary
           Atmosphere. Basic Concepts and Practical Methods, Dordrecht: Kluwer
           Academic Publishers.
        3. http://mathworld.wolfram.com/AssociatedLegendrePolynomial.html
        4. http://www.mathworks.com/help/matlab/ref/legendre.html  
    '''
#
    nk = kmax+1
    qk = np.zeros(nk)
#
#   k=m: Qmm(x)=c0*[sqrt(1-x2)]^m
    c0 = 1.0
    for ik in range(2, 2*m+1, 2):
        c0 = c0 - c0/ik
    qk[m] = np.sqrt(c0)*np.power(np.sqrt( 1.0 - x*x ), m)
#
#	Q{k-1}m(x), Q{k-2}m(x) -> Qkm(x)
    m1 = m*m - 1.0
    m4 = m*m - 4.0
    for ik in range(m+1, nk):
        c1 = 2.0*ik - 1.0
        c2 = np.sqrt( (ik + 1.0)*(ik - 3.0) - m4 )
        c3 = 1.0/np.sqrt( (ik + 1.0)*(ik - 1.0) - m1 )
        qk[ik] = ( c1*x*qk[ik-1] - c2*qk[ik-2] )*c3
    return qk
#==============================================================================
#
def gsitm(m, mu0, srfa, nit, ng1, nlr, dtau, xk):
    '''
    Task:
        Solve RTE in a basic scenario using Gauss-Seidel (GS) iterations.
    In:
        m      i       Fourier moment: m = 0, 1, 2, ....
        mu0    d       cos(SZA) > 0
        srfa   d       Lambertian surface albedo
        nit    i       number of iterations, nit > 0
        ng1    i       number of gauss nodes per hemisphere
        nlr    i       number of layer elements dtau, tau0 = dtau*nlr
        dtau   d       thickness of element layer (integration step over tau)
        ssa    d       single scattering albedo
        xk     d[nk]   expansion moments*ssa/2, (2k+1) included, nk=len(xk)   
    Out:
        mug, wg    d[ng1*2]        Gauss nodes & weights
        Iup, Idn   d[nlr+1, ng1]   intensity, I = f(tau), @ Gauss nodes
    Tree:
        gsit()
            > gauszw() - computes Gauss zeros and weights.
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
    mup, w = gauszw(0.0, 1.0, ng1)
    mug = np.concatenate((-mup, mup))
    wg = np.concatenate((w, w))
#
    pk = np.zeros((ng2, nk))
    p = np.zeros(ng2)
    if m == 0:
        pk0 = polleg(mu0, nk-1)
        for ig in range(ng2):
            pk[ig, :] = polleg(mug[ig], nk-1)
            p[ig] = np.dot(xk, pk[ig, :]*pk0)
    else:
        pk0 = polqkm(m, mu0, nk-1)
        for ig in range(ng2):
            pk[ig, :] = polqkm(m, mug[ig], nk-1)
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
    T = wpij[0:ng1, 0:ng1].copy() 
    R = wpij[0:ng1, ng1:ng2].copy()
#    print('is wpij in C-order?:', wpij.flags.c_contiguous)
#    print('is T,R in C-order?:', T.flags.c_contiguous, T.flags.c_contiguous)
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
def sfiup(m, mu, mu0, srfa, nlr, dtau, xk, mug, wg, Ig05, Igboa):
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
        pk0 = polleg(mu0, nk-1)
        pku = polleg(mu, nk-1) 
        for ig in range(ng2):
            pk[ig, :] = polleg(mug[ig], nk-1)   
    else:
        pk0 = polqkm(m, mu0, nk-1)
        pku = polqkm(m, mu, nk-1) 
        for ig in range(ng2):
            pk[ig, :] = polqkm(m, mug[ig], nk-1) 
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
    Iup = np.copy(I1up) # boundary condition: Iup[nb-1, :] = 0.0
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
#
def sfidn(m, mu, mu0, nlr, dtau, xk, mug, wg, Ig05):
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
        pk0 = polleg(mu0, nk-1)
        pku = polleg(mu, nk-1) 
        for ig in range(ng2):
            pk[ig, :] = polleg(mug[ig], nk-1)   
    else:
        pk0 = polqkm(m, mu0, nk-1)
        pku = polqkm(m, mu, nk-1) 
        for ig in range(ng2):
            pk[ig, :] = polqkm(m, mug[ig], nk-1) 
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
#
def sglsup(mu, mu0, azr, tau0, xk):
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
        pk = polleg(nui, nk-1)
        p[inu] = np.dot(xk, pk)
#  
    mup = -mu
    I11up = p*mu0/(mu0 + mup)*(1.0 - np.exp(-tau0/mup - tau0/mu0))
#
    return I11up
#==============================================================================
#
def sglsdn(mu, mu0, azr, tau0, xk):
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
        pk = polleg(nui, nk-1)
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
    prnt_scrn = False
    scalef = 0.5/np.pi               # TOA flux scaling factor, 1/2pi
    fname_xk_aer = 'xk0036_0695.txt' # phase *matrix* moments
    fname_bmark = 'test_gsit_R&A_srf.txt'
#
    phasefun = 'a'                 # 'a': aerosol (case sensitive); rayleigh otherwise
    nit = 10                       # number of iterations: nit = 0 returns SS
    ng1 = 16                       # number of Gauss nodes per hemisphere
    nlr = 100                      # number of layer elements
    dtau = 0.01                    # integration step over tau
    ssa = 0.99999999               # single scattering albedo
    sza = 45.0                     # solar zenith angle
#               
    mu0 = np.cos(np.radians(sza))
    muup = np.array([-0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9])
    mudn = -muup
    nmu = len(muup)
    azd = np.array([0.0, 45.0, 90.0, 135.0, 180.0])
    azr = np.radians(azd)
    naz = len(azd)
    nrows = nmu*naz
#
    if nit == 0:
        print("WARNING: nit = 0 -> MS = double scattering")   
#
    if phasefun == 'a':
        print("Aerosol:")
        xk_all = np.loadtxt(fname_xk_aer, skiprows=8)
        xk = xk_all[:, 0]
        icol = 9
        nm = 10
        srfa = 0.3                     # surface albedo
    else:
        print("Rayleigh:")
        xk = np.array([1.0, 0.0, 0.5])
        icol = 5
        nm = 3
        srfa = 0.0                     # surface albedo
#
    time_start = time.time()
#
#   Compute SS at TOA & BOA
    Itoa = np.zeros((nmu, naz))
    if srfa > 0.0:
        for imu, mu in enumerate(muup):
            Itoa[imu, :] = sglsup(mu, mu0, azr, nlr*dtau, 0.5*ssa*xk) + \
                               2.0*srfa*mu0*np.exp(-nlr*dtau/mu0)*np.exp(nlr*dtau/mu)
    else:
        for imu, mu in enumerate(muup):
            Itoa[imu, :] = sglsup(mu, mu0, azr, nlr*dtau, 0.5*ssa*xk)
#
    Iboa = np.zeros((nmu, naz))      
    for imu, mu in enumerate(mudn):
        Iboa[imu, :] = sglsdn(mu, mu0, azr, nlr*dtau, 0.5*ssa*xk)
#
#   Compute Fourier moments for MS
    deltm0 = 1.0
    for m in range(nm):
#
#       Solve RTE at Gauss nodes & all boundaries
        mug, wg, Igup, Igdn = gsitm(m, mu0, srfa, nit, ng1, nlr, dtau, 0.5*ssa*xk)
#
#       Compute intensity at midpoint of every layer, both up & down 
        Ig05 = np.zeros((nlr, 2*ng1))
        for ilr in range(nlr):
            Iup05 = 0.5*(Igup[ilr, :] + Igup[ilr+1, :]) 
            Idn05 = 0.5*(Igdn[ilr, :] + Igdn[ilr+1, :])
            Ig05[ilr, :] = np.concatenate((Iup05, Idn05))
#
#       Accumulate Fourier series
        cma = deltm0*np.cos(m*azr)
        for imu, mu in enumerate(muup):
            Ims_toa = sfiup(m, mu, mu0, srfa, nlr, dtau, 0.5*ssa*xk, mug, wg, Ig05, Igdn[nlr, :])  
            Itoa[imu, :] += Ims_toa*cma
        for imu, mu in enumerate(mudn):
            Ims_boa = sfidn(m, mu, mu0, nlr, dtau, 0.5*ssa*xk, mug, wg, Ig05)  
            Iboa[imu, :] += Ims_boa*cma
#
#       Kroneker delta = 2 for m > 0
        deltm0 = 2.0
        print('m =', m)
#
#   Scale to unit flux on TOA
    Itoa *= scalef
    Iboa *= scalef
#
    time_end = time.time()
#    
#   Test vs benchmark  
#   Read benchmark:
    dat = np.loadtxt(fname_bmark, comments='#', skiprows=1)
    Ibmup = dat[0:nrows, icol]
    Ibmdn = dat[nrows:, icol]
#
    print(' ')
    print(" TOA:")
    print("   azd   mu   gsit         err, %")
    Ibup = np.transpose(np.reshape(Ibmup, (naz, nmu)))
    err = 100.0*(Itoa/Ibup - 1.0)
    for iaz, az in enumerate(azd):
        for imu, mu in enumerate(muup):
            print(" %5.1f %5.1f  %.4e  %.2f" %(az, mu, Itoa[imu, iaz], err[imu, iaz]))
    emax = np.amax(np.abs(err))
    eavr = np.average(np.abs(err))
    print(" max & avr errs: %.2f  %.2f     <<<" %(emax, eavr))
#
    print(' ')
    print(" BOA:")
    print("   azd   mu   gsit         err, %")
    Ibdn = np.transpose(np.reshape(Ibmdn, (naz, nmu)))
    err = 100.0*(Iboa/Ibdn - 1.0)
    for iaz, az in enumerate(azd):
        for imu, mu in enumerate(mudn):
            print(" %5.1f %5.1f  %.4e  %.2f" %(az, mu, Iboa[imu, iaz], err[imu, iaz]))
    emax = np.amax(np.abs(err))
    eavr = np.average(np.abs(err))
    print(" max & avr errs: %.2f  %.2f     <<<" %(emax, eavr))
#
    print("gsit runtime = %.2f sec."%(time_end-time_start))
#==============================================================================