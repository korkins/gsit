import time
import numpy as np
from numba import jit
#
@jit(nopython=True, cache=True)
def schmidt_polynomial(m, x, kmax):
    '''
    Task:
        To compute the Qkm(x) plynomials for all k = m:kmax & Fourier order
        m > 0. Qkm(x) = 0 is returned for all k < m.
    In:
        m      i   Fourier order (as in theory cos(m*phi)): m = 1,2,3....
        x      f   abscissa
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
if __name__ == "__main__":
#
#
    mu = np.linspace(-1.0, 1.0, 1001)
    nmu = len(mu)
    kmax = 256
    nm = 128
    qkmu = np.zeros((nm, nmu, kmax+1))
    time_start = time.time()
    for im in range(nm):
        for imu in range(nmu):
            qkmu[im, imu, :] = schmidt_polynomial(im, mu[imu], kmax)
    time_end = time.time()
#
    print("polqkm = %.3f sec."%(time_end-time_start))
#==============================================================================