import time
import numpy as np
from def_gauss_seidel_iterations_m import gauss_seidel_iterations_m
from def_source_function_integrate_up import source_function_integrate_up
from def_source_function_integrate_down import source_function_integrate_down
from def_single_scattering_up import single_scattering_up
from def_single_scattering_down import single_scattering_down
#==============================================================================
#
if __name__ == "__main__":
#
    prnt_scrn = False  # print out intensities on the screen
    fname_bmark = 'test_gsit_R&A_srf.txt'  # file with benchmark data
#
    phasefun = 'a'     # 'r': rayleigh; aerosol otherwise (case sensitive)
    nit = 10           # number of iterations
    ng1 = 8            # number of Gauss nodes per hemisphere
    nlr = 100          # number of layer elements
    dtau = 0.01        # integration step over tau
    ssa = 0.99999999   # single scattering albedo
    sza = 45.0         # solar zenith angle, degrees
    muup = np.linspace(-0.2, -0.9, num=8)  # cos(view zenith angle): upward
    azd = np.array([0.0, 45.0, 90.0, 135.0, 180.0])  # relative azimuths
#
    if phasefun == 'r':
        print("Rayleigh:")
        nm = 3
        xk = np.array([1.0, 0.0, 0.5])
        srfa = 0.0  # surface albedo
        icol = 3
    else:
        print("Aerosol:")
        nm = 10
        xk = np.array([1.000000, 2.084911, 2.459134, 2.234752, \
                       1.873098, 1.492956, 1.164725, 0.881976, \
                       0.689172, 0.506016, 0.402247, 0.289742, \
                       0.232980, 0.165427, 0.133290, 0.093127, \
                       0.074444, 0.050682, 0.039800, 0.025983, \
                       0.019929, 0.012274, 0.009257, 0.005312, \
                       0.004015, 0.002120, 0.001638, 0.000764, \
                       0.000602, 0.000211, 0.000178, 0.000033, \
                       0.000037, 0.000004, 0.000005, 0.000000])
        srfa = 0.3     # surface albedo
        icol = 4 
#------------------------------------------------------------------------------
#
    time_start = time.time()
#
    mu0 = np.cos(np.radians(sza))
    mudn = -muup
    nmu = len(muup)
    azr = np.radians(azd)
    naz = len(azd)
    nrows = nmu*naz  
#
#   Compute SS at TOA & BOA
    Itoa = np.zeros((nmu, naz))
    if srfa > 0.0:
        for imu, mu in enumerate(muup):
            Itoa[imu, :] = single_scattering_up(mu, mu0, azr, nlr*dtau, 0.5*ssa*xk) + \
                               2.0*srfa*mu0*np.exp(-nlr*dtau/mu0)*np.exp(nlr*dtau/mu)
    else:
        for imu, mu in enumerate(muup):
            Itoa[imu, :] = single_scattering_up(mu, mu0, azr, nlr*dtau, 0.5*ssa*xk)
#
    Iboa = np.zeros((nmu, naz))      
    for imu, mu in enumerate(mudn):
        Iboa[imu, :] = single_scattering_down(mu, mu0, azr, nlr*dtau, 0.5*ssa*xk)
#
#   Compute Fourier moments for MS
    time_gsitm = 0.0
    deltm0 = 1.0
    for m in range(nm):
#
#       Solve RTE at Gauss nodes & all boundaries
        t1 = time.time()
        mug, wg, Igup, Igdn = gauss_seidel_iterations_m(m, mu0, srfa, nit, ng1, nlr, dtau, 0.5*ssa*xk)
        t2 = time.time()
        time_gsitm += t2 - t1
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
            Ims_toa = source_function_integrate_up(m, mu, mu0, srfa, nlr, dtau, 0.5*ssa*xk, mug, wg, Ig05, Igdn[nlr, :])  
            Itoa[imu, :] += Ims_toa*cma
        for imu, mu in enumerate(mudn):
            Ims_boa = source_function_integrate_down(m, mu, mu0, nlr, dtau, 0.5*ssa*xk, mug, wg, Ig05)  
            Iboa[imu, :] += Ims_boa*cma
#
#       Kronecker delta = 2 for m > 0
        deltm0 = 2.0
        print('m =', m)
#   end for m
#
#   Scale to unit flux on TOA
    Itoa *= 0.5/np.pi
    Iboa *= 0.5/np.pi
#
    time_end = time.time()
#
#------------------------------------------------------------------------------
#
#   Test vs benchmark:
    dat = np.loadtxt(fname_bmark, comments='#', skiprows=1)
    Ibmup = dat[0:nrows, icol]
    Ibmdn = dat[nrows:, icol]
#
    print("\nTOA:")
    Ibup = np.transpose(np.reshape(Ibmup, (naz, nmu)))
    err = 100.0*(Itoa/Ibup - 1.0)
    if prnt_scrn:
        print("   azd   mu   gsit         err, %")
        for iaz, az in enumerate(azd):
            for imu, mu in enumerate(muup):
                print(" %5.1f %5.1f  %.4e  %.2f" %(az, mu, Itoa[imu, iaz], err[imu, iaz]))
    emax = np.amax(np.abs(err))
    eavr = np.average(np.abs(err))
    print(" max & avr errs: %.2f  %.2f" %(emax, eavr))
#
    print("\nBOA:")
    Ibdn = np.transpose(np.reshape(Ibmdn, (naz, nmu)))
    err = 100.0*(Iboa/Ibdn - 1.0)
    if prnt_scrn:
        print("   azd   mu   gsit         err, %")
        for iaz, az in enumerate(azd):
            for imu, mu in enumerate(mudn):
                print(" %5.1f %5.1f  %.4e  %.2f" %(az, mu, Iboa[imu, iaz], err[imu, iaz]))
    emax = np.amax(np.abs(err))
    eavr = np.average(np.abs(err))
    print(" max & avr errs: %.2f  %.2f" %(emax, eavr))
#
    print("\nmultiple scattering runtime = %.2f sec."%time_gsitm)
    print("gsit total runtime = %.2f sec."%(time_end-time_start))
#==============================================================================