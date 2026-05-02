# GSIT Radiative Transfer Code

## Summary:

Code 'gsit' simulates the transfer of unpolarized, monochromatic solar radiation in a plane-parallel atmosphere over a reflecting Lambertian surface. Both the surface and the atmosphere are spatially homogeneous (uniform). Multiple scattering is solved numerically using the deterministic Gauss-Seidel iteration method - hence, the name.

For details, see the RT guide: S. Korkin, A.M. Sayer, A. Ibrahim, A. Lyapustin, "A practical guide to writing a radiative transfer code", Computer Physics Communications, 271: 108198, 2022. doi: https://doi.org/10.1016/j.cpc.2021.108198

For a version of 'gsit' that accounts for multiple layers, including the solution within the atmosphere, see: https://github.com/korkins/gsit_multilayer.


## Instructions:

In gsit.py, line 15:

    phasefun = 'r' # Rayleigh case
    
        TOA: max & avr errs: 0.02  0.02
        BOA: max & avr errs: 0.02  0.02
        
        multiple scattering runtime = 0.04 sec.
        gsit total runtime = 0.13 sec.

    phasefun = 'a' # Aerosol over Lambertian surface
    
        TOA: max & avr errs: 0.10  0.07
        BOA: max & avr errs: 0.12  0.06

        multiple scattering runtime = 0.06 sec.
        gsit total runtime = 0.18 sec.

The maximum (max) and average (avr) errors relative to the benchmark are given in %.


## Tree & LOC:

```
gsit (40)  # input and test commands are not counted
   |
   +-single_scattering_up (11)
   |                    |
   |                    +-legendre_polynomial (13)
   |
   +-single_scattering_down (14)
   |                      |
   |                      +-legendre_polynomial
   |
   +-gauss_seidel_iterations_m (71)
   |                         |
   |                         +-gauss_zeroes_weights (25)
   |                         |
   |                         +-legendre_polynomial
   |                         |
   |                         +-schmidt_polynomial (14)
   |
   +-source_function_integrate_up (42)
   |                            |
   |                            +-legendre_polynomial
   |                            |
   |                            +-schmidt_polynomial
   |
   +-source_function_integrate_down (38)
                                  |
                                  +-legendre_polynomial
                                  |
                                  +-schmidt_polynomial
```

LOC = 40 + 11 + 13 + 14 + 71 + 25 + 14 + 42 + 38 = 268


## Errata, Modifications, and Notes:

1. Files uploaded to the journal repository at the time of publication do NOT reflect the changes below.

2. In multiple scattering simulations (function 'gauss_seidel_iterations_m'), 'gsit' originally used all expansion moments of the phase function, xk[:]. This is numerically harmless but impractical for efficiency. Now, 'gsit.py' contains the following change:
line 72: nk = min(ng1 * 2, len(xk))
line 78: ... = gauss_seidel_iterations_m(..., 0.5 * ssa * xk[0 : nk])

Thus, for Rayleigh with len(xk) = 3 and ng2 = ng1 * 2 = 8 * 2 = 16 ordinates, nk = 3: all moments will be used. For Aerosol, with len(xk) = 36 and the same ng2 = 16, only the first 16 moments will be used in the second and higher scattering orders. The (exact) single scattering still uses all 36 expansion moments of the phase function.

3. Fig. 4 is missing an element, properly shown in Fig. B of this paper: https://doi.org/10.1016/j.jqsrt.2022.108194. Fig. B with caption is uploaded to this repository for convenience: Korkin_etal_jqsrt(2022)_FigB_Caption.jpg. In its right-hand side, note element (c3) missing from Fig. 4 of the RT guide - but not from the 'gsit' code (otherwise the tests would fail).

4. In the code snippets presented in the RT guide as figures, we use bold font for arrays. Since we did this manually, a few elements have mistakenly not been bold. Here is the list of those spotted so far:
```
       Fig.12(e), line 42: tau[ib-1]
       Fig.12(f), line 48: mup
       Fig.12(i), line 73: tau[ib-1]
       Fig.12(j), line 82: tau[nb-2]
                  line 87: Idn05
                  line 89: tau[ib]
       Fig.13(c), line 35: tau[ib-1]
       Fig.13(e), line 43: wpij and Ig05[ib-1, :]
                  line 45: tau[ib-1]
       Fig.14,    line 45: tau[nb-2]
                  line 50: tau[ib]
       Sec.4.2, code snippet, middle line: mask[jl]
```

5. Eq. (17), right-most term (direct solar beam bouncing) is missing the factor of π/μ₀. See Eqs. (3), where the factor is present. The code also accounts for this factor properly (see the Aerosol over Lambertian surface test).

6. Typo: p. 12, after Eq. (15), the Wolfram website is assigned the footnote #16, but the correct one is #17 (p. 11) or #19 (p. 12) - the two web links are identical.

7. Sec. 4.2 and Figs. 16(a,b): splitting atmosphere into equally thick element layers 'dtau' for integration over optical thickness 'tau' works well for simulating signal reflected from TOA or transmitted through BOA. See, e.g., Table 1 (column NL - number of optical layers - cases with NL > 1) in http://dx.doi.org/10.1016/j.jqsrt.2017.04.035. However, for simulation of light scattering inside atmosphere, the alternation of the extinction profile - Fig. 16(b) - may introduce errors. For example, in the upper atmosphere, many optically thin layers are combined into one dtau, making the result within that part of the atmosphere incorrect (as expected). For a slightly different strategy, see 'splittau.py' in 'gsit_multilayer' code (reference above).

8. Figs. 13(e) and 14: single scattering contribution is added (lines 45 and 50, respectively) to properly reflect the physical process. At returns (lines 47 and 52, respectively), it is subtracted for numerical reason (single scattering correction - see Fig. 15(d)). Clearly, one can drop both steps: neither add, nor subtract single scattering at the step of integration over tau at user-defined directions (up and down). This results in simulation of light scattered multiple times (2+). Exact single scattering is added separately.

9. Fig. 12(c): consider precomputing the arrays mug and wg in lines 14-16 and pass them as parameters.

10. Fig. 12(i) lines 64-67, Fig. 12(j) lines 78-83, Fig. 13(c) line 32, Fig. 13(e) line 41, Fig. 14 lines 44-64: in all these cases, we explicitly code the first step in the loop for clarity. This is, however, optional: including the first step in the main loop would make the code shorter (see 'gsit_multilayer')

-EOF-
