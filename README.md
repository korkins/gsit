# GSIT Radiative Transfer Code

## Summary:

Code 'gsit' simulates the transfer of unpolarized monochromatic solar radiation in a plane-parallel atmosphere over a reflecting surface. Both the surface and the atmosphere are homogeneous (uniform) at all directions. The multiple scattering is numerically solved using the deterministic method of Gauss-Seidel iterations - hence, the name.

For details, see S. Korkin, A.M. Sayer, A. Ibrahim, A. Lyapustin, "A practical guide to writing a radiative transfer code", Computer Physics Communications, 271: 108198, 2022. doi: https://doi.org/10.1016/j.cpc.2021.108198

For a version of 'gsit' accounting for multiple layers, including solution inside atmosphere, see: https://github.com/korkins/gsit_multilayer.


## Instructions:

In gsit.py, line 15:

    phasefun = 'r' # Rayleigh case
        TOA:
         max & avr errs: 0.02  0.02

        BOA:
         max & avr errs: 0.02  0.02

        multiple scattering runtime = 0.04 sec.
        gsit total runtime = 0.13 sec.

    phasefun = 'a' # Aerosol over Lambertian surface
        TOA:
         max & avr errs: 0.10  0.07

        BOA:
         max & avr errs: 0.12  0.06

        multiple scattering runtime = 0.06 sec.
        gsit total runtime = 0.18 sec.

Maximum (max) and average (avr) errors vs. benchmark is in %.


## Tree & LOC:

## TREE & LOC:

```
gsit(40)  # input and test commands are not counted
   |
   +-single_scattering_up(11)
   |   |
   |   +-legendre_polynomial(13)
   |
   +-single_scattering_down(14)
   |   |
   |   +-legendre_polynomial(13)
   |
   +-gauss_seidel_iterations_m(71)
   |   |
   |   +-gauss_zeroes_weights(25)
   |   |
   |   +-legendre_polynomial(13)
   |   |
   |   +-schmidt_polynomial(14)
   |
   +-source_function_integrate_up(42)
   |   |
   |   +-legendre_polynomial(13)
   |   |
   |   +-schmidt_polynomial(14)
   |
   +-source_function_integrate_down(38)
       |
       +-legendre_polynomial(13)
       |
       +-schmidt_polynomial(14)
```

LOC = 40 + 11 + 14 + 71 + 25 + 42 + 38 + 13 + 14 = 268


## Erratum, Modifications, and Notes:

1. In multiple scattering simulations (function 'gauss_seidel_iterations' ), 'gsit' originally used all expansion moments of the phase function, xk. This is numerically harmless, but impractical in terms of efficiency. Now, 'gsit.py' contains the following change:
line 72: nk = min(ng1*2, len(xk))
line 78: ... = gauss_seidel_iterations_m(..., 0.5*ssa*xk[0:nk])

Thus, for Rayleigh with len(xk) = 3 and ng2 = ng1*2 = 8*2 = 16 ordiantes, nk = 3: all moments will be used. For Aerosol, with len(xk) = 36 and the same ng2 = 16, only 16 moments will be used in second and higher scattering orderes. The (exact) single scattering still uses all 36 expansion moments of the phase function.

Files uploaded to CPC repository remained unchanged.

2. Fig.4 is missing an element, properly shown in Fig.B of this paper: https://doi.org/10.1016/j.jqsrt.2022.108194. Fig.B with caption is uploaded to this repository for convenince. In its right-hand side, note element (c3) missing from Fig.4 of the RT guide - but not from the 'gsit' code (test would fail otherwise).

3. In the code snippets, presented in the Guide as figures, we use bold font for arrays. Since we did it manually, a few elements have mistakenly not been "bolded". Here is the list of those spotted so far:
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

4. Eq.(17), right-most term (direct solar beam bouncing) is missing the factor of pi/mu0. See Eqs.(3), where the factor is present. The code also accounts fro this factor properly (see the Aerosol-over-Lambertian surface test).

5. Typo: p.12, after Eq.(15), the Wolfram website is assigned the footnote #16, but the correct one is #17 (p.11) or #19 (p.12) - the two weblinks are identical.

6. Sec.4.2 and Figs.16(ab): splitting atmopshere into equally thick element layeres 'dtau' for integration over optical thicknes 'tau' works well for simulating signal reflected from TOA or tranmitted through BOA. See, e.g., Table 1 (column NL - number of optical layeres - cases with NL > 1) in http://dx.doi.org/10.1016/j.jqsrt.2017.04.035. However, for simulation of light scattering inside atmopshere, the alternation of exticntion profile - Fig.16(b) - may introduce errors. For a slightly different strategy, see 'splittau.py' in 'gsit_multilayer' (reference above).

7. Figs.13(e) and 14: single scattering contribution is added (lines 45 and 50, respectively) to properly reflect physical process. At returns (lines 47 and 52, respectively), it is subtracted for numerical reason (single scattering correction - see Fig.15(d)). Clearly, one can drop both steps: neithher add, not subtract single scattering at the step of integration over tau at user defined directions (up and down). This results in simulation of light scattered multiple times (2+), exact single scattering is added separately.

8. Fig.12(c): consider precomputing the arrays mug and wg in lines 14-16 and pass them as parameteres.

9. Fig.12(i) lines 64-67, Fig.12(j) lines 78-83, Fig.13(c) line 32, Fig.13(e) line 41, Fig.14 lines 44-64: in all these cases, we explicitely code the first step in the loop pursuing clarity. This step is, however, optional: including the first step in the main loop would make the code shorter.

-EOF-
