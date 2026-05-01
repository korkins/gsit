# GSIT Radiative Transfer Code

## Overview

This code (**GSIT**) simulates the transfer of **unpolarized monochromatic solar radiation** in a **plane-parallel atmosphere over a reflecting surface**.

Both atmosphere and surface are assumed to be **homogeneous (uniform in all directions)**. Multiple scattering is solved using the **deterministic Gauss–Seidel iterative method**, which gives the code its name.

For details, see:

* S. Korkin, A.M. Sayer, A. Ibrahim, A. Lyapustin  
  *A practical guide to writing a radiative transfer code*  
  Computer Physics Communications, 271: 108198 (2022)  
  https://doi.org/10.1016/j.cpc.2021.108198

A multilayer extension (including internal atmospheric solution) is available here:

* https://github.com/korkins/gsit_multilayer

---

## Brief Instructions

In `gsit.py`, **line 15**, select one of two test cases:

---

### Case 1 — Rayleigh atmosphere (`phasefun = 'r'`)

* **TOA**: max error = 0.02 %, avg error = 0.02 %
* **BOA**: max error = 0.02 %, avg error = 0.02 %
* **Multiple scattering runtime**: 0.04 s  
* **Total runtime**: 0.13 s  

---

### Case 2 — Aerosol over Lambertian surface (`phasefun = 'a'`)

* **TOA**: max error = 0.10 %, avg error = 0.07 %
* **BOA**: max error = 0.12 %, avg error = 0.06 %
* **Multiple scattering runtime**: 0.06 s  
* **Total runtime**: 0.18 s  

---

## Benchmark Notes

* Errors are given in **percent (%) relative to reference benchmark**
* TOA = Top of Atmosphere
* BOA = Bottom of Atmosphere
* Results are validated against deterministic benchmark solutions

---

## Code Structure (Tree & LOC)
gsit (40)
│
├── single_scattering_up (11)
│ └── legendre_polynomial (13)
│
├── single_scattering_down (14)
│ └── legendre_polynomial (13)
│
├── gauss_seidel_iterations_m (71)
│ ├── gauss_zeroes_weights (25)
│ ├── legendre_polynomial (13)
│ └── schmidt_polynomial (14)
│
├── source_function_integrate_up (42)
│ ├── legendre_polynomial (13)
│ └── schmidt_polynomial (14)
│
└── source_function_integrate_down (38)
├── legendre_polynomial (13)
└── schmidt_polynomial (14)



**Total LOC: 268**

---

## Key Features

* Deterministic Gauss–Seidel multiple scattering solver
* Rayleigh and aerosol scattering support
* Fast execution (sub-second runtime)
* Benchmark-validated accuracy
* Designed for clarity and extensibility

---

## Reference

GSIT method is described in:

* https://doi.org/10.1016/j.cpc.2021.108198

---

## Notes

* TOA: Top of Atmosphere  
* BOA: Bottom of Atmosphere  
* Plane-parallel homogeneous atmosphere assumption  
* Code prioritizes clarity and research prototyping over optimization  
* Suitable for later translation to C/Fortran implementations  

---
