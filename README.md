# GSIT Radiative Transfer Code

## Overview

This repository provides a Python implementation of the **Gauss–Seidel Iterative Technique (GSIT)** for solving the scalar radiative transfer equation in a **plane-parallel atmosphere over a reflecting surface**.

The atmosphere and surface are assumed to be **homogeneous (uniform in all directions)**. The radiative transfer problem with multiple scattering is solved using a **deterministic Gauss–Seidel iterative scheme**.

For details, see:

* S. Korkin, A.M. Sayer, A. Ibrahim, A. Lyapustin  
  *A practical guide to writing a radiative transfer code*  
  Computer Physics Communications, 271: 108198 (2022)  
  https://doi.org/10.1016/j.cpc.2021.108198

A multilayer extension of the model is available here:

* https://github.com/korkins/gsit_multilayer

---

## Brief Instructions

In `gsit.py`, **line 15**, select the scattering case:

---

### Case 1 — Rayleigh atmosphere

* **TOA**: max error = 0.02 %, avg error = 0.02 %
* **BOA**: max error = 0.02 %, avg error = 0.02 %
* **Multiple scattering runtime**: 0.04 s
* **Total runtime**: 0.13 s

---

### Case 2 — Aerosol over Lambertian surface

* **TOA**: max error = 0.10 %, avg error = 0.07 %
* **BOA**: max error = 0.12 %, avg error = 0.06 %
* **Multiple scattering runtime**: 0.06 s
* **Total runtime**: 0.18 s

---

## Benchmark Notes

* Errors are given in **percent (%) relative to benchmark**
* TOA = Top of Atmosphere
* BOA = Bottom of Atmosphere
* Single-layer benchmark cases are validated against deterministic reference solutions
* Aerosol case includes truncated phase function expansion for efficiency in higher orders

---

## Code Structure (Overview)

GSIT is organized into the following main components:

* **Single scattering**
  - upward and downward contributions using Legendre expansion

* **Multiple scattering solver**
  - Gauss–Seidel iterations over Fourier moments
  - Gaussian quadrature integration

* **Source function integration**
  - upward and downward radiance accumulation

---

## Reference

The theoretical formulation of GSIT is described in:

* https://doi.org/10.1016/j.cpc.2021.108198

---

## Notes

* The code is designed for **clarity and research prototyping**
* Suitable for extension to multilayer atmospheres
* A more advanced multilayer version is available in the linked repository above
