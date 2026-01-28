# Pseudospectra Certification

This page documents the `CertifScripts` module for rigorous pseudospectrum computation.

## Overview

The pseudospectrum certification tools compute rigorous resolvent bounds along contours
in the complex plane. This is useful for:

- Verifying spectral gaps
- Computing pseudospectral radii
- Stability analysis of matrices

The certification uses adaptive arc refinement to efficiently sample the contour,
with several execution modes optimized for different scenarios.

## Basic Usage

The main entry point is [`run_certification`](@ref BallArithmetic.CertifScripts.run_certification),
which takes a matrix and a [`CertificationCircle`](@ref BallArithmetic.CertifScripts.CertificationCircle)
defining the contour.

Use [`compute_schur_and_error`](@ref BallArithmetic.CertifScripts.compute_schur_and_error)
to precompute the Schur decomposition and backward error, which is then passed to the
certification routine.

## Execution Modes

### Serial Mode (Float64)

The basic serial mode computes SVD at each sample point in Float64 precision.
Suitable for moderate-sized problems and contours not too close to the spectrum.

### Parallel Mode (Distributed.jl)

For large problems, certification can be parallelized across multiple Julia workers.
Load the `Distributed` extension:

```julia
using BallArithmetic, Distributed
using BallArithmetic.CertifScripts

addprocs(4)  # Add 4 workers
result = run_certification(A, circle, workers())
```

The parallel extension provides:
- Automatic worker management via [`dowork`](@ref BallArithmetic.CertifScripts.dowork)
- Job distribution via channels
- Checkpoint/snapshot support via [`save_snapshot!`](@ref BallArithmetic.CertifScripts.save_snapshot!)
  and [`choose_snapshot_to_load`](@ref BallArithmetic.CertifScripts.choose_snapshot_to_load)

### Cached Ogita Mode

When consecutive sample points are close together (which happens naturally during
adaptive bisection), we can reuse SVD information via Ogita's iterative refinement.
This provides approximately 2x speedup for adaptive certification.

Use [`run_certification_ogita`](@ref BallArithmetic.CertifScripts.run_certification_ogita)
for this mode.

The caching system tracks SVD results at sample points and reuses them for nearby points.
This is particularly effective during adaptive arc refinement via
[`adaptive_arcs!`](@ref BallArithmetic.CertifScripts.adaptive_arcs!).

Worker functions for distributed Ogita mode:
- [`dowork_ogita`](@ref BallArithmetic.CertifScripts.dowork_ogita) - Float64 precision worker
- [`dowork_ogita_bigfloat`](@ref BallArithmetic.CertifScripts.dowork_ogita_bigfloat) - BigFloat precision worker

### BigFloat Precision Mode

For contours very close to the spectrum (resolvent bounds > 10^15), Float64 precision
is insufficient. The BigFloat mode uses:

1. Compute Float64 SVD as initial approximation
2. Refine to BigFloat precision using Ogita's algorithm
3. Due to quadratic convergence, 4 iterations from Float64 saturate 256-bit precision

This mode is essential when verifying tight spectral gaps or computing pseudospectral
radii near eigenvalues.

## When to Use Each Mode

| Scenario | Recommended Mode |
|----------|------------------|
| Moderate resolvent (< 10^8) | Serial Float64 |
| Large matrix, moderate resolvent | Parallel Float64 |
| Tight contour, resolvent 10^8 - 10^15 | Cached Ogita |
| Very tight contour, resolvent > 10^15 | BigFloat Ogita |

## Helper Functions

Additional utility functions:
- [`points_on`](@ref BallArithmetic.CertifScripts.points_on) - Generate sample points on circle
- [`set_schur_matrix!`](@ref BallArithmetic.CertifScripts.set_schur_matrix!) - Configure Schur matrix
- [`configure_certification!`](@ref BallArithmetic.CertifScripts.configure_certification!) - Set up certification parameters
- [`bound_res_original`](@ref BallArithmetic.CertifScripts.bound_res_original) - Compute resolvent bound
- [`poly_from_roots`](@ref BallArithmetic.CertifScripts.poly_from_roots) - Construct polynomial from roots
