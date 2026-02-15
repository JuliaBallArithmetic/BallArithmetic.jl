# Eigenvalues, Sylvester Equations, and Projectors

This page documents verified eigenvalue computation, Sylvester equation solvers, and spectral projectors.

## Overview

We implement several algorithms to compute rigorous enclosures of eigenvalues, following
Ref. [Miyajima2012](@cite). The interested reader may refer to the treatment in
[Eigenvalues in Arb](https://fredrikj.net/blog/2018/12/eigenvalues-in-arb/) for a deeper discussion on the topic.

## Standard Eigenvalue Problems

### Main Functions

- [`rigorous_eigenvalues`](@ref) - General eigenvalue verification
- [`rump_2022a_eigenvalue_bounds`](@ref) - Individual eigenvector error bounds (Rump 2022a method)
- [`rump_lange_2023_cluster_bounds`](@ref) - Schur-based cluster bounds
- [`refine_cluster_bounds`](@ref) - Iterative refinement of cluster bounds

### Result Types

- [`RigorousEigenvaluesResult`](@ref) - Result from rigorous eigenvalue computation
- [`Rump2022aResult`](@ref) - Result with individual eigenvector bounds
- [`RumpLange2023Result`](@ref) - Schur-based result

## Generalized Eigenvalue Problems

For the generalized eigenvalue problem ``Ax = \lambda Bx``:

- [`verify_generalized_eigenpairs`](@ref) - Verify all eigenpairs
- [`rigorous_generalized_eigenvalues`](@ref) - Full rigorous computation
- [`compute_beta_bound`](@ref) - Compute β bound for B matrix

### Result Types

- [`GEVResult`](@ref) - Generalized eigenvalue result
- [`RigorousGeneralizedEigenvaluesResult`](@ref) - Full result with bounds

## Sylvester Equations

Fast verified computation for solutions of Sylvester equations ``AX + XB = C``,
following Ref. [MiyajimaSylvester2013](@cite).

- [`sylvester_miyajima_enclosure`](@ref) - General Sylvester solver
- [`triangular_sylvester_miyajima_enclosure`](@ref) - Triangular variant

## Spectral Projectors

Computation of spectral projectors for eigenvalue clustering,
following Ref. [MiyajimaInvariantSubspaces2014](@cite).

- [`miyajima_spectral_projectors`](@ref) - Main spectral projector computation
- [`compute_spectral_projector_schur`](@ref) - Schur-based projector
- [`compute_spectral_projector_hermitian`](@ref) - Hermitian case

### Eigenspace Projection

- [`project_onto_eigenspace`](@ref) - Project vector onto eigenspace
- [`project_onto_schur_subspace`](@ref) - Project onto Schur subspace
- [`compute_eigenspace_projector`](@ref) - Compute eigenspace projection matrix
- [`compute_schur_projector`](@ref) - Compute Schur subspace projector

### Result Types

- [`RigorousSpectralProjectorsResult`](@ref) - Spectral projector result
- [`SchurSpectralProjectorResult`](@ref) - Schur-based result

## Block Schur Decomposition

Rigorous block Schur decomposition for eigenvalue clustering.

- [`rigorous_block_schur`](@ref) - Block Schur decomposition
- [`extract_cluster_block`](@ref) - Extract diagonal block for cluster
- [`refine_schur_decomposition`](@ref) - Newton-based iterative refinement
- [`rigorous_schur_bigfloat`](@ref) - BigFloat precision variant
- [`certify_schur_decomposition`](@ref) - Wrap a `SchurRefinementResult` in `BallMatrix` form

### Auxiliary Functions

- [`newton_schulz_orthogonalize!`](@ref) - Newton-Schulz orthogonalization

### Result Types

- [`RigorousBlockSchurResult`](@ref) - Block Schur result
- [`SchurRefinementResult`](@ref) - Refinement result

## Limitations of Mixed-Precision Schur Refinement

The iterative Schur refinement algorithm ([BujanovicKressnerSchroder2022](@cite)) achieves
fast convergence by solving the triangular matrix equation ``\operatorname{stril}(TL - LT) = -E``
in **low precision** (Float64). This works well when the eigenvalues of ``T`` are
well-separated in Float64 arithmetic. However, the approach can fail on
matrices with extreme eigenvalue clustering.

### When does it fail?

The triangular solve requires dividing by eigenvalue differences
``T_{ii} - T_{jj}``. When these differences fall below Float64 machine epsilon
(``\approx 2.2 \times 10^{-16}``), the low-precision solver cannot distinguish
them and zeroes out the corresponding entries of ``L``. This prevents the
correction ``W = L - L^H`` from fixing the associated components of the residual.

**Symptom:** the orthogonality defect ``\|Q^H Q - I\|`` converges quadratically
(Newton-Schulz keeps working), but the **residual**
``\|\operatorname{stril}(Q^H A Q)\|_2 / \|A\|_F`` **stalls** at a fixed value,
typically around ``10^{-22}`` to ``10^{-16}``, far above the target tolerance.

Solving the triangular equation in BigFloat instead does not help either: when
eigenvalue gaps are tiny (e.g. ``10^{-60}``), the entries of ``L`` become enormous
(``\|L\| \gg 1``), violating the small-perturbation assumption of the Newton-like
iteration, which then diverges.

### Example: GKW transfer operator (257 × 257)

A concrete example arises from Gauss-Kuzmin-Wirsing transfer operators truncated
at ``K = 256``. The resulting matrix has:

| Property | Value |
|----------|-------|
| Condition number | ``\approx 1.5 \times 10^{61}`` |
| Eigenvalues with ``|\lambda| < 10^{-60}`` | 42 out of 257 |
| Eigenvalues with ``|\lambda| < 10^{-20}`` | 209 out of 257 |
| Minimum eigenvalue separation | ``\approx 3.5 \times 10^{-60}`` |
| Spectral radius | ``\approx 1.0`` |

With a Float64 Schur seed, refinement stalls at residual ``\approx 3.75 \times 10^{-22}``
(target: ``\approx 8.9 \times 10^{-160}``), because Float64 cannot resolve 80%
of the eigenvalue spectrum.

### Recommended alternatives

For matrices where the eigenvalue separation is below Float64 resolution:

1. **Direct BigFloat Schur** via `GenericSchur.jl`: compute the Schur decomposition
   directly at the target precision, bypassing iterative refinement entirely.
   Pass the result via the `schur_seed` keyword argument of
   [`rigorous_schur_bigfloat`](@ref) to certify it.

2. **Block Schur refinement**: group nearly-degenerate eigenvalue clusters into
   blocks and refine the block structure, avoiding division by small eigenvalue
   differences. See [`rigorous_block_schur`](@ref).

