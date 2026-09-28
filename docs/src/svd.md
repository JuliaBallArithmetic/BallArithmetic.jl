# Singular Value Decomposition

This page documents verified SVD computation and related tools.

## Rigorous SVD

The [`rigorous_svd`](@ref) function computes verified singular value decomposition with
guaranteed error bounds on the singular values and vectors. It returns a
[`RigorousSVDResult`](@ref) containing the factors and bounds.

## Adaptive Ogita SVD

The [`adaptive_ogita_svd`](@ref) function uses iterative refinement to achieve tighter
singular value bounds. This is particularly useful when high precision is needed.
The refinement algorithm is based on Ref. [OgitaAishima2020](@cite).

Related functions:
- [`ogita_svd_refine`](@ref) - Single refinement step
- [`OgitaSVDRefinementResult`](@ref) - Result type for refinement
- [`AdaptiveSVDResult`](@ref) - Result type for adaptive SVD

## Block Gershgorin enclosure in the Schur basis

The [`schur_gershgorin_enclosure`](@ref) function encloses the spectrum by inflated
Gershgorin discs in a unitary basis of the midpoint matrix, grouped into clusters, and
returns a [`SchurGershgorinResult`](@ref). The basis is permuted, not transformed, so the
remainder is the off-block part of `Z*AZ`; for a basis that decouples the blocks see
[`miyajima2014a_schurnewton`](@ref).

## Singular Value Bounds

Low-level functions for computing rigorous bounds on singular values:

- [`svdbox`](@ref) - Box enclosure for singular values
- [`collatz_upper_bound_L2_opnorm`](@ref) - Collatz bound for L2 operator norm
- [`upper_bound_L2_opnorm`](@ref) - Upper bound on L2 operator norm
- [`qi_intervals`](@ref) - Qi's singular value intervals
- [`qi_sqrt_intervals`](@ref) - Qi's intervals with square root bounds
