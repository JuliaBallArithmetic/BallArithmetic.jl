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

### Auxiliary Functions

- [`newton_schulz_orthogonalize!`](@ref) - Newton-Schulz orthogonalization

### Result Types

- [`RigorousBlockSchurResult`](@ref) - Block Schur result
- [`SchurRefinementResult`](@ref) - Refinement result

