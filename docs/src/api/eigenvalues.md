# API - Eigenvalues & SVD

Verified eigenvalue and singular value computation.

## Standard Eigenvalues

```@docs
RigorousEigenvaluesResult
rigorous_eigenvalues
evbox
```

## Rump 2022a Method

```@docs
Rump2022aResult
rump_2022a_eigenvalue_bounds
```

## Rump-Lange 2023 Method

```@docs
RumpLange2023Result
rump_lange_2023_cluster_bounds
refine_cluster_bounds
```

## Generalized Eigenvalues

```@docs
RigorousGeneralizedEigenvaluesResult
rigorous_generalized_eigenvalues
gevbox
GEVResult
verify_generalized_eigenpairs
compute_beta_bound
```

## Singular Value Decomposition

```@docs
RigorousSVDResult
rigorous_svd
svdbox
```

## Miyajima VBD

```@docs
MiyajimaVBDResult
miyajima_vbd
refine_svd_bounds_with_vbd
```

## Adaptive Ogita SVD

```@docs
OgitaSVDRefinementResult
AdaptiveSVDResult
ogita_svd_refine
adaptive_ogita_svd
```

## SVD Caching

```@docs
clear_svd_cache!
svd_cache_stats
set_svd_cache!
```

## Spectral Projectors

```@docs
RigorousSpectralProjectorsResult
miyajima_spectral_projectors
compute_invariant_subspace_basis
verify_projector_properties
projector_condition_number
```

## Block Schur Decomposition

```@docs
RigorousBlockSchurResult
rigorous_block_schur
extract_cluster_block
verify_block_schur_properties
estimate_block_separation
refine_off_diagonal_block
compute_block_sylvester_rhs
```

## Schur Spectral Projectors

```@docs
SchurSpectralProjectorResult
compute_spectral_projector_schur
compute_spectral_projector_hermitian
project_vector_spectral
verify_spectral_projector_properties
```

## Schur Refinement

```@docs
SchurRefinementResult
refine_schur_decomposition
rigorous_schur_bigfloat
certify_schur_decomposition
newton_schulz_orthogonalize!
```

## Riesz Projections

```@docs
project_onto_eigenspace
project_onto_schur_subspace
verified_project_onto_eigenspace
compute_eigenspace_projector
compute_schur_projector
```

## Singular Value Intervals

```@docs
qi_intervals
qi_sqrt_intervals
```
