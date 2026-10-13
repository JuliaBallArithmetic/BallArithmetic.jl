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
```

## Rump-Lange 2023 Method

```@docs
VerifySvdAllResult
verifysvdall
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

## Block Gershgorin enclosure

```@docs
SchurGershgorinResult
schur_gershgorin_enclosure
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

## Block Schur Decomposition

```@docs
RigorousBlockSchurResult
rigorous_block_schur
extract_cluster_block
verify_block_schur_properties
estimate_block_separation
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

