# API - Eigenvalues & SVD

Verified eigenvalue and singular value computation.

## Standard Eigenvalues

```@docs
RigorousEigenvaluesResult
rigorous_eigenvalues
evbox
```

## Rump's verifyeigall

```@docs
VerifyEigAllResult
verifyeigall
```

## Newton-Kantorovich Eigenpair

```@docs
NKEigenpairResult
```

## Rump-Lange 2023 Method

```@docs
VerifySvdAllResult
verifysvdall
BallArithmetic._rumplange2023_eig
```

## Generalized Eigenvalues

```@docs
RigorousGeneralizedEigenvaluesResult
rigorous_generalized_eigenvalues
gevbox
GEVResult
verify_generalized_eigenpairs
compute_beta_bound
miyajima_gev_enclosure
```

## Singular Value Decomposition

```@docs
RigorousSVDResult
rigorous_svd
svdbox
```

## Singular Value Theorems of Rump 2011 and Miyajima 2014

The enclosures the singular value routines select between, each named for the statement it
implements.

```@docs
BallArithmetic._rump2011_thm3_1
BallArithmetic._miyajima2014_thm4
BallArithmetic._miyajima2014_thm7
BallArithmetic._miyajima2014_thm10
BallArithmetic._miyajima2014_thm10_weyl
BallArithmetic._miyajima2014_thm11
BallArithmetic._svd_auto_theorem
```

## Schur-Newton Singular Values

```@docs
Miyajima2014aSchurNewtonResult
miyajima2014a_schurnewton
```

## Reordered Schur Form

```@docs
ordschur_ball
spectral_projector_error_bound
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
BallArithmetic._bujanovicetal2023_alg4
BallArithmetic._bujanovicetal2023_alg4_truncated
BallArithmetic._bujanovicetal2023_alg4_unordered
```

