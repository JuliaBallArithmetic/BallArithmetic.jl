# API - Matrix Decompositions

The verified decompositions: LU, QR, Cholesky, polar, Takagi and the singular value routines.

Each exported `verified_*` routine takes the arithmetic of the enclosure as a leading type
argument, `BigFloat` at `precision_bits` or `Float64`, encloses the input as a ball matrix in
that arithmetic, and returns a result type carrying a `certified` flag, which
[`iscertified`](@ref) reads. When the flag is `false` the radii are infinite and nothing is
proved. The statement each one establishes is in the docstring of its result type.

The routines whose names begin with an underscore are the steps of the published algorithms,
documented here because the exported routines are written in terms of them and because a
deviation from a paper has to be visible somewhere; they are not part of the interface.

## LU

The decomposition of a perturbed identity, Algorithm 3.2, is the common step of the LU, the
Cholesky and the QR enclosures; the LU driver itself is
[`BallArithmetic._rumpogita2024_lu`](@ref).

```@docs
verified_lu
VerifiedLUResult
BallArithmetic._rumpogita2024_alg3_2
```

## QR

```@docs
verified_qr
VerifiedQRResult
BallArithmetic._rumpogita2024_qr
cholesky_qr2
```

## Cholesky

[`verified_cholesky`](@ref), [`VerifiedCholeskyResult`](@ref) and the step
[`BallArithmetic._rumpogita2024_cholesky`](@ref) are documented with the linear system routines
and the algorithm steps, the factorisation being used there to solve and to test positive
definiteness.

## Polar

The enclosure comes from the singular value decomposition. The three iterations below are
floating-point algorithms returning approximations with no error bound, and are what an enclosure
can be started from.

```@docs
verified_polar
VerifiedPolarResult
BallArithmetic._rumpogita2024_polar
PolarIterationResult
polar_newton
polar_newton_schulz
polar_qdwh
```

## Takagi

```@docs
verified_takagi
VerifiedTakagiResult
BallArithmetic._rumpogita2024_takagi
```

## Schur

```@docs
BallArithmetic._rumpogita2024_schur
```

## Singular values and singular subspaces

Two routes, with different outputs. [`verified_svd`](@ref) encloses the singular values together
with orthonormal bases of the singular subspaces, clusters included, and is what
[`verified_polar`](@ref) is built on. [`certify_svd`](@ref) and the drivers around it enclose the
singular values from an approximate decomposition the caller supplies, with the residual and the
two orthogonality defects, and select the enclosure by one of the [`SVDMethod`](@ref) types.
[`svdbox`](@ref) and [`rigorous_svd`](@ref), the entry points, are documented with the eigenvalue
routines.

```@docs
verified_svd
SVDMethod
MiyajimaAuto
MiyajimaM1
MiyajimaM3
MiyajimaM4
certify_svd
svd_bounds
svd_lower_bound_sigma_min
rigorous_svd_m4
rigorous_svd_gpu
svd_bigfloat
BallArithmetic._rigorous_svd_bigfloat
BallArithmetic._miyajima2014_thm10_bounds
```

## Iterative refinement of an approximate decomposition

Floating-point refinement, run at a higher precision than the certification and carrying no error
bound of its own; the enclosure is made afterwards by [`certify_svd`](@ref).

```@docs
ogita_iterations_for_precision
BallArithmetic._ogita_svd_refine_impl
BallArithmetic._spectral_norm_bound
```

## Clusters and block enclosures

What the enclosures do where singular values or eigenvalues are not separated: the clusters are
read off the connectivity of the computed balls, and each cluster is enclosed as a whole.

```@docs
block_enclosure
overlap_components
BallArithmetic.sep_clusters
BallArithmetic._ComplexDiscs
BallArithmetic._RealIntervals
BallArithmetic._vbd_solve
BallArithmetic._certify_ball
```

## Floating-point factorisations used elsewhere

Approximations with no error bound, computed where a decomposition is needed as a candidate or as
a preconditioner. The two enum values are what the `method` of
[`compute_preconditioner`](@ref) selects; see [`PreconditionerType`](@ref).

```@docs
triangular_eigenvectors
LUFactorization
LDLTFactorization
```
