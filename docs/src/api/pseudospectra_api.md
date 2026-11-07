# API - Pseudospectra

The resolvent bounds, the arc covers and the certification drivers.

Four constructions are collected here, and they differ in what is paid once and what is paid at
each point `z`: a subdivision of a rectangle into cells proved inside or outside the
`ε`-pseudospectrum; two lower bounds of `σ_min(A − zI)` prepared from one verified decomposition
and evaluated cheaply afterwards; chains of discs along a curve, each disc carrying a certified
bound; and the bound for a block upper triangular matrix obtained after a Sylvester similarity,
whose certified singular value computation is on one diagonal block only. The driver that covers
a circle adaptively, and the parallel and refined modes around it, are on the
[CertifScripts](@ref "API - CertifScripts") page.

## Cells inside and outside the pseudospectrum

A rectangle of the complex plane is cut into cells, each decided by the bounds at its centre and
the 1-Lipschitz dependence of `σ_min(A − zI)` on `z`; the cells the bounds do not decide are cut
again, down to a given size.

```@docs
pseudospectrum_cells
PseudospectrumCells
PseudospectrumCell
sigma_min_upper
```

## Floors of the smallest singular value

Two lower bounds of `σ_min(A − zI)`, each prepared once from a verified decomposition of `A` and
then evaluated at any `z` without a further singular value computation, and the upper bound of
`‖(A − zI)⁻¹‖₂` that is their reciprocal. The routine that prepares the block bound,
[`block_resolvent_floor`](@ref), is on the [Algorithm Steps](@ref "API - Algorithm Steps") page.

```@docs
svd_frame_floor
SVDFrameFloor
BlockResolventFloor
sigma_min_floor
resolvent_bound
```

## Chains of discs along a curve

The curve is followed in floating point and the discs along it are certified: for each disc a ball
containing `σ_min(T − zI)` over the whole disc, and for the chain the overlap of consecutive discs.

```@docs
Enclosure
BallArithmetic.compute_enclosure
check_enclosure(::Enclosure)
BallArithmetic.bound_resolvent
```

## The resolvent of a block upper triangular matrix

For `T = [T₁₁ T₁₂; 0 T₂₂]` with `T₂₂` upper triangular, the similarity `S(X) = [I X; 0 I]` makes
the coupling block small and leaves the certified singular value computation on the `k×k` block
`T₁₁` alone, at the cost of the factor `κ₂(S(X))`. The bound itself is
[`parametric_resolvent_bound`](@ref), on the [Linear Systems](@ref "API - Linear Systems") page;
what follows is the part that does not depend on `z`, the choices that enter its three terms, and
the two drivers that search over the split index and over the choices.

```@docs
sylvester_resolvent_precompute
solve_sylvester_oracle
SVDWarmStart
find_optimal_split
compare_all_configs
```

### The choices

```@docs
ResolventBoundConfig
config_v1
config_v2
config_v2p5
config_v3
NormEstimator
estimate_2norm
```
