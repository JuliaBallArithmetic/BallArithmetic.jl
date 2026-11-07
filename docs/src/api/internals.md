# API - Algorithm Steps

The steps of the published algorithms, each named for the statement it implements and each
reached through one of the exported routines rather than called directly. They are listed here
because the docstrings of the exported routines name them, so that a reader can follow a bound
back to the lemma it comes from.

## Rump 2022a

```@docs
BallArithmetic._rump2022a
BallArithmetic._rump2022aneumann
BallArithmetic._rump2022a_conditioned
BallArithmetic._rump2022a_discclusters
BallArithmetic._rump2022a_pencil
BallArithmetic._rump2022a_schur
BallArithmetic._rump2022a_schur_step6
BallArithmetic._rump2022a_prodK
BallArithmetic._rump2022a_prodK_pencil
BallArithmetic._rump2022a_transform
BallArithmetic._rump2022aneumann_transform
BallArithmetic._rump2022a_pencil_transform
BallArithmetic._vbd_unitary_basis
```

## Rump-Lange 2023

```@docs
BallArithmetic._rumplange2023
```

## Miyajima 2014a

```@docs
BallArithmetic._miyajima2014a_alg1
```

## Miyajima 2010, the generalized problem

Steps 1 to 3 of Algorithm 1 of [Miyajima2010](@cite) for the pencil `A x = λ B x`, and the two
`∞`-norm bounds they are built from.

```@docs
BallArithmetic.miyajima2010_algorithm1_steps1to3
BallArithmetic.bdd_R2
BallArithmetic._up_bound_Linf_opnorm
BallArithmetic._up_bound_Linf_norm
```

## Yamamoto 1980 and 1982, one eigenpair

The pieces of [`certify_eigenpair`](@ref): the vector that fixes the normalisation, the bordered
Jacobian, the norm the two defects are measured in, and the results returned when a test fails.

```@docs
BallArithmetic._compute_left_eigenvector
BallArithmetic._build_augmented_jacobian_ball
BallArithmetic._opnorm_bound
BallArithmetic._unverified_result
BallArithmetic._unverified_result_with_diagnostics
```

## Bujanović, Kressner and Schröder 2022, Schur refinement

The triangular matrix equation `stril(TL − LT) = −E` of [BujanovicEtAl2023](@cite),
its two solvers and the Sylvester equation the recursive one reduces to, with the iteration that
calls them.

```@docs
BallArithmetic._refine_schur_impl
BallArithmetic._solve_triangular_equation_direct!
BallArithmetic._solve_triangular_equation_recursive!
BallArithmetic._solve_sylvester_triangular!
BallArithmetic._stril
BallArithmetic._frobenius_norm
```

## Ogita and Aishima 2018, symmetric refinement

```@docs
BallArithmetic._refine_symmetric_eigen_impl
```

## Block Schur form

The two steps of [`rigorous_block_schur`](@ref) that act on the transformed matrix.

```@docs
BallArithmetic._apply_block_structure
BallArithmetic._compute_off_diagonal_norm
```

## Rump 2010

```@docs
BallArithmetic._rump2010_alg10_7
```

## Rump-Ogita 2024

```@docs
BallArithmetic._rumpogita2024_cholesky
BallArithmetic._rumpogita2024_lu
```

## Resolvent Floors

```@docs
block_resolvent_floor
```

## Matrix Classifiers

```@docs
BallArithmetic.is_M_matrix
```

## Spectral Radius

```@docs
BallArithmetic.collatz_upper_bound
```
