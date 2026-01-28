# BallArithmetic.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaBallArithmetic.github.io/BallArithmetic.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaBallArithmetic.github.io/BallArithmetic.jl/dev/)
[![Build Status](https://github.com/JuliaBallArithmetic/BallArithmetic.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaBallArithmetic/BallArithmetic.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaBallArithmetic/BallArithmetic.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaBallArithmetic/BallArithmetic.jl)

A Julia package for **rigorous numerical linear algebra** using midpoint-radius (ball) arithmetic.

## Overview

Ball Arithmetic represents real and complex numbers as balls: a center `c` and radius `r` defining the set `{x : |x - c| ≤ r}`. All arithmetic operations are implemented to guarantee that the result contains the true mathematical result, accounting for floating-point rounding errors.

This package focuses on **verified numerical linear algebra**, implementing algorithms from the research of Rump, Miyajima, Ogita, Oishi, and collaborators to provide a posteriori guaranteed error bounds for matrix computations.

### Related Packages

- [IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl) - Interval arithmetic with inf-sup representation
- [ArbNumerics.jl](https://github.com/JeffreySarnoff/ArbNumerics.jl) - Wraps the [Arb](https://arblib.org/) library for arbitrary-precision ball arithmetic

## Features

### Core Types

- **`Ball{T, NT}`** - Scalar ball with center of type `NT` (real or complex) and radius of type `T`
- **`BallMatrix{T, NT}`** - Matrix of balls stored as separate center and radius matrices for efficient BLAS operations
- **`BallVector{T, NT}`** - Vector of balls with the same storage strategy

### Rigorous Matrix Arithmetic

Matrix products are computed using rounding-mode-controlled algorithms following [Rump 1999](https://link.springer.com/article/10.1023/A:1022374804152):

- **MMul3, MMul4, MMul5** - Different precision/speed tradeoffs for matrix multiplication
- **Oishi-Rump product** - Rounding-controlled complex matrix multiplication
- Supports both `Float64` and `BigFloat` precision

### Verified Eigenvalue Enclosures

Multiple algorithms for computing rigorous eigenvalue bounds:

| Algorithm | Reference | Description |
|-----------|-----------|-------------|
| `miyajima_eigenvalue_bounds` | Miyajima 2012 | Block-diagonalization approach |
| `rump_2022a_eigenvalue_bounds` | Rump 2022a | Individual eigenvector error bounds |
| `rump_lange_2023` | Rump-Lange 2023 | Schur-based eigenvalue enclosure |
| `verified_gev_enclosure` | Various | Generalized eigenvalue problem Ax = λBx |

Features include:
- Standard eigenvalue problem (Ax = λx)
- Generalized eigenvalue problem (Ax = λBx)
- Hermitian-optimized variants
- Schur decomposition refinement
- Spectral projector computation
- Riesz projection interfaces

### Verified SVD and Norm Bounds

- **`rigorous_svd`** - Verified singular value decomposition
- **`adaptive_ogita_svd`** - Adaptive precision SVD following Ogita et al.
- **`miyajima_vbd`** - Verified block diagonalization
- **Operator norm bounds**: L1, L2, L∞ norms with rigorous error control
- **Collatz iteration** with underflow protection

### Linear System Solvers

- **Krawczyk method** - Interval Newton for verified solutions
- **HBR method** - Hansen-Bliek-Rohn enclosure
- **Gaussian elimination** with interval pivoting
- **Backward substitution** for triangular systems
- **Sylvester equation solvers** following Miyajima 2013
- Overdetermined system support

### Additional Features

- **Pseudospectra**: Rigorous contour computation via `CertifScripts`
- **Matrix regularity verification**
- **Determinant enclosures**
- **M-matrix classification**

## Quick Start

```julia
using BallArithmetic

# Create a ball matrix
A = BallMatrix([1.0 2.0; 3.0 4.0])

# Add uncertainty
A_uncertain = A + Ball(0.0, 0.01) * I

# Compute verified eigenvalues
result = miyajima_eigenvalue_bounds(A_uncertain)
println("Eigenvalue enclosures: ", result.eigenvalues)

# Solve a linear system with verification
b = BallVector([1.0, 2.0])
x = krawczyk_linear_solve(A, b)
```

## Precision Support

The library supports multiple floating-point types:

```julia
# Float64 (default)
A = BallMatrix([1.0 2.0; 3.0 4.0])

# BigFloat for extended precision
setprecision(BigFloat, 256) do
    A_big = BallMatrix(BigFloat[1 2; 3 4])
    # Operations maintain BigFloat precision
end
```

## Package Extensions

BallArithmetic provides optional extensions that are automatically loaded when the corresponding packages are available:

### IntervalArithmetic.jl Extension

Seamless conversion between `IntervalArithmetic.Interval` and `Ball` types:

```julia
using BallArithmetic, IntervalArithmetic

# Convert Interval to Ball
iv = interval(1.0, 2.0)
b = Ball(iv)

# Convert matrices
A_interval = [interval(1.0, 1.1) interval(2.0, 2.1); interval(3.0, 3.1) interval(4.0, 4.1)]
A_ball = BallMatrix(A_interval)

# Convert Ball back to Interval
iv_back = interval(b)
```

### ArbNumerics.jl Extension

Convert arbitrary-precision `ArbReal` and `ArbComplex` matrices to `BallMatrix{Float64}` while preserving rigorous error bounds:

```julia
using BallArithmetic, ArbNumerics

# High-precision Arb computation
setprecision(ArbReal, 256)
A_arb = ArbReal[ArbReal("1.1") ArbReal("2.2"); ArbReal("3.3") ArbReal("4.4")]

# Convert to BallMatrix with rigorous rounding error accounting
A_ball = BallMatrix(A_arb)
```

### FFTW.jl Extension

Rigorous FFT with a priori error bounds following Higham (1996):

```julia
using BallArithmetic, FFTW

A = BallMatrix(randn(64, 64))
F = fft(A)  # Returns BallMatrix with rigorous error bounds

v = BallVector(randn(128))
f = fft(v)  # Returns BallVector with rigorous error bounds
```

### Distributed.jl Extension

Parallel pseudospectrum certification across multiple Julia workers:

```julia
using BallArithmetic, Distributed
using BallArithmetic.CertifScripts

A = BallMatrix(randn(100, 100))
circle = CertificationCircle(0.0 + 0.0im, 1.0)

# Run certification with 4 workers
result = run_certification(A, circle, 4)

# Or use existing worker pool
addprocs(4)
result = run_certification(A, circle, workers())
```

Features include:
- Adaptive arc refinement with configurable η threshold
- Ogita SVD cache for faster evaluation of nearby points
- BigFloat precision mode for very small radii (< 10⁻¹⁵)
- Snapshot checkpointing for long-running computations

## Implementation Notes

### Rounding Mode Control

The library uses IEEE 754 directed rounding modes via `setrounding(T, RoundUp)` and `setrounding(T, RoundDown)` to obtain rigorous bounds. This requires BLAS libraries that respect rounding modes - the package automatically configures OpenBLAS with `ConsistentFPCSR=1`.

## References

Key papers implemented in this package:

- Rump, S.M. (1999). "Fast and Parallel Interval Arithmetic". *BIT Numerical Mathematics* 39(3), 534-554.
- Miyajima, S. (2012). "Verified computation for the Hermitian eigenvalue problem". *Reliable Computing*.
- Miyajima, S. (2013). "Fast verified computation for the solution of the Sylvester equation". *Numerical Algorithms*.
- Rump, S.M. (2022). "Verified Error Bounds for All Eigenvalues and Eigenvectors of a Matrix".
- Rump, S.M., Lange, M. (2023). "Verified eigenvalue and singular value computation".
- Oishi, S., Rump, S.M. (2024). "Verified norm bounds for matrices".

See the [documentation](https://JuliaBallArithmetic.github.io/BallArithmetic.jl/stable/) for the complete reference list.

## Contributing

Contributions are welcome! Please open issues for bugs or feature requests on the [GitHub repository](https://github.com/JuliaBallArithmetic/BallArithmetic.jl).

## License

This package is licensed under the MIT License.
