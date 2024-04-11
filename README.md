# BallArithmetic

This package implements Ball Arithmetic, with a focus on Numerical Linear Algebra.

Ball Arithmetic introduces a type, called `Ball`, 
which represents a subset of the real numbers of the type $(c-r, c+r)$
such that the computed sum of two balls `X, Y` is guaranteed to
contain the set $\{x+y \mid x\in X, y\in Y\}$ of all the possible 
sums of elements of `X` and `Y`; this operation may return a bigger set,
due to rounding, etc... 
Due to this inclusion property, we can use these computed entities
to control the error of numercial operations and even prove theorems in Mathematics.
Other packages implementing a similar phylosophy are 
[IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl), which implements Interval Arithmetic, and [ArbNumerics](https://jeffreysarnoff.github.io/ArbNumerics.jl/stable/) which uses 
Ball Arithmetic and wraps [Arb](https://arblib.org/).

The library does not limit itself to the sum, but implements 
all the other arithmetic operations, but our main interest is to implement
linear algebra operations following the ideas of [Rump, BIT, 1999](https://link.springer.com/article/10.1023/A:1022374804152).


The fundamental idea is that, exchanging precision with speed, to use optimized Blas operations acting on matrices to compute enclosures
of matrix-vector and matrix-matrix products.
To do so, we introduce a type `BallMatrix`, which contains 
a matrix of centers and a matrix of radiuses.
The implementation then guarantees that the result of a matrix-matrix
product or a matrix vector product is contained in the set defined by the 
resulting BallMatrix or BallVector.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaBallArithmetic.github.io/BallArithmetic.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaBallArithmetic.github.io/BallArithmetic.jl/dev/)
[![Build Status](https://github.com/JuliaBallArithmetic/BallArithmetic.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaBallArithmetic/BallArithmetic.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaBallArithmetic/BallArithmetic.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaBallArithmetic/BallArithmetic.jl)
