```@meta
CurrentModule = BallArithmetic
```

# BallArithmetic

Documentation for [BallArithmetic](https://github.com/JuliaBallArithmetic/BallArithmetic.jl).

In this package we use the tecniques first introduced in Ref. [Rump1999](@cite), following the more recent work Ref. [RevolTheveny2013](@cite)
to implement a rigorous matrix product in mid-radius arithmetic.

This allows to implement numerous algorithms developed by Rump, Miyajima,
Ogita and collaborators to obtain a posteriori guaranteed bounds.

The main object are BallMatrices, i.e., a couple containing a center matrix and a radius matrix.

```@repl
using BallArithmetic
A = ones((2, 2))
bA = BallMatrix(A, A/128)
bA^2
```













