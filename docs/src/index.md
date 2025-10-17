```@meta
CurrentModule = BallArithmetic
```

# BallArithmetic

Documentation for [BallArithmetic](https://github.com/JuliaBallArithmetic/BallArithmetic.jl).

In this package we use the tecniques first introduced in Ref. [Rump1999](@cite), following the more recent work Ref. [RevolTheveny2013](@cite)
to implement a rigorous matrix product in mid-radius arithmetic.

This allows to implement numerous algorithms developed by Rump, Miyajima,
Ogita and collaborators to obtain a posteriori guaranteed bounds.

The main object are BallMatrices, i.e., midpoint matrices equipped with
non-negative radii that provide rigorous entrywise enclosures.

## Sylvester equations

[`sylvester_miyajima_enclosure`](@ref) provides a componentwise enclosure for
solutions of the Sylvester equation following the fast verification method of
Ref. [MiyajimaSylvester2013](@cite).  When the data originate from an upper
triangular Schur factor `T`, [`triangular_sylvester_miyajima_enclosure`](@ref)
extracts the blocks `T₁₁`, `T₁₂`, and `T₂₂`, solves the associated Sylvester
system `T₂₂' Y₂ - Y₂ T₁₁' = T₁₂'`, and returns the Miyajima enclosure for the
unknown block `Y₂`.

## `BallMatrix`

`BallMatrix` is the midpoint-radius companion of the scalar [`Ball`](@ref)
type.  The midpoint matrix stores the approximation we would normally
compute in floating-point arithmetic, whereas the radius matrix captures
all sources of uncertainty (input radii, floating-point error, subnormal
padding, …).  Each method documented below updates both components so the
result remains a rigorous enclosure.

### Constructors and accessors

The constructors delegate to the underlying [`BallArray`](@ref) to perform
shape and type validation.  Working through them in order provides a tour
of how the storage is organised:

### Arithmetic

Binary operations follow a common pattern: operate on the midpoint data as
if the values were exact, then grow the radius using outward rounding.
The comments inside `src/types/matrix.jl` walk through the steps in more
detail.


```@repl
using BallArithmetic
A = ones((2, 2))
bA = BallMatrix(A, A/128)
bA^2
```

```@autodocs
Modules = [BallArithmetic]
```













