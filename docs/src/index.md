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

### Rounding-mode controlled products

Some matrix enclosures benefit from explicitly steering the floating-point
rounding mode.  The wrapper [`oishi_MMul`](@ref BallArithmetic.oishi_MMul)
implements the Oishi–Rump product, which evaluates the real and imaginary
parts of `F*G` with downward and upward rounding and returns the result as a
`BallMatrix`.  The routine is particularly useful when replicating the
eigenvalue and singular value enclosures described in Ref.
[@RumpOishi2001](@cite).

Internally we also expose the auxiliary kernels from Ref.
[@Miyajima2010](@cite).  The helpers `_ccrprod`, `_cr`, `_iprod`, and `_ciprod`
implement Algorithms 4–7 and propagate rectangular or ball bounds through
matrix products.  They are available for advanced workflows that need direct
access to the underlying interval data.

```@example oishi
using BallArithmetic
setprecision(BigFloat, 128) do
    F = Complex{BigFloat}[1 + im 2; 3 - im 4]
    G = Complex{BigFloat}[2 - im 1; -1 3 + im]
    B = BallArithmetic.oishi_MMul(F, G)
    (mid(B), rad(B))
end
```

```@autodocs
Modules = [BallArithmetic]
```













