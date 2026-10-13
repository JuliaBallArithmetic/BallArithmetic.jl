"""
    GenericLinearAlgebraExt

Extension for BallArithmetic providing native BigFloat SVD via GenericLinearAlgebra.jl.

GenericLinearAlgebra provides pure-Julia implementations of linear algebra routines
that work with arbitrary numeric types, including BigFloat. For matrices larger than
~50×50, the native BigFloat SVD is faster than Float64 SVD + Ogita refinement.

# Performance comparison (256-bit BigFloat):
- 50×50:   GLA ~2.5x faster than Ogita refinement
- 100×100: GLA ~3x faster than Ogita refinement
- 200×200: GLA ~3x faster than Ogita refinement

# Usage
```julia
using BallArithmetic, GenericLinearAlgebra

A_bf = BigFloat.(randn(100, 100))
result = svd_bigfloat(A_bf)  # Uses GenericLinearAlgebra's native SVD
```
"""
module GenericLinearAlgebraExt

using BallArithmetic
using GenericLinearAlgebra
using LinearAlgebra

"""
    svd_bigfloat(A::AbstractMatrix{T}) where T<:Union{BigFloat, Complex{BigFloat}}

Compute SVD of a BigFloat matrix using GenericLinearAlgebra's native implementation.

This is faster than Float64 SVD + Ogita refinement for matrices larger than ~50×50.

# Returns
`SVD` factorization object with U, S, Vt fields.

# Example
```julia
using BallArithmetic, GenericLinearAlgebra
A = BigFloat.(randn(100, 100))
F = svd_bigfloat(A)
println("σ_min = ", F.S[end])
```
"""
function BallArithmetic.svd_bigfloat(A::AbstractMatrix{T}) where T<:Union{BigFloat, Complex{BigFloat}}
    return svd(A)
end

end # module
