"""
    GenericSchurExt

Extension enabling NJD-based verified block diagonalisation for
arbitrary-precision matrices (BigFloat, Double64, etc.).

Loading GenericSchur.jl provides `schur` and `ordschur` for generic
`AbstractFloat` types, which the NJD algorithm needs for Phase 1.
GenericLinearAlgebra provides `svd` and `svdvals` for generic types,
which the SVD staircase (Phase 2) needs.

Once both packages are loaded, `miyajima_vbd_njd` works with
`BallMatrix{BigFloat}` input without any additional code changes.

# Usage
```julia
using BallArithmetic, GenericSchur, GenericLinearAlgebra

A = BallMatrix(BigFloat.(randn(4, 4)))
result = miyajima_vbd_njd(A)
```

# References

* [Miyajima2021NJD](@cite) Miyajima, S. (2021). "Verified computation of the matrix
  exponential", J. Comput. Appl. Math. 396, 113614.
* [KagstromRuhe1980](@cite) Kågström, B. & Ruhe, A. (1980). "An algorithm for numerical
  computation of the Jordan normal form of a complex matrix", ACM Trans.
  Math. Softw. 6(3), 398–419. (Algorithm 560)
"""
module GenericSchurExt

using BallArithmetic
using GenericSchur
using GenericLinearAlgebra

# GenericSchur extends LinearAlgebra.schur!/ordschur! for generic
# AbstractFloat types via type dispatch — no wrapper code needed.
# GenericLinearAlgebra extends LinearAlgebra.svd!/svdvals! similarly.
#
# This extension exists to:
# 1. Declare the dependency so Pkg resolves GenericSchur correctly
# 2. Provide the _njd_schur_available flag for helpful error messages
# 3. Serve as documentation that both packages are needed for BigFloat NJD

function __init__()
    BallArithmetic._set_generic_schur_available(true)
end

end # module
