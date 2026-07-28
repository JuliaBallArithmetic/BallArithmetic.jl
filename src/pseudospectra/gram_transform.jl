# Gram-weighted resolvent certification by change of variables.
#
# For a Hermitian positive definite Gram matrix G, the weighted norm
# ‖x‖_G = √(x*Gx) satisfies ‖x‖_G = ‖L x‖₂ for the Cholesky factor G = L*L,
# hence for every matrix M
#
#     ‖M‖_G = ‖L M L⁻¹‖₂
#
# exactly, with no condition-number penalty.  Since L (A - z)⁻¹ L⁻¹ =
# (L A L⁻¹ - z)⁻¹, a resolvent bound in the G-norm follows from certifying the
# transformed matrix Ã = L A L⁻¹ with the ordinary 2-norm machinery:
#
#     ‖(A - z)⁻¹‖_G = ‖(Ã - z)⁻¹‖₂
#
# This is the sharp alternative to inflating a 2-norm bound by √κ₂(G) through
# the `Cbound` keyword, which pays that factor at every point of the contour.
#
# Rigor: `verified_cholesky` returns an enclosure 𝐋 of the exact factor L*, and
# the ball products enclose {L A L⁻¹ : L ∈ 𝐋}, hence in particular the exact
# Ã* = L* A L*⁻¹.  A 2-norm resolvent bound certified for the enclosure Ã is
# therefore a bound for Ã*, i.e. a bound on ‖(A - z)⁻¹‖_G.

"""
    GramTransform{MT, RT}

Verified data for a Gram-weighted change of variables, as produced by
[`gram_transform`](@ref).

# Fields
- `factor::MT`: enclosure of a factor `L` with `G = L*L`, so `‖x‖_G = ‖Lx‖₂`.
- `factor_inv::MT`: enclosure of `L⁻¹`.
- `cond_factor::RT`: rigorous upper bound on `κ₂(L) = √κ₂(G)`.
- `cond_gram::RT`: rigorous upper bound on `κ₂(G)`. This is the factor the cheap
  `Cbound = √κ₂(G)` route would have paid at *every* point of the contour; the
  change of variables avoids it entirely.
- `gram_residual::Union{Nothing, RT}`: rigorous upper bound on
  `‖L*L - G‖₂ / ‖G‖₂`, or `nothing` when no `gram` was supplied (then `L`
  defines the norm and there is nothing to compare against).
- `source::Symbol`: `:cholesky`, `:diagonal` or `:user_factor` — how the factor,
  and hence the certified norm, was obtained.
"""
struct GramTransform{MT <: BallMatrix, RT <: Real}
    factor::MT
    factor_inv::MT
    cond_factor::RT
    cond_gram::RT
    gram_residual::Union{Nothing, RT}
    source::Symbol
end

function Base.show(io::IO, gt::GramTransform)
    print(io, "GramTransform(n = ", size(gt.factor, 1),
        ", κ₂(G) ≤ ", gt.cond_gram,
        gt.gram_residual === nothing ? "" : ", ‖L*L-G‖/‖G‖ ≤ $(gt.gram_residual)",
        ", ", gt.source, ")")
end

#####################################################################
# Directed-rounding helpers                                         #
#####################################################################

_gram_up(f::Function, ::Type{T}) where {T} = setrounding(f, T, RoundUp)
_gram_down(f::Function, ::Type{T}) where {T} = setrounding(f, T, RoundDown)

#####################################################################
# Rigorous radius-type conversion (outward rounding)                #
#####################################################################

_gram_conv_err(x::Real, y::Real) = abs(x - y)
_gram_conv_err(x::Complex, y::Complex) = abs(real(x) - real(y)) + abs(imag(x) - imag(y))

"""
    _gram_convert(M::BallMatrix, T2)

Re-type the enclosure `M` so its radii have type `T2`, rounding *outward* so
that the result still encloses everything `M` did.  This is what lets the
Cholesky run in BigFloat — where the factor enclosure is far tighter — while the
certifier still receives a `Float64` matrix.
"""
function _gram_convert(M::BallMatrix{T, NT}, ::Type{T2}) where {T, NT, T2 <: AbstractFloat}
    T === T2 && return M
    NT2 = NT <: Complex ? Complex{T2} : T2

    c2 = convert.(NT2, M.c)
    back = convert.(NT, c2)
    r2 = Matrix{T2}(undef, size(M.r, 1), size(M.r, 2))

    for i in eachindex(r2)
        err = _gram_conv_err(M.c[i], back[i])
        total = _gram_up(T) do
            M.r[i] + err
        end
        r2[i] = T2(total, RoundUp)
    end

    return BallMatrix(c2, r2)
end

"""
    _gram_promote(A::BallMatrix, T2)

Widen `A` to radius type `T2` (exact when `T2` is the wider type).
"""
function _gram_promote(A::BallMatrix{T, NT}, ::Type{T2}) where {T, NT, T2 <: AbstractFloat}
    T === T2 && return A
    NT2 = NT <: Complex ? Complex{T2} : T2
    return BallMatrix(convert.(NT2, A.c), convert.(T2, A.r))
end

#####################################################################
# Verified inverses                                                 #
#####################################################################

"""
    _verify_inverse(F::BallMatrix, X::BallMatrix)

Turn an *approximate* inverse `X ≈ F⁻¹` into a rigorous enclosure of `F⁻¹`.

With `R = I - X F` we have `F⁻¹ = (I - R)⁻¹ X`, hence
`‖F⁻¹ - X‖₂ ≤ ‖R‖₂‖X‖₂ / (1 - ‖R‖₂)` whenever `‖R‖₂ < 1`.  Every entry of the
deviation is bounded by that quantity, so inflating `X` uniformly by it yields a
valid enclosure — and `‖R‖₂ < 1` also proves `F` invertible.  Looser than a
triangular back-substitution, but it runs at BLAS speed and applies to an
arbitrary invertible factor.
"""
function _verify_inverse(F::BallMatrix{T, NT}, X::BallMatrix{T, NT}) where {T, NT}
    n = size(F, 1)
    Id = _identity_ballmatrix(n, F)
    R = Id - X * F

    nR = upper_bound_L2_opnorm(R)
    if !(nR < 1)
        throw(ArgumentError("could not verify the inverse of the Gram factor: " *
                            "‖I - X·L‖₂ ≤ $nR is not < 1. The factor is too " *
                            "ill-conditioned at this precision."))
    end
    nX = upper_bound_L2_opnorm(X)

    # Denominator rounded *down* so that the quotient stays an upper bound.
    denom = _gram_down(T) do
        one(T) - nR
    end
    δ = _gram_up(T) do
        (nR * nX) / denom
    end
    radii = _gram_up(T) do
        X.r .+ δ
    end

    return BallMatrix(X.c, radii)
end

"""
    _triangular_inverse(L::BallMatrix)

Enclosure of `L⁻¹` for upper (or lower) triangular `L`, by rigorous back-
(resp. forward-) substitution against the identity.  Tighter than
[`_verify_inverse`](@ref), but `O(n³)` in scalar ball operations rather than
BLAS-speed.
"""
function _triangular_inverse(L::BallMatrix)
    n = size(L, 1)
    Id = _identity_ballmatrix(n, L)
    if istriu(L.c) && istriu(L.r)
        return backward_substitution(L, Id)
    elseif istril(L.c) && istril(L.r)
        return forward_substitution(L, Id)
    else
        throw(ArgumentError("_triangular_inverse requires a triangular factor"))
    end
end

function _gram_factor_inverse(L::BallMatrix, method::Symbol)
    n = size(L, 1)
    triangular = (istriu(L.c) && istriu(L.r)) || (istril(L.c) && istril(L.r))

    chosen = method === :auto ? ((triangular && n <= 256) ? :backsub : :verify) : method

    if chosen === :backsub
        triangular || throw(ArgumentError("inverse_method = :backsub requires a " *
                                          "triangular Gram factor; use :verify"))
        return _triangular_inverse(L)
    elseif chosen === :verify
        X = BallMatrix(inv(Matrix(L.c)))
        return _verify_inverse(L, _gram_promote(X, eltype(L.r)))
    else
        throw(ArgumentError("unknown inverse_method $(repr(method)); " *
                            "expected :auto, :backsub or :verify"))
    end
end

#####################################################################
# Building the factor                                               #
#####################################################################

function _check_gram_shape(G::AbstractMatrix, hermitian_tol::Real)
    n = size(G, 1)
    size(G, 2) == n ||
        throw(DimensionMismatch("the Gram matrix must be square, got $(size(G))"))

    scale = maximum(abs, G)
    iszero(scale) &&
        throw(ArgumentError("the Gram matrix is zero, hence not positive definite"))
    deviation = maximum(abs, G - G')
    if deviation > hermitian_tol * scale
        throw(ArgumentError("the Gram matrix is not Hermitian " *
                            "(max|G - G*| = $deviation, max|G| = $scale). The " *
                            "G-norm is only defined for Hermitian G; symmetrise " *
                            "it explicitly if that is what you intend."))
    end
    return n
end

_is_diagonal_gram(G::AbstractMatrix) = all(iszero, G - Diagonal(diag(G)))

"""
    _diagonal_gram_factor(G)

Diagonal fast path: the exact Cholesky factor is `L = diag(√dᵢ)`, enclosed
entrywise by the rigorous ball `sqrt`, and `L⁻¹ = diag(1/√dᵢ)` by the ball
reciprocal.  Avoids both the Cholesky and the `O(n³)` inverse, which matters
because a diagonal weight is the common case (orthogonal-polynomial and Fourier
bases).
"""
function _diagonal_gram_factor(G::AbstractMatrix{ET}) where {ET}
    RG = real(ET)
    n = size(G, 1)
    d = real.(diag(G))

    L_c = zeros(RG, n, n);  L_r = zeros(RG, n, n)
    Li_c = zeros(RG, n, n); Li_r = zeros(RG, n, n)

    for i in 1:n
        d[i] > 0 || throw(ArgumentError("the Gram matrix is not positive definite: " *
                                       "diagonal entry $i is $(d[i])"))
        s = sqrt(Ball(RG(d[i]), zero(RG)))
        inv_s = Ball(one(RG), zero(RG)) / s
        L_c[i, i] = s.c;      L_r[i, i] = s.r
        Li_c[i, i] = inv_s.c; Li_r[i, i] = inv_s.r
    end

    return BallMatrix(L_c, L_r), BallMatrix(Li_c, Li_r)
end

"""
    _gram_residual_bound(G, L)

Rigorous upper bound on `‖L*L - G‖₂ / ‖G‖₂`.  The numerator is bounded with
[`upper_bound_L2_opnorm`](@ref) over the ball product; the denominator needs a
rigorous *lower* bound on `‖G‖₂`, for which `maxᵢ Gᵢᵢ = maxᵢ eᵢ*Geᵢ` serves for
Hermitian positive definite `G`.
"""
function _gram_residual_bound(G::AbstractMatrix, L::BallMatrix{WT, NT}) where {WT, NT}
    n = size(G, 1)
    L_adj = BallMatrix(collect(L.c'), collect(L.r'))
    residual = L_adj * L - BallMatrix(convert.(NT, G))
    numerator = upper_bound_L2_opnorm(residual)

    denominator = _gram_down(WT) do
        maximum(i -> real(convert(NT, G[i, i])), 1:n)
    end
    denominator > 0 || return WT(Inf)
    return _gram_up(WT) do
        numerator / denominator
    end
end

"""
    gram_transform(gram; kwargs...) -> GramTransform

Build the verified data for certifying resolvent bounds in the weighted norm
`‖x‖_G = √(x*Gx)` induced by the Hermitian positive definite Gram matrix
`gram`, via the change of variables `Ã = L A L⁻¹` with `G = L*L`.

Pass the result to [`apply_gram_transform`](@ref), or — more conveniently — pass
`gram` directly as a keyword to `run_certification`, `run_certification_ogita`
or `run_certification_parametric`, which do both steps for you.

The factor comes from [`verified_cholesky`](@ref), so positive definiteness of
`gram` is *proven*, not assumed; the call raises if it cannot be established.

# Keyword Arguments
- `factor = nothing`: use this factor instead of computing a Cholesky, given as
  a plain matrix or a `BallMatrix` enclosure. Worth supplying when it is
  analytically known (diagonal weights, banded factors). Note that with
  `factor` the *supplied* factor defines the certified norm `‖Lx‖₂`; it is not
  checked against `gram`, and `gram` may be omitted entirely.
- `factor_inv = nothing`: an approximate inverse of the factor. It is *verified*
  before use — an unverified inverse would silently break rigor, so a bad one
  raises rather than being trusted. Supplying it replaces the `O(n³)` scalar
  back-substitution with a BLAS-speed inverse plus verification.
- `radius_type = Float64`: radius type of the enclosures handed to the
  certifier.
- `use_bigfloat = true`: precision of the verified Cholesky, forwarded to
  [`verified_cholesky`](@ref). The BigFloat factor enclosure is many orders of
  magnitude tighter (radii ~1e-31 against ~1e-14 for a `Float64` Gram matrix),
  and that tightness propagates straight into `Ã`, so it is worth the cost; the
  transformed matrix is rounded outward to `radius_type` afterwards. Set it to
  `false` to trade enclosure width for speed.
- `precision_bits = 256`: BigFloat precision when `use_bigfloat = true`.
- `inverse_method = :auto`: `:backsub` for the tight `O(n³)` scalar triangular
  substitution, `:verify` for the BLAS-speed approximate-inverse-plus-
  verification route. `:auto` picks `:backsub` for triangular factors with
  `n ≤ 256` and `:verify` otherwise.
- `hermitian_tol = 1e-10`: relative tolerance of the Hermitian check on `gram`.

# Example
```julia
G = collect(Diagonal([1.0, 4.0, 9.0]))     # weighted ℓ² norm
gt = gram_transform(G)
Ã = apply_gram_transform(BallMatrix(A), gt)
```
"""
function gram_transform(gram;
        factor = nothing,
        factor_inv = nothing,
        radius_type::Type = Float64,
        use_bigfloat::Bool = true,
        precision_bits::Int = 256,
        inverse_method::Symbol = :auto,
        hermitian_tol::Real = 1e-10)

    gram === nothing && factor === nothing &&
        throw(ArgumentError("gram_transform needs either a Gram matrix or a factor"))
    # Validated up front so a typo is caught even on the paths that never consult it
    inverse_method in (:auto, :backsub, :verify) ||
        throw(ArgumentError("unknown inverse_method $(repr(inverse_method)); " *
                            "expected :auto, :backsub or :verify"))

    G = gram === nothing ? nothing : collect(gram)
    G === nothing || _check_gram_shape(G, hermitian_tol)

    L = nothing
    L_inv = nothing
    source = :cholesky

    if factor !== nothing
        L = factor isa BallMatrix ? factor : BallMatrix(collect(factor))
        source = :user_factor
        G === nothing || size(L, 1) == size(G, 1) ||
            throw(DimensionMismatch("factor and gram have incompatible sizes"))
    elseif _is_diagonal_gram(G)
        L, L_inv = _diagonal_gram_factor(G)
        source = :diagonal
    else
        chol = verified_cholesky(G; precision_bits = precision_bits,
            use_bigfloat = use_bigfloat)
        chol.success ||
            throw(ArgumentError("verified Cholesky of the Gram matrix failed: G " *
                                "could not be proven positive definite. Try " *
                                "use_bigfloat = true or a larger precision_bits."))
        L = chol.G
    end

    WT = eltype(L.r)

    if factor_inv !== nothing
        X = factor_inv isa BallMatrix ? factor_inv : BallMatrix(collect(factor_inv))
        L_inv = _verify_inverse(L, _gram_promote(X, WT))
    elseif L_inv === nothing
        L_inv = _gram_factor_inverse(L, inverse_method)
    end

    residual = G === nothing ? nothing : _gram_residual_bound(G, L)

    nL = upper_bound_L2_opnorm(L)
    nLi = upper_bound_L2_opnorm(L_inv)
    cond_factor = _gram_up(WT) do
        WT(nL) * WT(nLi)
    end
    cond_gram = _gram_up(WT) do
        cond_factor * cond_factor
    end

    RT = radius_type
    return GramTransform(L, L_inv, RT(cond_factor), RT(cond_gram),
        residual === nothing ? nothing : RT(residual), source)
end

"""
    apply_gram_transform(A::BallMatrix, gt::GramTransform) -> BallMatrix

Return the enclosure of `Ã = L A L⁻¹`, whose 2-norm resolvent is the `G`-norm
resolvent of `A`:

    ‖(A - z)⁻¹‖_G = ‖(Ã - z)⁻¹‖₂    for every z ∉ σ(A) = σ(Ã)

The product is formed in the factor's working precision and then rounded outward
to `A`'s radius type, so a BigFloat factor still yields a `Float64` enclosure.
"""
function apply_gram_transform(A::BallMatrix{T, NT}, gt::GramTransform) where {T, NT}
    size(gt.factor, 1) == size(A, 1) ||
        throw(DimensionMismatch("Gram factor has size $(size(gt.factor)) but A has " *
                                "size $(size(A))"))
    WT = eltype(gt.factor.r)
    Aw = _gram_promote(A, WT)
    transformed = (gt.factor * Aw) * gt.factor_inv
    return _gram_convert(transformed, T)
end

"""
    _prepare_gram(A, gram, factor, factor_inv, schur_data; kwargs...)

Shared front end for the certificators.  Returns `(A′, gram_info)` where `A′` is
either `A` unchanged (no Gram matrix requested, `gram_info === nothing`) or the
transformed matrix `Ã = L A L⁻¹` together with its [`GramTransform`](@ref).
"""
function _prepare_gram(A::BallMatrix{T, NT}, gram, factor, factor_inv, schur_data;
        kwargs...) where {T, NT}
    if gram === nothing && factor === nothing
        factor_inv === nothing ||
            throw(ArgumentError("gram_factor_inv was supplied without gram or gram_factor"))
        return A, nothing
    end

    schur_data === nothing ||
        throw(ArgumentError("schur_data cannot be combined with gram / gram_factor: " *
                            "the Schur form would have to belong to the transformed " *
                            "matrix Ã = L·A·L⁻¹, not to A. Transform A yourself with " *
                            "apply_gram_transform and pass the matching schur_data, " *
                            "or drop schur_data."))

    gt = gram_transform(gram; factor = factor, factor_inv = factor_inv,
        radius_type = real(T), kwargs...)
    return apply_gram_transform(A, gt), gt
end
