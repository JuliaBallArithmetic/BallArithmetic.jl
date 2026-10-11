# The exported LU decomposition with error bounds, and what the exported decompositions share:
# the working arithmetic selected by `use_bigfloat` and `precision_bits`, the input as a ball
# matrix in it, and the relative residual of an enclosure. The method is in `rump_ogita_2024.jl`.

# Helper to get appropriate BigFloat type (handles Complex)
_bigfloat_type(::Type{T}) where {T <: Real} = BigFloat
_bigfloat_type(::Type{Complex{T}}) where {T <: Real} = Complex{BigFloat}
_to_bigfloat(M::AbstractMatrix{T}) where {T} = convert.(_bigfloat_type(T), M)
_to_bigfloat(v::AbstractVector{T}) where {T} = convert.(_bigfloat_type(T), v)

# Helper to get the working float type (Float64 or BigFloat)
function _working_type(::Type{T}, use_bigfloat::Bool) where {T <: Real}
    use_bigfloat ? BigFloat : Float64
end
function _working_type(::Type{Complex{T}}, use_bigfloat::Bool) where {T <: Real}
    use_bigfloat ? Complex{BigFloat} : ComplexF64
end
function _to_working(M::AbstractMatrix{T}, use_bigfloat::Bool) where {T}
    use_bigfloat ? _to_bigfloat(M) : convert.(T <: Complex ? ComplexF64 : Float64, M)
end
function _to_working(v::AbstractVector{T}, use_bigfloat::Bool) where {T}
    use_bigfloat ? _to_bigfloat(v) : convert.(T <: Complex ? ComplexF64 : Float64, v)
end

# Runs `f` in the working arithmetic of the exported decompositions: Float64, or BigFloat at
# `precision_bits`.
_decomposition_arithmetic(f, use_bigfloat::Bool, precision_bits::Integer) =
    use_bigfloat ? setprecision(f, BigFloat, precision_bits) : f()

# The input as a ball matrix in the working arithmetic; called inside
# `_decomposition_arithmetic`. An entry with more digits than the working precision is rounded
# to nearest and the rounding error, which is computed exactly at the precision of the entry and
# then rounded up, goes into the radius, so that the ball contains the input.
function _decomposition_input(A::AbstractMatrix{S}, use_bigfloat::Bool) where {S}
    use_bigfloat || return BallMatrix(Matrix{S <: Complex ? ComplexF64 : Float64}(A))
    p = precision(BigFloat)
    nearest(x::Real) = BigFloat(x; precision = p)
    function excess(x::Real, c::BigFloat)
        px = precision(x)
        px <= p && return BigFloat(0; precision = p)
        d = setprecision(() -> abs(BigFloat(x) - c), BigFloat, px)
        return BigFloat(d, RoundUp; precision = p)
    end
    c = Matrix{S <: Complex ? Complex{BigFloat} : BigFloat}(undef, size(A))
    r = Matrix{BigFloat}(undef, size(A))
    for i in eachindex(A)
        x = A[i]
        if S <: Complex
            cr, ci = nearest(real(x)), nearest(imag(x))
            c[i] = complex(cr, ci)
            r[i] = add_up(excess(real(x), cr), excess(imag(x), ci))
        else
            c[i] = nearest(x)
            r[i] = excess(x, c[i])
        end
    end
    return BallMatrix(c, r)
end

# What a failed verification returns: nothing is enclosed.
_unverified_ball(::BallMatrix{T, CT}, m::Integer, n::Integer) where {T, CT} =
    BallMatrix(fill(CT(NaN), m, n), fill(T(Inf), m, n))

# An upper bound of ‖R̃‖_∞ / ‖Ã‖_∞ over the matrices R̃ of the ball R and Ã of the ball A.
function _relative_residual_bound(R::BallMatrix{T}, A::BallMatrix{T}) where {T}
    num = T(upper_bound_L_inf_opnorm(R))
    Am, Ar = mid(A), rad(A)
    den = zero(T)
    for i in axes(Am, 1)
        s = zero(T)
        for j in axes(Am, 2)
            s = add_down(s, max(sub_down(abs_down(Am[i, j]), Ar[i, j]), zero(T)))
        end
        den = max(den, s)
    end
    return den > 0 ? div_up(num, den) : T(Inf)
end

"""
    VerifiedLUResult{LM, UM, RT}

Result of [`verified_lu`](@ref).

# Fields
- `L::LM`: ball matrix containing the unit lower triangular (trapezoidal) factor
- `U::UM`: ball matrix containing the upper triangular (trapezoidal) factor
- `p::Vector{Int}`: row permutation
- `success::Bool`: whether the verification succeeded; when `false` the radii are infinite and
  nothing is proved
- `residual_norm::RT`: an upper bound of `‖L̃Ũ − A[p, q]‖_∞ / ‖A‖_∞` over all `L̃ ∈ L`, `Ũ ∈ U`
- `q::Vector{Int}`: column permutation, the identity unless the matrix has more columns than rows

# Statement
When `success` is `true`, `A[p, q]` has a unique decomposition `LU` with `L` unit lower and `U`
upper triangular, and these factors lie in the ball matrices `L` and `U`.

# Reference
S. M. Rump and T. Ogita, *Verified Error Bounds for Matrix Decompositions*, SIAM J. Matrix Anal.
Appl. **45**(4) (2024), 2155-2183, doi 10.1137/24M165096X, Section 3.
"""
struct VerifiedLUResult{LM <: BallMatrix, UM <: BallMatrix, RT <: Real}
    L::LM
    U::UM
    p::Vector{Int}
    success::Bool
    residual_norm::RT
    q::Vector{Int}
end
VerifiedLUResult(L, U, p, success, residual_norm) =
    VerifiedLUResult(L, U, p, success, residual_norm, collect(1:size(U, 2)))

"""
    _lu_perturbed_identity(E::AbstractMatrix{T}; E_rad=nothing, precision_bits::Int=256, use_bigfloat::Bool=true) where T

Compute verified LU decomposition of I + E where E is a small perturbation.

This is the core algorithm from Section 3.1 of Rump & Ogita (2024).
For ‖E‖∞ < 1, the matrix I + E has a unique LU decomposition.

# Arguments
- `E_rad`: optional nonnegative radius matrix turning `E` into the interval
  matrix `E ± E_rad`, so that the returned enclosures are valid for *every*
  matrix in that interval. All bounds of Section 3.1 depend on `E` only
  through `|E|` and are monotone in it, so they are evaluated at `|E| + E_rad`;
  the offsets are centred at `E`, hence `E_rad` is added to their radii.
  Defaults to an exact (point) `E`.
- `precision_bits`: Precision for BigFloat computation (ignored if use_bigfloat=false)
- `use_bigfloat`: If true, use BigFloat for high precision; if false, use Float64 with directed rounding

# Returns
Tuple (L_offset, U_offset, L_inv_offset, U_inv_offset, success) where:
- L = I + L_offset (L_offset is strictly lower triangular)
- U = I + U_offset (U_offset is upper triangular with zero diagonal for square case)
- L⁻¹ = I + L_inv_offset
- U⁻¹ = I + U_inv_offset

# Algorithm (from equations 3.1-3.7 in the paper)
The key insight is that L[i,k] - E[i,k] can be bounded by an outer product,
allowing O(n²) computation of verified bounds.
"""
function _lu_perturbed_identity(E::AbstractMatrix{T};
        E_rad = nothing,
        precision_bits::Int = 256,
        use_bigfloat::Bool = true) where {T}
    m, n = size(E)
    mn = min(m, n)

    # Set precision for rigorous computation (only needed for BigFloat mode)
    old_prec = precision(BigFloat)
    if use_bigfloat
        setprecision(BigFloat, precision_bits)
    end

    try
        # Convert to working precision type
        WT = _working_type(T, use_bigfloat)
        RWT = real(WT)
        E_w = _to_working(E, use_bigfloat)

        # Radius of the input perturbation.  All the bounds below depend on E
        # only through |E| and are monotone in it, so they are evaluated at
        # |E| + E_rad; the offset midpoints stay centred at E and pick up E_rad
        # in their radii.
        R_w = E_rad === nothing ? zeros(RWT, m, n) :
              convert.(RWT, _to_working(E_rad, use_bigfloat))
        any(<(0), R_w) && throw(ArgumentError("E_rad must be nonnegative"))
        absE_w = setrounding(RWT, RoundUp) do
            abs.(E_w) .+ R_w
        end
        R_stril = _strict_lower_triangular(R_w)
        R_triu = _upper_triangular(R_w)

        # Check convergence condition: ‖E_n‖∞ < 1
        absE_n = m >= n ? absE_w : absE_w[1:m, 1:m]
        E_norm = setrounding(RWT, RoundUp) do
            maximum(sum(absE_n, dims = 2))
        end

        if E_norm >= 1
            # Cannot verify - perturbation too large
            return nothing, nothing, nothing, nothing, false
        end

        # Extract triangular parts
        E_stril = _strict_lower_triangular(E_w)  # Strictly lower triangular
        E_triu = _upper_triangular(E_w)          # Upper triangular (including diagonal)
        absE_stril = _strict_lower_triangular(absE_w)
        absE_triu = _upper_triangular(absE_w)

        # Equation (3.1): Bound on |L^[ℓ] - E^[ℓ]|
        # |L^[ℓ] - E^[ℓ]| ≤ (sum(|E^[ℓ]|, 2) · max(|E^[u]_n|))^[ℓ] / (1 - ‖E_n‖∞)
        row_sums_E_stril = setrounding(RWT, RoundUp) do
            vec(sum(absE_stril, dims = 2))
        end
        col_maxes_E_triu = vec(maximum(absE_triu[1:mn, 1:mn], dims = 1))

        # Outer product bound (strictly lower triangular part only).  The
        # denominator is rounded down so that the quotients stay upper bounds.
        denom = setrounding(RWT, RoundDown) do
            one(RWT) - E_norm
        end
        denom > 0 || return nothing, nothing, nothing, nothing, false
        Delta_L = zeros(RWT, m, mn)
        setrounding(RWT, RoundUp) do
            for j in 1:(mn - 1)
                for i in (j + 1):m
                    Delta_L[i, j] = row_sums_E_stril[i] * col_maxes_E_triu[j] / denom
                end
            end
        end

        # L = I + E^[ℓ] + C^[ℓ] where |C^[ℓ]| ≤ Δ^[ℓ]
        L_offset_mid = E_stril[1:m, 1:mn]
        L_offset_rad = setrounding(RWT, RoundUp) do
            Delta_L .+ R_stril[1:m, 1:mn]
        end

        # Equation (3.4): Bound on L⁻¹
        # L⁻¹ = I - E^[ℓ] + δ where |δ| ≤ Δ^[ℓ] + (sum(G,2)·max(G))^[ℓ] / (1 - ‖G‖∞)
        G, G_norm = setrounding(RWT, RoundUp) do
            Gm = absE_stril[1:m, 1:mn] .+ Delta_L
            Gm, maximum(sum(Gm, dims = 2))
        end

        if G_norm >= 1
            return nothing, nothing, nothing, nothing, false
        end

        row_sums_G = setrounding(RWT, RoundUp) do
            vec(sum(G, dims = 2))
        end
        col_maxes_G = vec(maximum(G, dims = 1))

        G_denom = setrounding(RWT, RoundDown) do
            one(RWT) - G_norm
        end
        G_denom > 0 || return nothing, nothing, nothing, nothing, false

        delta_L_inv = copy(Delta_L)
        setrounding(RWT, RoundUp) do
            for j in 1:(mn - 1)
                for i in (j + 1):m
                    delta_L_inv[i, j] += row_sums_G[i] * col_maxes_G[j] / G_denom
                end
            end
        end

        L_inv_offset_mid = -E_stril[1:m, 1:mn]
        L_inv_offset_rad = setrounding(RWT, RoundUp) do
            delta_L_inv .+ R_stril[1:m, 1:mn]
        end

        # Equation (3.5): Bound on U
        # U = I_n + E^[u]_n + C^[u] where the bound involves L
        # For m ≥ n case
        if m >= n
            B = copy(absE_triu[1:n, 1:n])
            for j in 1:(n - 1)
                for i in (j + 1):n
                    B[i, j] = Delta_L[i, j]
                end
            end

            row_sums_GL = setrounding(RWT, RoundUp) do
                vec(sum(G[1:n, 1:n], dims = 2))
            end
            col_maxes_B = vec(maximum(B, dims = 1))

            GL_norm = maximum(row_sums_GL)
            if GL_norm >= 1
                return nothing, nothing, nothing, nothing, false
            end

            GL_denom = setrounding(RWT, RoundDown) do
                one(RWT) - GL_norm
            end
            GL_denom > 0 || return nothing, nothing, nothing, nothing, false

            Delta_U = zeros(RWT, n, n)
            setrounding(RWT, RoundUp) do
                for j in 1:n
                    for i in 1:j  # Upper triangular including diagonal
                        Delta_U[i, j] = row_sums_GL[i] * col_maxes_B[j] / GL_denom
                    end
                end
            end

            U_offset_mid = E_triu[1:n, 1:n]
            U_offset_rad = setrounding(RWT, RoundUp) do
                Delta_U .+ R_triu[1:n, 1:n]
            end

            # U⁻¹ bounds (equation 3.7)
            GU, GU_norm = setrounding(RWT, RoundUp) do
                GUm = absE_triu[1:n, 1:n] .+ Delta_U
                GUm, maximum(sum(GUm, dims = 2))
            end

            if GU_norm >= 1
                return nothing, nothing, nothing, nothing, false
            end

            row_sums_GU = setrounding(RWT, RoundUp) do
                vec(sum(GU, dims = 2))
            end
            col_maxes_GU = vec(maximum(GU, dims = 1))

            GU_denom = setrounding(RWT, RoundDown) do
                one(RWT) - GU_norm
            end
            GU_denom > 0 || return nothing, nothing, nothing, nothing, false

            delta_U_inv = copy(Delta_U)
            setrounding(RWT, RoundUp) do
                for j in 1:n
                    for i in 1:j
                        delta_U_inv[i, j] += row_sums_GU[i] * col_maxes_GU[j] / GU_denom
                    end
                end
            end

            U_inv_offset_mid = -E_triu[1:n, 1:n]
            U_inv_offset_rad = setrounding(RWT, RoundUp) do
                delta_U_inv .+ R_triu[1:n, 1:n]
            end
        else
            # m < n case: L is m×m, U is m×n
            # Similar but with different dimensions
            E_m = E_w[1:m, 1:m]
            E_m_triu = _upper_triangular(E_m)
            absE_m_triu = _upper_triangular(absE_w[1:m, 1:m])
            R_m_triu = _upper_triangular(R_w[1:m, 1:m])
            R_U = [R_m_triu R_w[1:m, (m + 1):n]]

            B_m = [absE_m_triu absE_w[1:m, (m + 1):n]]
            for j in 1:(m - 1)
                for i in (j + 1):m
                    B_m[i, j] = Delta_L[i, j]
                end
            end

            row_sums_GL = setrounding(RWT, RoundUp) do
                vec(sum(G[1:m, 1:m], dims = 2))
            end
            col_maxes_B = vec(maximum(B_m, dims = 1))

            GL_norm = maximum(row_sums_GL)
            if GL_norm >= 1
                return nothing, nothing, nothing, nothing, false
            end

            GL_denom = setrounding(RWT, RoundDown) do
                one(RWT) - GL_norm
            end
            GL_denom > 0 || return nothing, nothing, nothing, nothing, false

            Delta_U = zeros(RWT, m, n)
            setrounding(RWT, RoundUp) do
                for j in 1:n
                    for i in 1:min(j, m)
                        Delta_U[i, j] = row_sums_GL[i] * col_maxes_B[j] / GL_denom
                    end
                end
            end

            U_offset_mid = [E_m_triu E_w[1:m, (m + 1):n]]
            U_offset_rad = setrounding(RWT, RoundUp) do
                Delta_U .+ R_U
            end

            # U⁻¹ for m < n (only left m×m block is invertible)
            GU, GU_norm = setrounding(RWT, RoundUp) do
                GUm = absE_m_triu .+ Delta_U[1:m, 1:m]
                GUm, maximum(sum(GUm, dims = 2))
            end

            if GU_norm >= 1
                return nothing, nothing, nothing, nothing, false
            end

            row_sums_GU = setrounding(RWT, RoundUp) do
                vec(sum(GU, dims = 2))
            end
            col_maxes_GU = vec(maximum(GU, dims = 1))

            GU_denom = setrounding(RWT, RoundDown) do
                one(RWT) - GU_norm
            end
            GU_denom > 0 || return nothing, nothing, nothing, nothing, false

            delta_U_inv = zeros(RWT, m, m)
            setrounding(RWT, RoundUp) do
                for j in 1:m
                    for i in 1:j
                        delta_U_inv[i, j] = Delta_U[i, j] +
                                            row_sums_GU[i] * col_maxes_GU[j] / GU_denom
                    end
                end
            end

            U_inv_offset_mid = -E_m_triu
            U_inv_offset_rad = setrounding(RWT, RoundUp) do
                delta_U_inv .+ R_m_triu
            end
        end

        return (L_offset_mid, L_offset_rad), (U_offset_mid, U_offset_rad),
        (L_inv_offset_mid, L_inv_offset_rad), (U_inv_offset_mid, U_inv_offset_rad), true

    finally
        setprecision(BigFloat, old_prec)
    end
end

"""
    _strict_lower_triangular(A::AbstractMatrix)

Extract strictly lower triangular part of A (below diagonal).
"""
function _strict_lower_triangular(A::AbstractMatrix{T}) where {T}
    m, n = size(A)
    L = zeros(T, m, n)
    for j in 1:min(m - 1, n)
        for i in (j + 1):m
            L[i, j] = A[i, j]
        end
    end
    return L
end

"""
    _upper_triangular(A::AbstractMatrix)

Extract upper triangular part of A (including diagonal).
"""
function _upper_triangular(A::AbstractMatrix{T}) where {T}
    m, n = size(A)
    U = zeros(T, m, n)
    for j in 1:n
        for i in 1:min(j, m)
            U[i, j] = A[i, j]
        end
    end
    return U
end

"""
    verified_lu(A::AbstractMatrix; precision_bits = 256, use_bigfloat = true)

Inclusions of the factors of the LU decomposition of the `m × n` matrix `A`, by Section 3 of
Rump and Ogita (2024); the method is described at [`_rumpogita2024_lu`](@ref). Returns a
[`VerifiedLUResult`](@ref): when `success` is `true`, `A[p, q] = LU` with `L` of size
`m × min(m, n)` unit lower triangular and `U` of size `min(m, n) × n` upper triangular, both in
the ball matrices returned. `p` comes from partial pivoting; `q` is the identity for `m ≥ n`, and
for `m < n` the column permutation of Section 3.4 of the paper.

# Arguments
- `use_bigfloat`: with `true`, the default, the computation runs in BigFloat at `precision_bits`
  and the factors are enclosed to about that precision; with `false` it runs in Float64 and the
  radii are of the order of the rounding unit times the size of the factors. A BigFloat input is
  always treated in BigFloat.
- `precision_bits`: the BigFloat precision. A BigFloat input with more digits is rounded to it
  and the rounding is enclosed.
- `use_double_precision`: kept for compatibility; it has no effect.

# Example
```julia
A = randn(100, 100)
r = verified_lu(A; use_bigfloat = false)
r.success      # A[r.p, r.q] = L U with L in r.L and U in r.U
```

# Reference
S. M. Rump and T. Ogita, *Verified Error Bounds for Matrix Decompositions*, SIAM J. Matrix Anal.
Appl. **45**(4) (2024), 2155-2183, doi 10.1137/24M165096X, Section 3.
"""
function verified_lu(A::AbstractMatrix{S};
        precision_bits::Int = 256,
        use_double_precision::Bool = true,
        use_bigfloat::Bool = true) where {S <: Union{
        Float64, ComplexF64, BigFloat, Complex{BigFloat}}}
    bigfloat = use_bigfloat || real(S) === BigFloat
    return _decomposition_arithmetic(bigfloat, precision_bits) do
        Ab = _decomposition_input(A, bigfloat)
        T = eltype(rad(Ab))
        m, n = size(Ab)
        k = min(m, n)
        r = _rumpogita2024_lu(Ab)
        r === nothing && return VerifiedLUResult(_unverified_ball(Ab, m, k),
            _unverified_ball(Ab, k, n), collect(1:m), false, T(Inf), collect(1:n))
        Apq = BallMatrix(mid(Ab)[r.p, r.q], rad(Ab)[r.p, r.q])
        residual_norm = _relative_residual_bound(r.L * r.U - Apq, Apq)
        return VerifiedLUResult(r.L, r.U, r.p, true, residual_norm, r.q)
    end
end

# Stub for Double64 extension
"""
    verified_lu_double64(A; precision_bits=256)

Fast verified LU using Double64 oracle. Requires DoubleFloats.jl.
"""
function verified_lu_double64 end

# Stub for MultiFloat extension
"""
    verified_lu_multifloat(A; precision_bits=256, float_type=Float64x4)

Fast verified LU using MultiFloat oracle. Requires MultiFloats.jl.
"""
function verified_lu_multifloat end

export VerifiedLUResult, verified_lu, verified_lu_double64, verified_lu_multifloat
