# Sums of matrix products in higher precision by splitting the factors, after
#
#   M. Lange and S. M. Rump, "Accurate floating-point matrix residuals", manuscript (the
#   algorithm of INTLAB's `prodK`, in INTLAB from Version 14).
#
# Each factor is written as a sum of k + 1 matrices, A = A⁽¹⁾ + … + A⁽ᵏ⁾ + A̲, the rows of A and the
# columns of B each split at their own powers of two, chosen from the Euclidean norms so that every
# product A⁽ⁱ⁾B⁽ʲ⁾ is computed without rounding error by an ordinary matrix product, in any order
# of evaluation. The product is then evaluated as the paper's equation (27),
#
#   AB = Σ_{i,j ≤ k} A⁽ⁱ⁾B⁽ʲ⁾ + (A − A̲) B̲ + A̲ B,
#
# where only the last two products, the remainder terms, are rounded, and they are smaller than
# |A||B| by the factor reached by the k-th splitting point. All matrix products go to the BLAS.
#
# What is implemented, with the paper's numbering: Algorithm 1 and 2 (Split), Algorithm 3 (SplitK),
# Algorithm 4 (the splitting points from the norms), Lemma 4 and Corollary 5 (exactness of the
# leading products, with λ = 1), Section 4 (complex input), Definition 12 and Lemma 13 (several
# splitting points), the points (25) with q_a = q_b (balanced), the evaluation (27), and the
# separate treatment of the summands of a residual (Section 8.2).
#
# What is not: the choice λ < 1 with its second bound (15); unbalanced splittings and their
# points (23), (26); the Karatsuba and Toom-Cook evaluations of Section 7 (Algorithm 6); the
# blocked summation of Section 8.1 (Lemma 15); unevaluated sums of more than one matrix as one
# factor (Algorithm 7) and output in more than two terms with the normalisation (35); sparse
# factors (Section 8.4); the use of SYRK.
#
# The error matrix is ours: the paper gives the a priori bound (14) and does not list how `prodK`
# computes its second output. Here the leading products are exact by Lemma 4, the rounding of the
# two remainder products is bounded by the standard bound for a floating-point inner product with
# |B̲| ≤ ν⁽ᵏ⁾/2 and |A̲| ≤ μ⁽ᵏ⁾/2 entrywise (Definition 12), which makes the bound a sum of two
# rank-one matrices, and the final summation is by `two_sum`, whose errors are kept.

const _LangeRumpFloat = Union{Float32, Float64}

# the smallest power of two not below x, for x > 0 finite
function _next_power_of_two(x::T) where {T <: AbstractFloat}
    f, e = frexp(x)
    return f == T(1) / 2 ? x : ldexp(one(T), e)
end

# The high part of Algorithm 1 (Split): with t = (2^p − 2^(p−2)) μ, a′ = fl(fl(t + a) − t).
@inline _lr_high(a::T, t::T) where {T <: AbstractFloat} = (t + a) - t
@inline _lr_high(a::Complex{T}, t::T) where {T <: AbstractFloat} = complex(
    (t + real(a)) - t, (t + imag(a)) - t)

# Algorithm 2, with one splitting point for each row (`byrow`) or for each column: `hi + lo == A`
# exactly, `hi[i, j]` an integer multiple of the point of its row or column and `|lo[i, j]|` at most
# half of it (for complex entries, the real and the imaginary part each). No `@simd`: it would allow
# `(t + a) − t` to be reassociated.
function _lr_split!(hi::Matrix{S}, lo::Matrix{S}, A::Matrix{S}, t::Vector{T},
        byrow::Bool) where {T, S <: Union{T, Complex{T}}}
    m, n = size(A)
    @inbounds for j in 1:n
        tj = byrow ? zero(T) : t[j]
        for i in 1:m
            a = A[i, j]
            h = _lr_high(a, byrow ? t[i] : tj)
            hi[i, j] = h
            lo[i, j] = a - h
        end
    end
    return hi, lo
end

# Upper bounds of the 1-norms of the rows (`byrow`) or of the columns of A, with |z| ≤ |Re z| + |Im z|.
function _lr_one_norms(A::Matrix{S}, byrow::Bool) where {S}
    T = real(S)
    s = vec(sum(z -> abs(real(z)) + abs(imag(z)), A; dims = byrow ? 2 : 1))
    N = (S <: Complex ? 2 : 1) * size(A, byrow ? 2 : 1)
    c = add_up(one(T), gamma_bound(N + 1, T))
    return mul_up.(c, s)
end

# Algorithm 4 (ComputeMu) and Algorithm 3 (SplitK) for the rows (`byrow`) or the columns of A.
# `N` is the length of the inner products as real ones (twice the inner dimension for complex data,
# Theorem 10) and `q` the constant of Corollary 5. Returns the k leading parts, the remainder, the
# first splitting points and the constant ratio τ between consecutive ones, or `nothing` where a
# splitting constant overflows.
#
# Line 7 of Algorithm 4 bounds the squared norm from the floating-point sum of squares; here that
# step is: for s̃ the floating-point sum of M squares in any order, the exact sum is at most
# (s̃ + M η)(1 + γ_{M+2}), η the underflow unit. The later points are those of (25),
# μ⁽ⁱ⁾ = τ μ⁽ⁱ⁻¹⁾ with τ the smallest power of two not below (√N/2)/q.
function _lr_split_k(A::Matrix{S}, byrow::Bool, k::Integer, q::T,
        N::Integer) where {T <: _LangeRumpFloat, S <: Union{T, Complex{T}}}
    s = vec(sum(abs2, A; dims = byrow ? 2 : 1))
    M = (S <: Complex ? 2 : 1) * size(A, byrow ? 2 : 1)
    half_sqrtN = div_up(sqrt_up(T(N)), T(2))
    den = sub_down(q, half_sqrtN)
    den > one(T) || return nothing
    c = add_up(one(T), gamma_bound(M + 2, T))
    under = mul_up(T(M), subnormal_min(T))
    μ = similar(s)
    @inbounds for i in eachindex(s)
        if iszero(s[i])
            μ[i] = one(T)           # a zero vector splits into zeros at any point
        else
            su = mul_up(c, add_up(s[i], under))
            μ[i] = _next_power_of_two(div_up(sqrt_up(su), den))
        end
    end
    τ = _next_power_of_two(div_up(half_sqrtN, q))
    # Lemma 13, condition (22): μ⁽¹⁾ ≥ 2^(2−p)|a| holds since q ≤ 2^(p−2); the second needs
    τ >= T(2)^(1 - precision(T)) || return nothing
    tfac = T(2)^precision(T) - T(2)^(precision(T) - 2)
    parts = Matrix{S}[]
    rest = A
    pt = copy(μ)
    for _ in 1:k
        t = tfac .* pt
        all(isfinite, t) || return nothing
        hi, lo = similar(A), similar(A)
        _lr_split!(hi, lo, rest, t, byrow)
        push!(parts, hi)
        rest = lo
        pt = τ .* pt
    end
    return parts, rest, μ, τ
end

# (s, e) ← two_sum(s, x) entrywise, with e and Σ|e| accumulated. Real arrays; complex matrices
# are passed reinterpreted.
function _lr_cascade!(s::AbstractArray{T}, e::AbstractArray{T}, eabs::AbstractArray{T},
        x::AbstractArray{T}) where {T <: AbstractFloat}
    @inbounds for i in eachindex(s, e, eabs, x)
        t, es = two_sum(s[i], x[i])
        s[i] = t
        e[i] += es
        eabs[i] += abs(es)
    end
    return nothing
end

_lr_real_view(X::Matrix{T}) where {T <: AbstractFloat} = X
_lr_real_view(X::Matrix{Complex{T}}) where {T <: AbstractFloat} = reinterpret(T, X)

# A A_t B_t with B_t (or A_t) diagonal: every entry is one product, so `two_product` gives it
# exactly as two matrices (four for two complex factors).
function _lr_diagonal_pieces!(pieces, keys, A::Matrix{S}, d::Vector{S}, bycolumn::Bool,
        sgn) where {S}
    m, n = size(A)
    scale(i, j) = bycolumn ? d[j] : d[i]
    if S <: Complex
        X1, X2, E1, E2 = (Matrix{S}(undef, m, n) for _ in 1:4)
        @inbounds for j in 1:n, i in 1:m
            a, b = A[i, j], sgn * scale(i, j)
            p1, e1 = two_product(real(a), real(b))
            p2, e2 = two_product(-imag(a), imag(b))
            p3, e3 = two_product(real(a), imag(b))
            p4, e4 = two_product(imag(a), real(b))
            X1[i, j], X2[i, j] = complex(p1, p3), complex(p2, p4)
            E1[i, j], E2[i, j] = complex(e1, e3), complex(e2, e4)
        end
        append!(pieces, (X1, X2, E1, E2))
        append!(keys, (2, 2, typemax(Int), typemax(Int)))
    else
        X, E = Matrix{S}(undef, m, n), Matrix{S}(undef, m, n)
        @inbounds for j in 1:n, i in 1:m
            X[i, j], E[i, j] = two_product(A[i, j], sgn * scale(i, j))
        end
        append!(pieces, (X, E))
        append!(keys, (2, typemax(Int)))
    end
    return nothing
end

"""
    _lange_rump_two_terms(terms; k = 2) -> (P1, P2, R) or nothing

`Σ_t sgn_t A_t B_t` for `terms` an iterable of `(A_t, B_t, sgn_t)`, point matrices of `Float32` or
`Float64` entries, real or complex, and `sgn_t = ±1`, as an unevaluated sum of two matrices with
`|Σ_t sgn_t A_t B_t − P1 − P2| ≤ R` entrywise (the modulus, for complex data).

The method is the splitting scheme of M. Lange and S. M. Rump, "Accurate floating-point matrix
residuals" (manuscript; INTLAB's `prodK`): each factor is split into `k` leading parts and a
remainder by their Algorithms 1 to 4, the rows of `A_t` and the columns of `B_t` at the points
(25) with `q_a = fl(√2^p)`, `q_b = fl((1 − u)2^p/q_a)` (Algorithm 5), and the product is their
equation (27): `k²` matrix products that are exact by their Lemma 4 and Corollary 5, and two
rounded remainder products. It corresponds to `prodK(…, k, 'OutputTerms', 2)`, "as if" in
`(k/2 + 1)`-fold precision. A summand with a diagonal factor is formed entry by entry with
`two_product`. The summands are split separately (Section 8.2 of the paper).

`R` is the sum of

* for each summand, with `n` the inner dimension, `γ(|A − A̲| |B̲| + |A̲| |B|) + |δ||B̲|` bounded
  through `|B̲| ≤ ν⁽ᵏ⁾/2` and `|A̲| ≤ μ⁽ᵏ⁾/2` entrywise (`1/√2` in place of `1/2` for complex
  data), `γ = γ_n` (`γ_{n+2}` for complex), `δ` the rounding of `fl(A − A̲)`, and an underflow term;
* the rounding of the accumulation of the `two_sum` errors of the final summation.

Returns `nothing`, and the caller falls back on the accumulation by `compensated_terms2`, where a
sign is not `±1`, where a splitting constant overflows or a product of splitting points underflows,
where `2^p` does not exceed the squared inner dimension by enough for Corollary 5, or where the
result is not finite.

The exactness of the leading products requires a matrix product that only adds and multiplies
(as the reference BLAS and OpenBLAS do); a Strassen-type product would not qualify (footnote 3 of
the paper).

Not implemented from the paper: `λ < 1`, unbalanced splittings, the Karatsuba evaluation, the
blocked summation of Section 8.1, multiple-precision input (Algorithm 7), more than two output
terms, sparse factors.
"""
function _lange_rump_two_terms(terms; k::Integer = 2)
    k >= 1 || throw(ArgumentError("_lange_rump_two_terms: k = $k must be at least 1"))
    S = promote_type((promote_type(eltype(A), eltype(B)) for (A, B, _) in terms)...)
    T = real(S)
    T <: _LangeRumpFloat || return nothing
    all(t -> abs(t[3]) == 1, terms) || return nothing
    p = precision(T)
    u = unit_roundoff(T)
    η = subnormal_min(T)
    qa = sqrt(T(2)^p)
    qb = ((one(T) - u) * T(2)^p) / qa
    m, nout = size(first(terms)[1], 1), size(first(terms)[2], 2)
    κ = S <: Complex ? T(0.7072) : T(1) / 2       # |entry of a remainder| ≤ κ × its splitting point
    infl = one(T) + 8 * eps(T)

    pieces = Matrix{S}[]
    keys = Int[]
    R = zeros(T, m, nout)
    for (At, Bt, sgn) in terms
        A, B = Matrix{S}(At), Matrix{S}(Bt)
        (size(A, 1) == m && size(B, 2) == nout && size(A, 2) == size(B, 1)) ||
            throw(DimensionMismatch("_lange_rump_two_terms: summands of sizes $(size(A)) × $(size(B))"))
        n = size(A, 2)
        if size(B, 1) == size(B, 2) && isdiag(B)
            _lr_diagonal_pieces!(pieces, keys, A, diag(B), true, S(sgn))
            R .+= (S <: Complex ? 8 : 2) * η
            continue
        elseif size(A, 1) == size(A, 2) && isdiag(A)
            _lr_diagonal_pieces!(pieces, keys, B, diag(A), false, S(sgn))
            R .+= (S <: Complex ? 8 : 2) * η
            continue
        end
        N = (S <: Complex ? 2 : 1) * n
        sa = _lr_split_k(A, true, k, qa, N)
        sb = _lr_split_k(B, false, k, qb, N)
        (sa === nothing || sb === nothing) && return nothing
        Ap, Ar, μ, τa = sa
        Bp, Br, ν, τb = sb
        # Lemma 4 needs μ⁽ⁱ⁾ν⁽ʲ⁾ ∈ F for all the points: the smallest product must be normal
        μk, νk = (τa^(k - 1)) .* μ, (τb^(k - 1)) .* ν
        minimum(μk) * minimum(νk) >= floatmin(T) || return nothing
        for i in 1:k, j in 1:k
            Z = Ap[i] * Bp[j]
            sgn == 1 || (Z .= .-Z)
            push!(pieces, Z)
            push!(keys, i + j)
        end
        # the remainder terms of (27); Ah = fl(A − A̲) with its rounding δ
        Ah, δ = similar(A), similar(A)
        Ahr, δr, Ar_, Arr = _lr_real_view(Ah), _lr_real_view(δ), _lr_real_view(A), _lr_real_view(Ar)
        @inbounds for i in eachindex(Ahr)
            Ahr[i], δr[i] = two_sum(Ar_[i], -Arr[i])
        end
        Ta, Tb = Ah * Br, Ar * B
        if sgn != 1
            Ta .= .-Ta
            Tb .= .-Tb
        end
        append!(pieces, (Ta, Tb))
        append!(keys, (2k + 1, 2k + 1))
        g = gamma_bound(n, S)
        u1 = add_up.(mul_up.(g, _lr_one_norms(Ah, true)), _lr_one_norms(δ, true))
        v1 = mul_up.(κ, νk)
        u2 = mul_up.(mul_up(g, κ), μk)
        v2 = _lr_one_norms(B, false)
        under = (S <: Complex ? 8 : 2) * n * η
        @inbounds for j in 1:nout, i in 1:m
            R[i, j] += (u1[i] * v1[j] + u2[i] * v2[j]) * infl + under
        end
    end

    # the final summation, the largest pieces first
    order = sortperm(keys)
    P1 = copy(pieces[order[1]])
    P2 = zeros(S, m, nout)
    eabs = zeros(T, size(_lr_real_view(P1)))
    for idx in order[2:end]
        _lr_cascade!(_lr_real_view(P1), _lr_real_view(P2), eabs, _lr_real_view(pieces[idx]))
    end
    # |P2 − Σ(two_sum errors)| ≤ γ_K Σ|errors| in each real component, K the number of pieces
    gK = mul_up(gamma_bound(2 * length(pieces) + 4, T), infl)
    if S <: Complex
        @inbounds for j in 1:nout, i in 1:m
            R[i, j] += gK * (eabs[2i - 1, j] + eabs[2i, j]) * infl
        end
    else
        @inbounds for i in eachindex(R)
            R[i] += gK * eabs[i] * infl
        end
    end
    # the sums into R were rounded to nearest: a few additions for each summand
    final = one(T) + (8 + 4 * length(pieces)) * eps(T)
    @inbounds for i in eachindex(R)
        R[i] = R[i] * final + η
    end
    (all(isfinite, P1) && all(isfinite, P2) && all(isfinite, R)) || return nothing
    return P1, P2, R
end
