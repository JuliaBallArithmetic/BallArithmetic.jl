# Error bounds for an approximate solution of a dense linear system, through an H-matrix.
#
# A. Minamihata, T. Ogita, S. M. Rump and S. Oishi, "Modified error bounds for approximate
# solutions of dense linear systems", Journal of Computational and Applied Mathematics 369
# (2020), 112546, doi 10.1016/j.cam.2019.112546. Theorem and equation numbers below are those of
# this paper.

"""
    VerifiedLinearSystemResult

The result of [`verified_linear_solve_hmatrix`](@ref).

  - `solution`: the ball vector with midpoint `x_approx` and radius `error_bound`; when
    `verified` it contains the solution of every system with data in the given balls;
  - `x_approx`: the approximate solution `x̃`;
  - `error_bound`: `ε` with `|Â⁻¹ b̂ − x̃| ≤ ε` entrywise, infinite when not `verified`;
  - `verified`: whether the hypothesis of the theorems was proved;
  - `method`: the bound that was computed;
  - `R`: the approximate inverse used;
  - `v`: the positive vector with `⟨C⟩ v > 0`, empty when none was found.
"""
struct VerifiedLinearSystemResult{T}
    solution::BallVector{T}
    x_approx::Vector{T}
    error_bound::Vector{T}
    verified::Bool
    method::Symbol
    R::Matrix{T}
    v::Vector{T}
end

"""
    mig(x::Ball)

A lower bound of the smallest absolute value of the points of `x`, zero when `x` contains zero.
"""
mig(x::Ball) = max(sub_down(abs_down(mid(x)), rad(x)), zero(rad(x)))

"""
    mag(x::Ball)

An upper bound of the largest absolute value of the points of `x`.
"""
mag(x::Ball) = add_up(abs_up(mid(x)), rad(x))

"""
    comparison_matrix(C::BallMatrix)

The comparison matrix `⟨C⟩` of Definition 2.7 of Minamihata, Ogita, Rump and Oishi (2020), in
floating point: [`mig`](@ref) of the entry on the diagonal, minus [`mag`](@ref) of the entry off
it. Both are rounded so that the matrix returned is entrywise below the comparison matrix of
`C`; a positive vector that it maps to a positive vector therefore proves that `C` is an
H-matrix (Remark 1 there).
"""
function comparison_matrix(C::BallMatrix)
    n = size(C, 1)
    n == size(C, 2) || throw(DimensionMismatch("the comparison matrix needs a square matrix"))
    return [i == j ? mig(C[i, i]) : -mag(C[i, j]) for i in 1:n, j in 1:n]
end

"""
    verified_linear_solve_hmatrix(A::BallMatrix, b::BallVector;
        method = :theorem_3_3, m = 1, R = nothing, x_approx = nothing)

An entrywise bound `ε` of the error of an approximate solution `x̃` of a real linear system:
`|Â⁻¹ b̂ − x̃| ≤ ε` for every matrix `Â` in `A` and every vector `b̂` in `b`, together with the
proof that every such `Â` is nonsingular. Returns a [`VerifiedLinearSystemResult`](@ref);
`verified` is `false`, and the bound infinite, when the hypothesis could not be proved.

The hypothesis, common to all four bounds: with `R` an approximate inverse and `C` a ball matrix
containing `R Â` for every `Â` in `A`, there is a vector `v > 0` with `u := ⟨C⟩ v > 0`, where
`⟨C⟩` is the comparison matrix (see [`comparison_matrix`](@ref)). Then `C` is an H-matrix, `Â`
and `R` are nonsingular, and `|(R Â)⁻¹| ≤ ⟨C⟩⁻¹` (Theorems 2.1 and 2.3, Remark 1). Write
`⟨C⟩ = D − E` with `D` the diagonal part, `G := E D⁻¹`, and `c` for an upper bound of
`|R (b̂ − Â x̃)|`.

`method` selects the bound:

  - `:theorem_3_1` (Rump): `ε = (D⁻¹ + v wᵀ) c` with `w_k := max_i G_ik / u_i`, equation (3);
  - `:theorem_3_2` (Minamihata, Sekine, Ogita, Rump and Oishi 2015):
    `ε = (D⁻¹ + v wᵀ) (I − Σ_{k=1}^m (Δ^{2k−1} − Δ^{2k})) c` with `Δ := u wᵀ − G`,
    equation (4);
  - `:theorem_3_3` (the default): `ε = min_{1 ≤ k ≤ m} ε^(k)`,
    `ε^(k) := D⁻¹ (c^(0) + … + c^(k−1)) + max_i (c^(k)_i / u_i) v`, with `c^(0) := c` and
    `c^(k) := G c^(k−1)`, equations (5) and (6);
  - `:corollary_3_2`: `ε = D⁻¹ r + α v` with
    `r := ⟨mid C⟩ |ỹ| + |mid(C) ỹ − R (b̂ − Â x̃)|`, `α := max_i (G r)_i / u_i` and `ỹ` an
    approximate solution of `mid(C) y = R (b − A x̃)`, equation (17).

`m` is the number of terms in Theorems 3.2 and 3.3. `R` defaults to the floating-point inverse
of `mid(A)` and `x_approx` to `R mid(b)`.

How it is computed. `C` and the residual `R (b − A x̃)` are products and sums of balls, so they
contain what they should for every point of `A` and `b`; the residual is formed in working
precision. `v` is the floating-point solution of `⟨C⟩ v = (1, …, 1)ᵀ`, the first of the two
choices named in Section 3 of the paper, and `u` is then enclosed, its lower end being the one
that has to be positive. The bounds of Theorems 3.1, 3.3 and of Corollary 3.2 are sums and
products of nonnegative numbers, rounded upwards, with `u` replaced by its lower end where it
divides. The bound of Theorem 3.2 has terms of both signs and is evaluated in balls, `w` being
any vector with `G ≤ u wᵀ`, which is all its proof uses; the upper end is returned.

What is not reproduced. The paper distinguishes two enclosures of `R A`, Method (a) from two
products with directed rounding and Method (b) from one product and the a priori bound
`2 n u |R| |A|`, and proposes Corollary 3.2 for the second. Here there is one enclosure, the
ball product of the package, and each of the four bounds is applied to it (Remark 5 of the paper
allows Corollary 3.2 on any enclosure); the comparison of costs between the two methods does not
carry over.

# Reference

A. Minamihata, T. Ogita, S. M. Rump and S. Oishi, "Modified error bounds for approximate
solutions of dense linear systems", Journal of Computational and Applied Mathematics 369
(2020), 112546, doi 10.1016/j.cam.2019.112546.
"""
function verified_linear_solve_hmatrix(A::BallMatrix{T}, b::BallVector{T};
        method::Symbol = :theorem_3_3, m::Int = 1,
        R::Union{Nothing, AbstractMatrix} = nothing,
        x_approx::Union{Nothing, AbstractVector} = nothing) where {T <: AbstractFloat}
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("A must be square"))
    n == length(b) || throw(DimensionMismatch("A and b have different sizes"))
    eltype(mid(A)) <: Real && eltype(mid(b)) <: Real ||
        throw(ArgumentError("verified_linear_solve_hmatrix is for real systems"))
    method in (:theorem_3_1, :theorem_3_2, :theorem_3_3, :corollary_3_2) ||
        throw(ArgumentError("unknown method $method: use :theorem_3_1, :theorem_3_2, " *
                            ":theorem_3_3 or :corollary_3_2"))
    m >= 1 || throw(ArgumentError("m must be at least 1"))

    Rm = if R === nothing
        try
            inv(mid(A))
        catch
            return _hmatrix_failure(mid(b), method, zeros(T, n, n))
        end
    else
        Matrix{T}(R)
    end
    size(Rm) == (n, n) || throw(DimensionMismatch("R must have the size of A"))
    x̃ = x_approx === nothing ? Rm * mid(b) : Vector{T}(x_approx)
    length(x̃) == n || throw(DimensionMismatch("x_approx must have the length of b"))
    (all(isfinite, Rm) && all(isfinite, x̃)) || return _hmatrix_failure(x̃, method, Rm)

    RB = BallMatrix(Rm)
    C = RB * A
    ρ = RB * (b - A * BallVector(x̃))
    (all(isfinite, mid(C)) && all(isfinite, rad(C)) && all(isfinite, mid(ρ)) &&
     all(isfinite, rad(ρ))) || return _hmatrix_failure(x̃, method, Rm)

    # ⟨C⟩ = D − E, the positive vector v and the two ends of u = ⟨C⟩ v
    M = comparison_matrix(C)
    d = diag(M)
    all(>(0), d) || return _hmatrix_failure(x̃, method, Rm)
    v = try
        M \ ones(T, n)
    catch
        return _hmatrix_failure(x̃, method, Rm)
    end
    (all(isfinite, v) && all(>(0), v)) || return _hmatrix_failure(x̃, method, Rm)
    u = BallMatrix(M) * BallVector(v)
    ulo = [inf(u[i]) for i in 1:n]
    all(>(0), ulo) || return _hmatrix_failure(x̃, method, Rm, v)

    c = [mag(ρ[i]) for i in 1:n]
    ε = if method === :theorem_3_1
        _hmatrix_theorem_3_1(M, d, v, ulo, c)
    elseif method === :theorem_3_2
        _hmatrix_theorem_3_2(M, d, v, u, ulo, c, m)
    elseif method === :theorem_3_3
        _hmatrix_theorem_3_3(M, d, v, ulo, c, m)
    else
        _hmatrix_corollary_3_2(C, ρ, M, d, v, ulo)
    end
    all(isfinite, ε) || return _hmatrix_failure(x̃, method, Rm, v)
    return VerifiedLinearSystemResult(BallVector(x̃, ε), x̃, ε, true, method, Rm, v)
end

function _hmatrix_failure(x̃::AbstractVector{T}, method::Symbol, Rm::Matrix{T},
        v::Vector{T} = T[]) where {T}
    ε = fill(T(Inf), length(x̃))
    return VerifiedLinearSystemResult(BallVector(Vector{T}(x̃), ε), Vector{T}(x̃), ε, false,
        method, Rm, v)
end

# an upper bound of G y for y ≥ 0, with G = E D⁻¹, E the off-diagonal part of −M and d its diagonal
function _hmatrix_G_times(M::Matrix{T}, d::Vector{T}, y::Vector{T}) where {T}
    n = length(y)
    q = [div_up(y[k], d[k]) for k in 1:n]
    g = zeros(T, n)
    @inbounds for k in 1:n, i in 1:n
        i == k && continue
        g[i] = add_up(g[i], mul_up(-M[i, k], q[k]))
    end
    return g
end

# w_k ≥ max_i G_ik / u_i, so that G ≤ u wᵀ
function _hmatrix_w(M::Matrix{T}, d::Vector{T}, ulo::Vector{T}) where {T}
    n = length(d)
    w = zeros(T, n)
    @inbounds for k in 1:n, i in 1:n
        i == k && continue
        w[k] = max(w[k], div_up(div_up(-M[i, k], d[k]), ulo[i]))
    end
    return w
end

# Theorem 3.1, equation (3): (D⁻¹ + v wᵀ) c
function _hmatrix_theorem_3_1(M::Matrix{T}, d::Vector{T}, v::Vector{T}, ulo::Vector{T},
        c::Vector{T}) where {T}
    n = length(c)
    w = _hmatrix_w(M, d, ulo)
    s = zero(T)
    for k in 1:n
        s = add_up(s, mul_up(w[k], c[k]))
    end
    return [add_up(div_up(c[i], d[i]), mul_up(v[i], s)) for i in 1:n]
end

# Theorem 3.3, equations (5) and (6)
function _hmatrix_theorem_3_3(M::Matrix{T}, d::Vector{T}, v::Vector{T}, ulo::Vector{T},
        c::Vector{T}, m::Int) where {T}
    n = length(c)
    best = fill(T(Inf), n)
    partial = copy(c)          # c^(0) + … + c^(k−1)
    ck = c
    for k in 1:m
        ck = _hmatrix_G_times(M, d, ck)
        α = maximum(div_up(ck[i], ulo[i]) for i in 1:n)
        for i in 1:n
            best[i] = min(best[i], add_up(div_up(partial[i], d[i]), mul_up(α, v[i])))
        end
        k == m && break
        for i in 1:n
            partial[i] = add_up(partial[i], ck[i])
        end
    end
    return best
end

# Theorem 3.2, equation (4), in balls: the terms have both signs. `u` is the enclosure of ⟨C⟩ v
# and `w` any vector with G ≤ u wᵀ; the matrix applied to c is entrywise above ⟨C⟩⁻¹ ≥ 0, so an
# upper bound of c may stand for it.
function _hmatrix_theorem_3_2(M::Matrix{T}, d::Vector{T}, v::Vector{T}, u::BallVector{T},
        ulo::Vector{T}, c::Vector{T}, m::Int) where {T}
    n = length(c)
    w = _hmatrix_w(M, d, ulo)
    wdot(y) = sum(Ball(w[k]) * y[k] for k in 1:n)
    # Δ y = u (wᵀ y) − G y
    function Δ(y)
        s = wdot(y)
        q = [y[k] / Ball(d[k]) for k in 1:n]
        z = [u[i] * s for i in 1:n]
        for k in 1:n, i in 1:n
            i == k && continue
            z[i] = z[i] - Ball(-M[i, k]) * q[k]
        end
        return z
    end
    p = [Ball(c[i]) for i in 1:n]
    t = copy(p)
    for j in 1:(2m)
        p = Δ(p)
        t = isodd(j) ? [t[i] - p[i] for i in 1:n] : [t[i] + p[i] for i in 1:n]
    end
    s = wdot(t)
    return [sup(t[i] / Ball(d[i]) + Ball(v[i]) * s) for i in 1:n]
end

# Corollary 3.2, equation (17): D⁻¹ r + α v
function _hmatrix_corollary_3_2(C::BallMatrix{T}, ρ::BallVector{T}, M::Matrix{T},
        d::Vector{T}, v::Vector{T}, ulo::Vector{T}) where {T}
    n = length(d)
    Cm = mid(C)
    ρm = mid(ρ)
    ỹ = try
        y = Cm \ ρm
        y + Cm \ (ρm - Cm * y)
    catch
        return fill(T(Inf), n)
    end
    all(isfinite, ỹ) || return fill(T(Inf), n)
    # r = ⟨mid C⟩ |ỹ| + |mid(C) ỹ − R (b − A x̃)|, an upper bound of it, not below zero
    defect = BallMatrix(Cm) * BallVector(ỹ) - ρ
    r = Vector{T}(undef, n)
    for i in 1:n
        s = Ball(abs(Cm[i, i])) * Ball(abs(ỹ[i]))
        for k in 1:n
            k == i && continue
            s = s - Ball(abs(Cm[i, k])) * Ball(abs(ỹ[k]))
        end
        r[i] = max(add_up(sup(s), mag(defect[i])), zero(T))
    end
    g = _hmatrix_G_times(M, d, r)
    α = maximum(div_up(g[i], ulo[i]) for i in 1:n)
    return [add_up(div_up(r[i], d[i]), mul_up(α, v[i])) for i in 1:n]
end

export VerifiedLinearSystemResult
export verified_linear_solve_hmatrix
export comparison_matrix, mig, mag
