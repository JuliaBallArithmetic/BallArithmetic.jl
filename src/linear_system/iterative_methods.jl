# Iterative enclosures of the solution set of a square interval linear system, after
#
#   J. Horáček, Interval linear and nonlinear systems, doctoral thesis, Charles University,
#   Prague, 2019, Section 5.7: the initial enclosures of Proposition 5.14 and (5.4), the
#   stopping criterion of Section 5.7.2, the Jacobi method (Algorithm 5.15) and the
#   Gauss-Seidel method (Algorithm 5.17).
#
# The solution set of A x = b is Σ = {x : A x = b for some A ∈ A, b ∈ b}. The systems are real.
# Not implemented: extended interval arithmetic for a diagonal entry that contains zero, which
# the thesis mentions; such a component is left as it is.

function _require_real_system(A::BallMatrix, b::BallVector, name)
    n = size(A, 1)
    (n == size(A, 2) && length(b) == n) ||
        throw(DimensionMismatch("$name: A must be square and b of the same order"))
    (eltype(mid(A)) <: Real && eltype(mid(b)) <: Real) ||
        throw(ArgumentError("$name is for real interval systems"))
    return n
end

# C ≈ mid(A)⁻¹ in floating point, or nothing where the inverse is not finite
function _midpoint_inverse(A::BallMatrix)
    C = try
        inv(mid(A))
    catch
        return nothing
    end
    return all(isfinite, C) ? C : nothing
end

"""
    interval_initial_enclosure(A::BallMatrix, b::BallVector; C = inv(mid(A)), method = :auto)
        -> BallVector or nothing

A box that contains the solution set of the square interval system `A x = b`, by Section 5.7.1
of Horáček (2019), with `C` a real matrix (any; the default is the floating-point inverse of
the midpoint):

- `method = :proposition`, Proposition 5.14: if `⟨CA⟩u ≥ v > 0` for some `u ≥ 0`, then
  `Σ ⊆ ‖Cb‖_v [−u, u]`, with `‖x‖_v = max_i |x_i|/v_i`. `u` is an approximate solution of
  `⟨CA⟩u = e` and `v` a lower bound of `⟨CA⟩u`, as the thesis suggests.
- `method = :norm`, formula (5.4): if `‖I − CA‖_∞ < 1`, then
  `Σ ⊆ ‖Cb‖_∞/(1 − ‖I − CA‖_∞) [−e, e]`.
- `method = :auto`: the first, and the second where the first does not apply.

`CA` and `Cb` are formed in ball arithmetic and every bound is rounded outwards. Returns
`nothing` where the condition of the method is not established.
"""
function interval_initial_enclosure(A::BallMatrix{T}, b::BallVector{T};
        C = nothing, method::Symbol = :auto) where {T}
    n = _require_real_system(A, b, "interval_initial_enclosure")
    method in (:auto, :proposition, :norm) ||
        throw(ArgumentError("method must be :auto, :proposition or :norm"))
    C === nothing && (C = _midpoint_inverse(A))
    C === nothing && return nothing
    bC = BallMatrix(Matrix{T}(C))
    CA, Cb = bC * A, bC * b
    magCb = add_up.(abs.(mid(Cb)), rad(Cb))
    all(isfinite, magCb) || return nothing
    if method != :norm
        mig = diagonal_abs_lower_bound(CA)
        mag = upper_abs(CA)
        comparison = -mag
        for i in 1:n
            comparison[i, i] = mig[i]
        end
        u = try
            comparison \ ones(T, n)
        catch
            fill(T(NaN), n)
        end
        if all(x -> isfinite(x) && x >= 0, u)
            v = Vector{T}(undef, n)
            for i in 1:n
                s = zero(T)
                for j in 1:n
                    j == i || (s = add_up(s, mul_up(mag[i, j], u[j])))
                end
                v[i] = sub_down(mul_down(mig[i], u[i]), s)
            end
            if all(>(0), v)
                κ = maximum(div_up(magCb[i], v[i]) for i in 1:n)
                return BallVector(zeros(T, n), mul_up.(κ, u))
            end
        end
        method == :proposition && return nothing
    end
    q = upper_bound_L_inf_opnorm(CA - I)
    q < 1 || return nothing
    return BallVector(zeros(T, n), fill(div_up(maximum(magCb), sub_down(one(T), q)), n))
end

"""
    IterativeResult{T, VT}

Result of [`interval_jacobi`](@ref) and [`interval_gauss_seidel`](@ref).

# Fields
- `solution`: a box containing the part of the solution set that lies in the initial box
- `converged`: whether the stopping criterion of Section 5.7.2 was met within `max_iterations`
- `iterations`: number of sweeps
- `final_width`: the largest radius of `solution`
- `convergence_rate`: the ratio of the largest radii of the last two iterates
- `empty`: `true` when an intersection was empty, which proves that the solution set does not
  meet the initial box; `solution` is then the last nonempty iterate and is not an enclosure
"""
struct IterativeResult{T, VT}
    solution::VT
    converged::Bool
    iterations::Int
    final_width::T
    convergence_rate::T
    empty::Bool
end

# (b_i − Σ_{j≠i} a_ij x_j)/a_ii, the formula (5.6); x_i itself where a_ii contains zero
function _jacobi_component(A::BallMatrix{T}, b::BallVector{T}, x::Vector{<:Ball}, i::Int) where {T}
    sub_down(abs(mid(A)[i, i]), rad(A)[i, i]) > 0 || return x[i]
    s = b[i]
    for j in eachindex(x)
        j == i || (s = s - A[i, j] * x[j])
    end
    return s / A[i, i]
end

# The common iteration: y_i = (b_i − Σ_{j≠i} a_ij x_j)/a_ii, then x_i ∩ y_i; Gauss-Seidel uses the
# components already updated in the sweep.
function _interval_splitting_iteration(A::BallMatrix{T}, b::BallVector{T}, x0, C,
        max_iterations::Int, tol, gauss_seidel::Bool, name) where {T}
    n = _require_real_system(A, b, name)
    if C !== nothing
        bC = BallMatrix(Matrix{T}(C))
        A, b = bC * A, bC * b
    end
    if x0 === nothing
        x0 = interval_initial_enclosure(A, b)
        x0 === nothing && return IterativeResult(BallVector(zeros(T, n), fill(T(Inf), n)),
            false, 0, T(Inf), T(NaN), false)
    end
    length(x0) == n || throw(DimensionMismatch("$name: x0 must have length $n"))
    if tol === nothing
        # (5.5): ε ≈ min wid(A_ij) × 10⁻⁵, with a floor for point matrices
        w = minimum(rad(A))
        tol = w > 0 ? 2 * w * T(1e-5) : sqrt(eps(T)) * max(one(T), maximum(abs, mid(x0)))
    end
    x = [x0[i] for i in 1:n]
    divisible = [sub_down(abs(mid(A)[i, i]), rad(A)[i, i]) > 0 for i in 1:n]
    rate = T(NaN)
    width = maximum(rad, x)
    for iter in 1:max_iterations
        old = copy(x)
        for i in 1:n
            divisible[i] || continue
            y = intersect_ball(_jacobi_component(A, b, gauss_seidel ? x : old, i), old[i])
            if y === nothing
                return IterativeResult(BallVector(mid.(old), rad.(old)), false, iter, width,
                    rate, true)
            end
            x[i] = y
        end
        new_width = maximum(rad, x)
        rate = width > 0 ? new_width / width : T(NaN)
        width = new_width
        if all(abs(inf(old[i]) - inf(x[i])) < tol && abs(sup(old[i]) - sup(x[i])) < tol
               for i in 1:n)
            return IterativeResult(BallVector(mid.(x), rad.(x)), true, iter, width, rate, false)
        end
    end
    return IterativeResult(BallVector(mid.(x), rad.(x)), false, max_iterations, width, rate,
        false)
end

"""
    interval_jacobi(A::BallMatrix, b::BallVector; x0 = nothing, preconditioner = nothing,
        max_iterations = 20, tol = nothing) -> IterativeResult

Algorithm 5.15 of Horáček (2019), the Jacobi method for a real square interval system
`A x = b`: from a box `x⁽⁰⁾` containing the solution set,

    y_i⁽ᵏ⁺¹⁾ = (b_i − Σ_{j≠i} a_ij x_j⁽ᵏ⁾)/a_ii,        x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ ∩ y⁽ᵏ⁺¹⁾,

until the stopping criterion of Section 5.7.2: both ends of every component move by less than
`tol`, or `max_iterations` sweeps (the thesis uses 20). `tol = nothing` takes (5.5),
`min_ij wid(A_ij) × 10⁻⁵`.

- `x0`: the initial box. When it contains the solution set, so does the result; when it
  contains only a part of it, the result contains that part. `nothing` takes
  [`interval_initial_enclosure`](@ref), and where that gives none the result has infinite
  radii and `converged = false`.
- `preconditioner`: a real matrix `C`; the iteration is then applied to `(CA) x = Cb`, formed in
  ball arithmetic, whose solution set contains that of the original system (Section 5.4).

A diagonal entry that contains zero leaves its component unchanged. An empty intersection is
reported in the field `empty`.
"""
function interval_jacobi(A::BallMatrix{T}, b::BallVector{T}; x0 = nothing,
        preconditioner = nothing, max_iterations::Int = 20, tol = nothing) where {T}
    return _interval_splitting_iteration(A, b, x0, preconditioner, max_iterations, tol, false,
        "interval_jacobi")
end

"""
    interval_gauss_seidel(A::BallMatrix, b::BallVector; x0 = nothing, preconditioner = nothing,
        max_iterations = 20, tol = nothing) -> IterativeResult

Algorithm 5.17 of Horáček (2019), the Gauss-Seidel method: the Jacobi method
([`interval_jacobi`](@ref), with the same arguments and the same statement) in which each
component uses the components already updated in the sweep,

    y_i⁽ᵏ⁺¹⁾ = (b_i − Σ_{j<i} a_ij x_j⁽ᵏ⁺¹⁾ − Σ_{j>i} a_ij x_j⁽ᵏ⁾)/a_ii.
"""
function interval_gauss_seidel(A::BallMatrix{T}, b::BallVector{T}; x0 = nothing,
        preconditioner = nothing, max_iterations::Int = 20, tol = nothing) where {T}
    return _interval_splitting_iteration(A, b, x0, preconditioner, max_iterations, tol, true,
        "interval_gauss_seidel")
end

export IterativeResult
export interval_initial_enclosure, interval_gauss_seidel, interval_jacobi
