# The shaving method for a square interval linear system, after
#
#   J. Horáček, Interval linear and nonlinear systems, doctoral thesis, Charles University,
#   Prague, 2019, Section 5.9 (an adapted version of M. Hladík and J. Horáček, reference [81] of
#   the thesis): Lemma 5.21, the sufficient condition of Section 5.9.1, the width of a slice by
#   Lemma 5.22 with the bounds (5.15) and (5.16), and the iterative improvement of Section 5.9.3.
#
# Not implemented: the update of the preconditioner by the Sherman-Morrison formula in the
# iterative improvement, which the thesis proposes for speed; the system is solved again.

using LinearAlgebra

"""
    sherman_morrison_inverse_update(A_inv::Matrix{T}, u::Vector{T}, v::Vector{T}) where {T}

Compute (A + uv^T)^(-1) using Sherman-Morrison formula.

# Sherman-Morrison Formula
    (A + uv^T)^(-1) = A^(-1) - (A^(-1)uv^T A^(-1)) / (1 + v^T A^(-1)u)

This allows O(n²) update of inverse for rank-1 perturbation instead of O(n³) recomputation.

# Arguments
- `A_inv`: Inverse of base matrix A (n×n)
- `u`: First vector of rank-1 update (n)
- `v`: Second vector of rank-1 update (n)

# Returns
(A + uv^T)^(-1)

# Example
```julia
A = [3.0 1.0; 1.0 2.0]
A_inv = inv(A)
u = [1.0, 0.0]
v = [0.0, 1.0]

# Efficiently compute inv(A + u*v')
A_updated_inv = sherman_morrison_inverse_update(A_inv, u, v)
```

# Notes
- O(n²) complexity vs O(n³) for full inverse
- Numerically stable if |1 + v^T A^(-1)u| is not too small
- A floating-point formula: the result is an approximation, with no error bound
"""
function sherman_morrison_inverse_update(A_inv::Matrix{T}, u::Vector{T}, v::Vector{T}) where {T}
    # Compute A^(-1) * u
    A_inv_u = A_inv * u

    # Compute v^T * A^(-1)
    vT_A_inv = v' * A_inv

    # Compute denominator: 1 + v^T * A^(-1) * u
    denom = 1 + dot(v, A_inv_u)

    if abs(denom) < eps(T) * 100
        error("Sherman-Morrison: Denominator too small, matrix may be singular")
    end

    # Compute rank-1 update: A^(-1) - (A^(-1)u)(v^T A^(-1)) / denom
    update = (A_inv_u * vT_A_inv) / denom

    return A_inv - update
end


"""
    ShavingResult{T, VT}

Result of [`interval_shaving`](@ref).

# Fields
- `solution`: the shaved box; it contains every point of the solution set that the initial box
  contains
- `shaved_amount`: the sum of the widths removed
- `iterations`: number of passes over the variables
- `components_shaved`: number of slices removed
- `empty`: `true` when a slice as wide as the box was removed, which proves that the initial box
  contains no point of the solution set; `solution` is then meaningless
"""
struct ShavingResult{T, VT}
    solution::VT
    shaved_amount::T
    iterations::Int
    components_shaved::Int
    empty::Bool
end

"""
    _shaving_lp(c, Aeq, beq, lb, ub) -> Vector or nothing

A minimiser of `c'v` subject to `Aeq v = beq`, `lb ≤ v ≤ ub`, in floating point, or `nothing`.
The method is supplied by the extension loaded with JuMP and HiGHS. The result is used as a
guess only and need not be verified.
"""
function _shaving_lp end

"""
    interval_shaving(A::BallMatrix, b::BallVector, x0::BallVector; passes = 1,
        improvements = 3, min_improvement = 1e-6, lp_solver = nothing) -> ShavingResult

The shaving method of Section 5.9 of Horáček (2019) for a real square interval system
`A x = b`: slices of the box `x0` that contain no point of the solution set are cut off. With
`x0` an enclosure of the solution set, so is the result.

For the variable `i` and the upper end, the slice of width `α` is the box with `x_i` restricted
to `[x̄_i − α, x̄_i]` (5.8), and similarly at the lower end. For each of the `2n` slices:

1. **A certificate for the midpoint system** (Section 5.9.1). By Lemma 5.21, `A x = b`, `x ∈ x`
   has no solution exactly when

       Aᵀw + y − z = 0,    bᵀw + x̄ᵀy − x̲ᵀz = −1,    y, z ≥ 0                        (5.9)

   is solvable. The linear programme `min b_cᵀw + x̄ᵀy − x̲ᵀz` subject to `A_cᵀw + y − z = 0`,
   `−e ≤ w ≤ e`, `y, z ≥ 0` is solved for the slice of width zero. If its value is not
   negative the slice is kept.
2. **A square interval system.** The entries of `y` and `z` that vanish at the optimum are
   fixed to zero; the remaining system (5.12) is completed to a square one, `C v = d`,
   `C ∈ C`, with an orthonormal basis of the null space of its midpoint, the right-hand sides
   chosen so that the normalised optimum solves it (5.13). Its solution set is enclosed by
   [`verifylss`](@ref). If the components `u` of the enclosure that come from `y` and `z` are
   nonnegative, (5.9) is solvable for every `A ∈ A`, `b ∈ b`, and the slice contains no point
   of the solution set.
3. **The width** (Section 5.9.2). The width `α` enters one coefficient, `C + α e_p e_qᵀ`, and by
   Lemma 5.22 the solution becomes `v − α v_q/(1 + α C⁻¹_qp) · C⁻¹_*p`. With `c` an enclosure of
   `C⁻¹_*p`, the solution set of `C u = e_p`, the largest `α` allowed is given by
   `α < −1/c̲_q` when `c̲_q < 0` (5.15) and, for every component `k` of `u` other than `q`,
   `α ≤ u̲_k / (sup(u_q c_k) − u̲_k c̲_q)` when the denominator is positive (5.16), all rounded
   downwards.
4. **Iterative improvement** (Section 5.9.3), `improvements` times: the coefficient is moved by
   the width found, the system enclosed again, `c` updated by `c/(1 + α c_q)`, with its `q`-th
   component computed as `1/(α + 1/c_q)`, and a further width added.

`passes` is the number of sweeps over the variables; a slice narrower than `min_improvement`
times the width of its variable is not cut.

`lp_solver(c, Aeq, beq, lb, ub)` must return a minimiser of `c'v` subject to `Aeq v = beq`,
`lb ≤ v ≤ ub`, or `nothing`. It is a heuristic: whatever it returns, a slice is cut only after
step 2 has verified the certificate. `nothing` uses JuMP with HiGHS, which must then be loaded
(`using JuMP, HiGHS`).
"""
function interval_shaving(A::BallMatrix{T}, b::BallVector{T}, x0::BallVector{T};
        passes::Int = 1, improvements::Int = 3, min_improvement::Real = 1e-6,
        lp_solver = nothing) where {T}
    n = _require_real_system(A, b, "interval_shaving")
    length(x0) == n || throw(DimensionMismatch("interval_shaving: x0 must have length $n"))
    lp = lp_solver === nothing ? _shaving_lp : lp_solver
    applicable(lp, zeros(T, 1), zeros(T, 1, 1), zeros(T, 1), zeros(T, 1), zeros(T, 1)) ||
        throw(ArgumentError("interval_shaving needs a linear programming solver: load JuMP " *
                            "and HiGHS, or pass `lp_solver`"))
    lo, hi = [inf(x0[i]) for i in 1:n], [sup(x0[i]) for i in 1:n]
    shaved, cuts, done = zero(T), 0, 0
    box() = BallVector([_ball_from_bounds(lo[i], hi[i]) for i in 1:n])
    for pass in 1:passes
        done = pass
        progressed = false
        for i in 1:n, upper in (true, false)
            width = sub_down(hi[i], lo[i])
            width > 0 || continue
            α = _slice_width(A, b, lo, hi, i, upper, lp, improvements)
            α > min_improvement * width || continue
            if α >= sub_up(hi[i], lo[i])
                return ShavingResult(box(), shaved, pass, cuts + 1, true)
            end
            upper ? (hi[i] = sub_up(hi[i], α)) : (lo[i] = add_down(lo[i], α))
            shaved += α
            cuts += 1
            progressed = true
        end
        progressed || break
    end
    return ShavingResult(box(), shaved, done, cuts, false)
end

# A width α ≥ 0 such that the slice of that width of the box [lo, hi], at the upper or the lower
# end of variable i, contains no point of the solution set; `Inf` when every width does.
function _slice_width(A::BallMatrix{T}, b::BallVector{T}, lo::Vector{T}, hi::Vector{T}, i::Int,
        upper::Bool, lp, improvements::Int) where {T}
    n = length(lo)
    slo, shi = copy(lo), copy(hi)
    upper ? (slo[i] = hi[i]) : (shi[i] = lo[i])           # the slice of width zero
    Ac, bc = mid(A), mid(b)
    # step 1: the linear programme in (w, y, z)
    Id = Matrix{T}(I, n, n)
    cost = vcat(bc, shi, -slo)
    sol = lp(cost, hcat(Matrix(transpose(Ac)), Id, -Id), zeros(T, n),
        vcat(fill(-one(T), n), zeros(T, 2n)), vcat(ones(T, n), fill(T(Inf), 2n)))
    (sol === nothing || length(sol) != 3n || !all(isfinite, sol)) && return zero(T)
    opt = dot(cost, sol)
    opt < -sqrt(eps(T)) * max(one(T), maximum(abs, cost)) * maximum(abs, sol) || return zero(T)
    thr = sqrt(eps(T)) * maximum(abs, sol)
    Y = [j for j in 1:n if sol[n + j] > thr]
    Z = [j for j in 1:n if sol[2n + j] > thr]
    N = n + length(Y) + length(Z)
    N >= n + 1 || return zero(T)
    vstar = vcat(sol[1:n], sol[n .+ Y], sol[2n .+ Z]) ./ (-opt)
    # step 2: the square interval system C v = d
    Cc, Cr = zeros(T, N, N), zeros(T, N, N)
    Cc[1:n, 1:n] .= transpose(Ac)
    Cr[1:n, 1:n] .= transpose(rad(A))
    for (t, j) in enumerate(Y)
        Cc[j, n + t] = one(T)
        Cc[n + 1, n + t] = shi[j]
    end
    for (t, j) in enumerate(Z)
        Cc[j, n + length(Y) + t] = -one(T)
        Cc[n + 1, n + length(Y) + t] = -slo[j]
    end
    Cc[n + 1, 1:n] .= bc
    Cr[n + 1, 1:n] .= rad(b)
    d = zeros(T, N)
    d[n + 1] = -one(T)
    if N > n + 1
        K = nullspace(Cc[1:(n + 1), :])
        size(K, 2) == N - n - 1 || return zero(T)
        Cc[(n + 2):N, :] .= transpose(K)
        d[(n + 2):N] .= transpose(K) * vstar
    end
    all(isfinite, Cc) || return zero(T)
    # the coefficient that carries the width: −x̲_i in the column of z_i for an upper slice,
    # x̄_i in the column of y_i for a lower one, both in row n + 1
    p = n + 1
    t = upper ? findfirst(==(i), Z) : findfirst(==(i), Y)
    q = t === nothing ? nothing : (upper ? n + length(Y) + t : n + t)
    C = BallMatrix(Cc, Cr)
    rhs = BallVector(d)
    v = _shaving_solve(C, rhs, n, N)
    v === nothing && return zero(T)
    q === nothing && return T(Inf)            # the system does not depend on the width
    # step 3
    ep = zeros(T, N)
    ep[p] = one(T)
    sc = try
        verifylss(C, BallVector(ep))
    catch
        return zero(T)
    end
    sc.certified || return zero(T)
    c = [sc.solution[k] for k in 1:N]
    total = zero(T)
    for it in 0:improvements
        α = _slice_step(v, c, q, n, N)
        (α > 0) || break
        isfinite(α) || return T(Inf)
        total = add_down(total, α)
        it == improvements && break
        # step 4: the system with the coefficient moved by α
        moved = C[p, q] + Ball(α)
        Cc[p, q], Cr[p, q] = mid(moved), rad(moved)
        C = BallMatrix(copy(Cc), copy(Cr))
        v = _shaving_solve(C, rhs, n, N)
        v === nothing && break
        den = Ball(one(T)) + Ball(α) * c[q]
        inf(den) > 0 || break
        cq = (inf(c[q]) > 0 || sup(c[q]) < 0) ? inv(Ball(α) + inv(c[q])) : c[q] / den
        c = [k == q ? cq : c[k] / den for k in 1:N]
    end
    return total
end

# the enclosure of the solution set of C v = d, when its components n+1:N are nonnegative
function _shaving_solve(C::BallMatrix{T}, rhs::BallVector{T}, n::Int, N::Int) where {T}
    s = try
        verifylss(C, rhs)
    catch
        return nothing
    end
    s.certified || return nothing
    v = [s.solution[k] for k in 1:N]
    all(isfinite(rad(v[k])) for k in 1:N) && all(inf(v[k]) >= 0 for k in (n + 1):N) ||
        return nothing
    return v
end

# the largest width allowed by (5.15) and (5.16), rounded downwards; Inf when neither restricts
function _slice_step(v::Vector{<:Ball}, c::Vector{<:Ball}, q::Int, n::Int, N::Int)
    T = radtype(typeof(v[1]))
    α = T(Inf)
    cq = inf(c[q])
    cq < 0 && (α = prevfloat(div_down(one(T), -cq)))
    for k in (n + 1):N
        k == q && continue
        vk = inf(v[k])
        D = sub_up(sup(v[q] * c[k]), mul_down(vk, cq))
        D > 0 && (α = min(α, div_down(vk, D)))
    end
    return α
end

export ShavingResult
export interval_shaving, sherman_morrison_inverse_update
