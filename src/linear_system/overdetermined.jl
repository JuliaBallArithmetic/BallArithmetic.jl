# Enclosures of the solution set of an overdetermined interval linear system A x = b, A of size
# m × n with m > n, after
#
#   J. Horáček, Interval linear and nonlinear systems, doctoral thesis, Charles University,
#   Prague, 2019, Chapter 6: the least squares approach (Section 6.2), the simple subsquares
#   algorithm (Algorithm 6.12), the selection of subsquares (Algorithm 6.14) and the multi-Jacobi
#   method (Section 6.8.3).
#
# Not implemented from that chapter: the preconditioners of Section 6.3, Gaussian elimination
# for an overdetermined system (6.4), Rohn's method (6.6).
#
# The solution set is Σ = {x : A x = b for some A ∈ A, b ∈ b}; it may be empty. The systems are
# real.

using Combinatorics

"""
    OverdeterminedResult{VT}

Result of the routines for overdetermined interval systems.

# Fields
- `solution`: a box containing the solution set (for [`interval_least_squares`](@ref), the set
  of least squares solutions) when `success` is `true`; infinite radii otherwise
- `success`: whether a finite enclosure was obtained
- `unsolvable`: `true` when the solution set is proved empty, by an empty intersection of
  enclosures; `solution` is then meaningless
- `method`: the method used
- `subsystems_checked`: number of square subsystems for which an enclosure was obtained
"""
struct OverdeterminedResult{VT}
    solution::VT
    success::Bool
    unsolvable::Bool
    method::Symbol
    subsystems_checked::Int
end

function _require_overdetermined(A::BallMatrix, b::BallVector, name)
    m, n = size(A)
    m > n || throw(DimensionMismatch("$name needs more equations than unknowns, got $m × $n"))
    length(b) == m || throw(DimensionMismatch("$name: b must have length $m"))
    (eltype(mid(A)) <: Real && eltype(mid(b)) <: Real) ||
        throw(ArgumentError("$name is for real interval systems"))
    return m, n
end

_no_enclosure(::Type{T}, n, method, checked) where {T} = OverdeterminedResult(
    BallVector(zeros(T, n), fill(T(Inf), n)), false, false, method, checked)

# an enclosure of the solution set of a square interval system by a method of Chapter 5, or
# nothing where the method gives none
function _square_enclosure(A::BallMatrix{T}, b::BallVector{T}, solver::Symbol) where {T}
    x = try
        if solver == :hbr
            r = hbr_method(A, b)
            r.success ? r.solution : nothing
        elseif solver == :verifylss
            r = verifylss(A, b)
            r.certified ? r.solution : nothing
        elseif solver == :gaussian_elimination
            r = interval_gaussian_elimination(A, b)
            r.success ? r.solution : nothing
        else
            throw(ArgumentError("unknown solver $solver"))
        end
    catch e
        e isa ArgumentError && rethrow()
        nothing
    end
    return (x !== nothing && all(isfinite, mid(x)) && all(isfinite, rad(x))) ? x : nothing
end

# x ∩ y componentwise for vectors of real balls, or nothing when a component is empty
function _intersect_boxes(x::Vector{<:Ball}, y::Vector{<:Ball})
    z = similar(x)
    for i in eachindex(x)
        w = intersect_ball(x[i], y[i])
        w === nothing && return nothing
        z[i] = w
    end
    return z
end

# `k` distinct elements of `S`, chosen at random: the procedure randsel of Algorithm 6.14
function _randsel(k::Int, S::Vector{Int}, rng)
    pool = copy(S)
    for i in 1:k
        j = rng === nothing ? rand(i:length(pool)) : rand(rng, i:length(pool))
        pool[i], pool[j] = pool[j], pool[i]
    end
    return sort!(pool[1:k])
end

"""
    select_subsquares(m, n; overlap = round(Int, n/3), rng = nothing) -> Vector{Vector{Int}}

Algorithm 6.14 of Horáček (2019): row index sets of square subsystems covering the `m`
equations of an `m × n` system. The first is chosen at random; each following one takes
`overlap` equations already covered and `n − overlap` not yet covered; the last takes the
remaining ones and completes with covered ones. Their number is (6.5),
`1 + ⌈(m − n)/(n − overlap)⌉`. The thesis reports `overlap ≈ n/3` as a reasonable choice.
"""
function select_subsquares(m::Int, n::Int; overlap::Int = round(Int, n / 3), rng = nothing)
    m > n || throw(DimensionMismatch("select_subsquares needs m > n"))
    0 <= overlap < n || throw(ArgumentError("overlap must satisfy 0 ≤ overlap < n"))
    subsquares = Vector{Int}[]
    covered, waiting = Int[], collect(1:m)
    while !isempty(waiting)
        indices = if isempty(covered)
            _randsel(n, waiting, rng)
        elseif length(waiting) <= n - overlap
            sort!(vcat(waiting, _randsel(n - length(waiting), covered, rng)))
        else
            sort!(vcat(_randsel(overlap, covered, rng), _randsel(n - overlap, waiting, rng)))
        end
        push!(subsquares, indices)
        covered = sort!(union(covered, indices))
        waiting = setdiff(waiting, indices)
    end
    return subsquares
end

function _subsquare_rows(m, n, selection, k, overlap, rng)
    if selection == :all
        return collect(combinations(1:m, n))
    elseif selection == :random
        return [_randsel(n, collect(1:m), rng) for _ in 1:k]
    elseif selection == :covering
        return select_subsquares(m, n; overlap, rng)
    end
    throw(ArgumentError("selection must be :all, :random or :covering"))
end

"""
    subsquares_method(A::BallMatrix, b::BallVector; selection = :covering, k = 10,
        overlap = round(Int, n/3), solver = :hbr, rng = nothing) -> OverdeterminedResult

Algorithm 6.12 of Horáček (2019), the simple subsquares algorithm, for a real overdetermined
interval system. The solution set lies in the solution set of every square subsystem
(Proposition 6.5), so:

1. select square subsystems: `selection = :random`, `k` random ones, as in the algorithm;
   `:covering`, those of [`select_subsquares`](@ref) (Algorithm 6.14); `:all`, all `binomial(m, n)`;
2. enclose the solution set of each by a method for square systems, `solver = :hbr`
   ([`hbr_method`](@ref)), `:verifylss` or `:gaussian_elimination`;
3. return the intersection of the enclosures;
4. if the intersection is empty, the system has no solution: `unsolvable = true`.

A subsquare for which the solver gives no enclosure is skipped. `success` is `false` when none
gives one.
"""
function subsquares_method(A::BallMatrix{T}, b::BallVector{T}; selection::Symbol = :covering,
        k::Int = 10, overlap = nothing, solver::Symbol = :hbr, rng = nothing) where {T}
    m, n = _require_overdetermined(A, b, "subsquares_method")
    overlap === nothing && (overlap = round(Int, n / 3))
    box = nothing
    checked = 0
    for rows in _subsquare_rows(m, n, selection, k, overlap, rng)
        x = _square_enclosure(A[rows, :], b[rows], solver)
        x === nothing && continue
        checked += 1
        y = [x[i] for i in 1:n]
        box = box === nothing ? y : _intersect_boxes(box, y)
        box === nothing && return OverdeterminedResult(
            BallVector(zeros(T, n), fill(T(Inf), n)), false, true, :subsquares, checked)
    end
    box === nothing && return _no_enclosure(T, n, :subsquares, 0)
    return OverdeterminedResult(BallVector(mid.(box), rad.(box)), true, false, :subsquares,
        checked)
end

"""
    multi_jacobi_method(A::BallMatrix, b::BallVector; x0 = nothing, overlap = round(Int, n/3),
        max_iterations = 20, tol = nothing, rng = nothing) -> OverdeterminedResult

Section 6.8.3 of Horáček (2019), the multi-Jacobi method: subsquares are selected by
Algorithm 6.14 ([`select_subsquares`](@ref)); then, instead of solving each, one iteration of
the Jacobi method (Algorithm 5.15) is applied to every subsquare from the current global box,
the global box is intersected with all the results, and this is repeated until the stopping
criterion of Section 5.7.2 or `max_iterations`.

- `x0`: a box containing the solution set. `nothing` takes the HBR enclosure of a subsquare, as
  in Example 6.15 of the thesis (the first selected subsquare for which it exists).
- Each subsquare is preconditioned by the floating-point inverse of its midpoint, the products
  formed in ball arithmetic. The thesis does not say how the diagonal of a subsquare is kept
  free of zero; this is the preconditioning of Section 5.4 applied to each subsquare.

An empty intersection proves that the system has no solution (`unsolvable = true`). `success`
is `false` when no initial box is found.
"""
function multi_jacobi_method(A::BallMatrix{T}, b::BallVector{T}; x0 = nothing,
        overlap = nothing, max_iterations::Int = 20, tol = nothing, rng = nothing) where {T}
    m, n = _require_overdetermined(A, b, "multi_jacobi_method")
    overlap === nothing && (overlap = round(Int, n / 3))
    systems = Tuple{BallMatrix{T, T}, BallVector{T, T}}[]
    for rows in select_subsquares(m, n; overlap, rng)
        As, bs = A[rows, :], b[rows]
        C = _midpoint_inverse(As)
        C === nothing && continue
        bC = BallMatrix(C)
        push!(systems, (bC * As, bC * bs))
        if x0 === nothing
            x0 = _square_enclosure(As, bs, :hbr)
        end
    end
    (x0 === nothing || isempty(systems)) && return _no_enclosure(T, n, :multi_jacobi, 0)
    length(x0) == n || throw(DimensionMismatch("multi_jacobi_method: x0 must have length $n"))
    if tol === nothing
        w = minimum(rad(A))
        tol = w > 0 ? 2 * w * T(1e-5) : sqrt(eps(T)) * max(one(T), maximum(abs, mid(x0)))
    end
    x = [x0[i] for i in 1:n]
    unsolvable(k) = OverdeterminedResult(BallVector(zeros(T, n), fill(T(Inf), n)), false, true,
        :multi_jacobi, k)
    for _ in 1:max_iterations
        old = x
        for (As, bs) in systems
            y = [_jacobi_component(As, bs, old, i) for i in 1:n]
            x = _intersect_boxes(x, y)
            x === nothing && return unsolvable(length(systems))
        end
        all(abs(inf(old[i]) - inf(x[i])) < tol && abs(sup(old[i]) - sup(x[i])) < tol
            for i in 1:n) && break
    end
    return OverdeterminedResult(BallVector(mid.(x), rad.(x)), true, false, :multi_jacobi,
        length(systems))
end

"""
    interval_least_squares(A::BallMatrix, b::BallVector; method = :supersquare,
        solver = :verifylss) -> OverdeterminedResult

Section 6.2 of Horáček (2019): an enclosure of the set of least squares solutions
(Definition 6.3)

    Σ_lsq = {x : AᵀA x = Aᵀb for some A ∈ A, b ∈ b},

whose hull contains that of the solution set of `A x = b`; the result is nonempty also when
the system has no solution.

- `method = :supersquare`: the square system of order `m + n`

      [I A; Aᵀ 0] [y; x] = [b; 0],

  enclosed by `solver`, a method for square systems (`:verifylss`, `:hbr` or
  `:gaussian_elimination`); the part `x` is returned. The two occurrences of each entry of `A`
  are treated as independent, which the thesis notes as a source of overestimation.
- `method = :normal_equations`: the interval normal equation `AᵀA x = Aᵀb`, with the products
  formed in ball arithmetic. The thesis warns that the overestimation of the product may be
  huge.

`success` is `false` when the solver gives no enclosure.
"""
function interval_least_squares(A::BallMatrix{T}, b::BallVector{T};
        method::Symbol = :supersquare, solver::Symbol = :verifylss) where {T}
    m, n = _require_overdetermined(A, b, "interval_least_squares")
    if method == :supersquare
        Mc, Mr = zeros(T, m + n, m + n), zeros(T, m + n, m + n)
        for i in 1:m
            Mc[i, i] = one(T)
        end
        Mc[1:m, (m + 1):(m + n)] .= mid(A)
        Mr[1:m, (m + 1):(m + n)] .= rad(A)
        Mc[(m + 1):(m + n), 1:m] .= transpose(mid(A))
        Mr[(m + 1):(m + n), 1:m] .= transpose(rad(A))
        rhs = BallVector(vcat(mid(b), zeros(T, n)), vcat(rad(b), zeros(T, n)))
        z = _square_enclosure(BallMatrix(Mc, Mr), rhs, solver)
        z === nothing && return _no_enclosure(T, n, :supersquare, 0)
        return OverdeterminedResult(BallVector(mid(z)[(m + 1):end], rad(z)[(m + 1):end]), true,
            false, :supersquare, 1)
    elseif method == :normal_equations
        At = BallMatrix(Matrix(transpose(mid(A))), Matrix(transpose(rad(A))))
        x = _square_enclosure(At * A, At * b, solver)
        x === nothing && return _no_enclosure(T, n, :normal_equations, 0)
        return OverdeterminedResult(x, true, false, :normal_equations, 1)
    end
    throw(ArgumentError("method must be :supersquare or :normal_equations"))
end

export OverdeterminedResult
export select_subsquares, subsquares_method, multi_jacobi_method, interval_least_squares
