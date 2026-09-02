# Implementation of RumpLange2023: verified bounds for all eigenvalues of a
# Hermitian matrix, with emphasis on clusters.
#
# Reference: Rump, S.M. & Lange, M., "Fast computation of error bounds for all
# eigenpairs of a Hermitian and all singular pairs of a rectangular matrix with
# emphasis on eigen- and singular value clusters", J. Comput. Appl. Math. 434
# (2023) 115332.
#
# This file follows Section 4 of that paper, i.e. Algorithm verifyeigall.

using LinearAlgebra

"""
    RumpLange2023Result

Container for the eigenvalue enclosures of [`rump_lange_2023_cluster_bounds`](@ref).

# Fields
- `eigenvectors::VT`: the approximate eigenvectors `X̃` used to build the bounds.
- `eigenvalues::ΛT`: the intervals `Lⱼ = Λ̃ⱼⱼ ± δⱼ`.
- `cluster_assignments::Vector{Int}`: the cluster index of each `Lⱼ`.
- `cluster_bounds::Vector{Ball{T, T}}`: the hull `⋃_{i ∈ µⱼ} Lᵢ` of each cluster.
- `num_clusters::Int`: the number of clusters.
- `cluster_residuals::Vector{T}`: the radius `δ` shared by the members of each cluster.
- `cluster_separations::Vector{T}`: the gap from each cluster hull to the nearest other hull.
- `cluster_sizes::Vector{Int}`: the number of eigenvalues in each cluster.
- `verified::Bool`: whether the algorithm reached a partition with mutually
  disjoint cluster hulls and finite radii.

When `verified` is true the following holds for every Hermitian matrix inside the
input ball matrix: there is a numbering of its eigenvalues with `λⱼ ∈ eigenvalues[j]`
for all `j`, the cluster hulls are mutually disjoint, and the hull of cluster `k`
contains exactly `cluster_sizes[k]` eigenvalues counted with multiplicity.
"""
struct RumpLange2023Result{T, VT, ΛT}
    eigenvectors::VT
    eigenvalues::ΛT
    cluster_assignments::Vector{Int}
    cluster_bounds::Vector{Ball{T, T}}
    num_clusters::Int
    cluster_residuals::Vector{T}
    cluster_separations::Vector{T}
    cluster_sizes::Vector{Int}
    verified::Bool
end

Base.length(result::RumpLange2023Result) = length(result.eigenvalues)
Base.getindex(result::RumpLange2023Result, i::Int) = result.eigenvalues[i]

"""
    rump_lange_2023_cluster_bounds(A::BallMatrix; hermitian=false, cluster_tol=1e-6, fast=true)

Compute verified enclosures for all eigenvalues of a Hermitian matrix, together
with the cluster structure of those enclosures; this is Algorithm `verifyeigall`
of Rump and Lange (2023).

Let `AX̃ ≈ X̃Λ̃` be an approximate eigendecomposition of the midpoint of `A` and let
`E := AX̃ - X̃Λ̃` be computed in ball arithmetic. Kahan's theorem, in the form given
by Theorem 4.1 of the paper, says that for any index set `µ` there exist `|µ|`
eigenvalues of `A` within

    δ(µ) := ‖E(:, µ)‖ / σ_min(X̃(:, µ))

of the `Λ̃ᵢᵢ` with `i ∈ µ`. Taking `µ = {j}` gives `δⱼ = ‖Eeⱼ‖ / ‖X̃eⱼ‖` and the
interval `Lⱼ := Λ̃ⱼⱼ ± δⱼ`, which contains an eigenvalue. The union of the `Lⱼ`
need not contain the whole spectrum, and Section 4 of the paper gives two
matrices where an eigenvalue lies in no `Lⱼ`; the remedy is to collect clusters
recursively. The algorithm therefore loops: it takes the connected components of
the intersection graph of the `Lⱼ`, recomputes `δ` blockwise for every component
of size at least two, which widens the intervals and may merge further
components, and repeats until the partition stops changing. At that fixed point
the cluster hulls are mutually disjoint, so each hull contains exactly as many
eigenvalues as it has members and the spectrum is contained in their union.

# Arguments
- `A::BallMatrix`: the matrix. Some Hermitian matrix must lie inside it, i.e.
  `|A[i,j] - conj(A[j,i])|` must not exceed the sum of the two radii; otherwise
  an `ArgumentError` is thrown. The bounds are then valid for every Hermitian
  matrix inside `A`.

# Keyword arguments
- `hermitian::Bool = false`: accepted for backwards compatibility and ignored;
  the method applies to Hermitian matrices only.
- `cluster_tol::Real = 1e-6`: eigenvalue approximations closer than this are put
  in the same cluster even when their intervals are disjoint. This is the `kappa`
  of Section 6 of the paper; it only merges clusters, so it cannot invalidate the
  enclosures, and setting it to zero clusters by interval overlap alone.
- `fast::Bool = true`: bound the block norms by `√(‖·‖₁‖·‖_∞)`. With `fast = false`
  the Perron root bound of Section 3 is used as well and the smaller of the two is
  taken, which gives radii that are never larger and often smaller.

# Returns
A [`RumpLange2023Result`](@ref).

# Notes
The method is for Hermitian matrices; for a general matrix use
[`rump_2022a_eigenvalue_bounds`](@ref), which implements the algorithm the paper
cites as its reference [13].

# References

* Rump S.M. and Lange M., J. Comput. Appl. Math. 434 (2023) 115332
"""
function rump_lange_2023_cluster_bounds(A::BallMatrix{T, NT};
                                        hermitian::Bool = false,
                                        cluster_tol::Real = 1e-6,
                                        fast::Bool = true) where {T, NT}
    size(A, 1) == size(A, 2) || throw(ArgumentError("A must be square"))
    _check_hermitian_ball(A)

    n = size(A, 1)

    # Approximate eigendecomposition of the midpoint. Only the upper triangle is
    # read, so the approximation is that of a Hermitian matrix; nothing about the
    # enclosures depends on it being a good one.
    F = eigen(Hermitian(mid(A)))
    λ = F.values
    X = F.vectors
    bX = BallMatrix(X)

    # E := A X̃ - X̃ Λ̃, in ball arithmetic.
    E = A * bX - bX * BallMatrix(Matrix(Diagonal(λ)))

    # δⱼ = ‖E eⱼ‖ / ‖X̃ eⱼ‖ for the singletons; each Lⱼ contains an eigenvalue.
    δ = Vector{T}(undef, n)
    for j in 1:n
        num = upper_bound_norm(E[:, j], 2)
        den = _column_norm_lower(X, j)
        δ[j] = den > zero(T) ? (@up num / den) : T(Inf)
    end

    clusters = _cluster_loop!(δ, λ, E, bX, T(cluster_tol), fast)

    return _assemble_result(bX, λ, δ, clusters)
end

"""
    _check_hermitian_ball(A::BallMatrix)

Throw an `ArgumentError` unless some Hermitian matrix lies inside `A`, i.e.
unless `|A[i,j] - conj(A[j,i])| ≤ rad(A[i,j]) + rad(A[j,i])` for all `i, j`.
"""
function _check_hermitian_ball(A::BallMatrix{T, NT}) where {T, NT}
    Ac = mid(A)
    Ar = rad(A)
    n = size(A, 1)
    for i in 1:n, j in i:n
        # The gap is rounded up and the tolerance down, so a matrix is rejected
        # only when it certainly holds no Hermitian matrix.
        gap = setrounding(T, RoundUp) do
            abs(Ac[i, j] - conj(Ac[j, i]))
        end
        tol = setrounding(T, RoundDown) do
            Ar[i, j] + Ar[j, i]
        end
        if gap > tol
            throw(ArgumentError(
                "rump_lange_2023_cluster_bounds applies to Hermitian matrices, and no " *
                "Hermitian matrix lies inside A: entries ($i,$j) and ($j,$i) differ by " *
                "$gap, more than the sum $tol of their radii. Use " *
                "rump_2022a_eigenvalue_bounds for a general matrix."))
        end
    end
    return nothing
end

"""
    _column_norm_lower(X::AbstractMatrix, j::Int)

Lower bound for the Euclidean norm of the `j`-th column of the floating point
matrix `X`.
"""
function _column_norm_lower(X::AbstractMatrix, j::Int)
    T = real(eltype(X))
    s = setrounding(T, RoundDown) do
        acc = zero(T)
        for i in axes(X, 1)
            acc += abs2(X[i, j])
        end
        acc
    end
    return sqrt_down(s)
end

"""
    _block_norm_upper(M::BallMatrix, fast::Bool)

Upper bound for the spectral norm of `M`. With `fast` the simple bound
`√(‖M‖₁‖M‖_∞)` is used; otherwise the Perron root bound of Section 3 of the paper
is computed as well and the smaller of the two is returned.
"""
function _block_norm_upper(M::BallMatrix{T, NT}, fast::Bool) where {T, NT}
    n1 = upper_bound_L1_opnorm(M)
    ninf = upper_bound_L_inf_opnorm(M)
    simple = sqrt_up(@up n1 * ninf)
    return fast ? simple : min(simple, collatz_upper_bound_L2_opnorm(M))
end

"""
    _sigma_min_lower(Y::BallMatrix)

Lower bound for the smallest singular value of `Y`, from `‖I - Y*Y‖ ≤ α < 1`
implying `σ_min(Y) ≥ √(1 - α)`; this is `singmin` of Section 3 of the paper. The
bound is zero when `α ≥ 1`.
"""
function _sigma_min_lower(Y::BallMatrix{T, NT}) where {T, NT}
    p = size(Y, 2)
    G = BallMatrix(Matrix{NT}(I, p, p)) - Y' * Y
    α = min(one(T), _block_norm_upper(G, false))
    α >= one(T) && return zero(T)
    return sqrt_down(@down one(T) - α)
end

"""
    _cluster_loop!(δ, λ, E, bX, tol, fast)

The while-loop of Algorithm `verifyeigall`. Starting from the columnwise radii
`δ`, take the connected components of the intersection graph of the intervals
`λⱼ ± δⱼ`, recompute `δ` blockwise on every component of size at least two, and
repeat until the number of components stops decreasing. Returns the final
partition as a vector of index vectors; `δ` is modified in place.
"""
function _cluster_loop!(δ::Vector{T}, λ, E, bX, tol::T, fast::Bool) where {T}
    n = length(δ)
    clusters = [[j] for j in 1:n]
    previous = n

    while true
        clusters = _components(λ, δ, tol)
        big = [v for v in clusters if length(v) > 1]

        (isempty(big) || length(clusters) == previous) && break
        previous = length(clusters)

        for v in big
            s = _sigma_min_lower(bX[:, v])
            e = _block_norm_upper(E[:, v], fast)
            d = s > zero(T) ? (@up e / s) : T(Inf)
            for j in v
                δ[j] = d
            end
        end
    end

    return clusters
end

"""
    _components(λ, δ, tol)

Connected components of the graph on `1:n` whose edges join `i` and `j` when the
intervals `λᵢ ± δᵢ` and `λⱼ ± δⱼ` intersect, or when `|λᵢ - λⱼ| ≤ tol`. Both
conditions are overlaps of the intervals `λⱼ ± max(δⱼ, tol/2)`, so the components
are found by one sweep over those intervals sorted by their lower end, and the
components produced have mutually disjoint hulls, which is what the theorem
behind the algorithm needs. The `tol` edges only merge, and merging is always
safe: a hull containing several components still contains as many eigenvalues as
it has members. Returns a vector of index vectors ordered by the hull.
"""
function _components(λ, δ::Vector{T}, tol::T) where {T}
    n = length(δ)
    half = tol / 2

    lo = setrounding(T, RoundDown) do
        [λ[j] - max(δ[j], half) for j in 1:n]
    end
    hi = setrounding(T, RoundUp) do
        [λ[j] + max(δ[j], half) for j in 1:n]
    end

    order = sortperm(lo)
    out = Vector{Vector{Int}}()
    current = [order[1]]
    reach = hi[order[1]]

    for idx in @view order[2:end]
        if lo[idx] <= reach
            push!(current, idx)
            reach = max(reach, hi[idx])
        else
            push!(out, sort!(current))
            current = [idx]
            reach = hi[idx]
        end
    end
    push!(out, sort!(current))

    return out
end

"""
    _assemble_result(bX, λ, δ, clusters)

Package the intervals `λⱼ ± δⱼ` and the partition into a [`RumpLange2023Result`](@ref).
"""
function _assemble_result(bX, λ, δ::Vector{T}, clusters) where {T}
    n = length(δ)
    k = length(clusters)

    eigenvalues = [Ball(λ[j], δ[j]) for j in 1:n]

    assignments = zeros(Int, n)
    sizes = zeros(Int, k)
    bounds = Vector{Ball{T, T}}(undef, k)
    residuals = zeros(T, k)

    for (c, v) in enumerate(clusters)
        sizes[c] = length(v)
        residuals[c] = maximum(δ[j] for j in v)
        for j in v
            assignments[j] = c
        end
        lower = setrounding(T, RoundDown) do
            minimum(λ[j] - δ[j] for j in v)
        end
        upper = setrounding(T, RoundUp) do
            maximum(λ[j] + δ[j] for j in v)
        end
        centre = setrounding(T, RoundNearest) do
            (lower + upper) / 2
        end
        radius = setrounding(T, RoundUp) do
            max(upper - centre, centre - lower)
        end
        bounds[c] = Ball(centre, radius)
    end

    separations = fill(T(Inf), k)
    disjoint = true
    for a in 1:k, b in 1:k
        a == b && continue
        gap = max(inf(bounds[a]) - sup(bounds[b]), inf(bounds[b]) - sup(bounds[a]))
        gap <= zero(T) && (disjoint = false)
        separations[a] = min(separations[a], max(zero(T), gap))
    end
    k == 1 && (separations[1] = T(Inf))

    verified = disjoint && all(isfinite, δ)

    return RumpLange2023Result(bX, eigenvalues, assignments, bounds, k,
                               residuals, separations, sizes, verified)
end

"""
    refine_cluster_bounds(result::RumpLange2023Result, A::BallMatrix; iterations=1)

Recompute the bounds of `result` with the sharper norm estimates, that is with
`fast = false`, so that every block norm is bounded by the smaller of the Perron
root bound and `√(‖·‖₁‖·‖_∞)`. The radii returned are never larger than those of
a run with `fast = true`, and the partition is never coarser.

The refinement of the eigenvalue approximations by Rayleigh quotients, Section 5
of the paper, is not implemented; `iterations` is accepted for backwards
compatibility and ignored, since the recomputation is not iterative.
"""
function refine_cluster_bounds(result::RumpLange2023Result,
                               A::BallMatrix{T, NT};
                               iterations::Int = 1) where {T, NT}
    iterations >= 1 || throw(ArgumentError("iterations must be ≥ 1"))
    return rump_lange_2023_cluster_bounds(A; fast = false)
end

# Export
export RumpLange2023Result, rump_lange_2023_cluster_bounds, refine_cluster_bounds
