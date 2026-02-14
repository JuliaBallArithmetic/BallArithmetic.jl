"""
    MiyajimaVBDResult

Container returned by [`miyajima_vbd`](@ref) encapsulating the data
produced by the verified block diagonalisation (VBD) step. The fields
contain the basis that block diagonalises the midpoint matrix, the
transformed enclosure, its block-diagonal truncation, the rigorous
remainder, and the Gershgorin clusters that certify how the spectrum
groups together.
"""
struct MiyajimaVBDResult{MT, BT, IT, RT, ET}
    """Basis that block diagonalises `mid(A)`."""
    basis::MT
    """Interval matrix expressed in the chosen basis."""
    transformed::BT
    """Block-diagonal truncation preserving the verified clusters."""
    block_diagonal::BT
    """Rigorous remainder satisfying `transformed = block_diagonal + remainder`."""
    remainder::BT
    """Index ranges identifying each spectral cluster."""
    clusters::Vector{UnitRange{Int}}
    """Gershgorin-type discs enclosing each diagonal entry."""
    cluster_intervals::Vector{IT}
    """Upper bound on `‖remainder‖₂` combining Collatz and block-separation bounds."""
    remainder_norm::RT
    """Eigenvalues associated with the diagonal of `mid(transformed)`."""
    eigenvalues::Vector{ET}
end

"""
    miyajima_vbd(A::BallMatrix; hermitian = false)

Perform Miyajima's verified block diagonalisation (VBD) on the square
ball matrix `A`.  The midpoint matrix is reduced either by an eigenvalue
decomposition (when `hermitian = true`) or by a unitary Schur form (for
the general case).  The enclosure is transported to that basis, the
Gershgorin discs are clustered, and a block-diagonal truncation together
with a rigorous remainder is produced.

Overlapping discs are grouped via their connectivity graph so that each
cluster becomes contiguous after a basis permutation.  The remainder
bound combines the classical Collatz estimate with a block-separation
bound that exploits the verified gaps between clusters.

When `hermitian = true` the routine expects `A` to be Hermitian and the
resulting eigenvalues and intervals are real.  Otherwise the Schur form
is used and the clusters are discs in the complex plane.
"""
function miyajima_vbd(A::BallMatrix{T, NT}; hermitian::Bool = false) where {T, NT}
    m, n = size(A)
    m == n || throw(ArgumentError("miyajima_vbd expects a square matrix"))

    basis, eigenvalues = hermitian ? _hermitian_diagonalisation(mid(A)) : _schur_diagonalisation(mid(A))
    identity_order = collect(1:n)

    current_basis = basis
    current_eigenvalues = eigenvalues

    intervals = nothing
    clusters = UnitRange{Int}[]
    transformed = nothing

    order = identity_order
    attempts = 0
    while true
        basis_ball = BallMatrix(current_basis)
        basis_adjoint = BallMatrix(adjoint(current_basis))
        transformed = basis_adjoint * A * basis_ball
        intervals = _vbd_gershgorin_intervals(transformed; hermitian)
        clusters, order = _interval_clusters(intervals)
        order == identity_order && break

        current_basis = current_basis[:, order]
        current_eigenvalues = current_eigenvalues[order]
        attempts += 1
        attempts > n && throw(ArgumentError("failed to permute Gershgorin clusters into contiguous blocks"))
    end

    basis = current_basis
    eigenvalues = current_eigenvalues

    transformed = Base.something(transformed)
    intervals = Base.something(intervals)

    block = _block_diagonal_part(transformed, clusters)
    remainder = transformed - block

    collatz_bound = collatz_upper_bound_L2_opnorm(remainder)
    block_bound = r2_infty_bound_by_blocks(transformed, intervals, clusters)
    remainder_norm = isfinite(block_bound) ? min(collatz_bound, block_bound) : collatz_bound

    return MiyajimaVBDResult(basis, transformed, block, remainder, clusters,
        intervals, remainder_norm, eigenvalues)
end

function _hermitian_diagonalisation(H::AbstractMatrix{T}) where {T}
    # For Diagonal matrices, eigendecomposition is trivial
    if H isa Diagonal
        n = size(H, 1)
        return (Matrix{T}(I, n, n), diag(H))
    end

    # For BigFloat matrices, try without alg keyword (Julia compat issue)
    # Julia 1.12+ passes alg=RobustRepresentations which BigFloat doesn't support
    if T <: BigFloat || (T <: Complex && real(T) <: BigFloat)
        # Use explicit call without keyword arguments
        try
            eig = eigen(Hermitian(H))
            return (eig.vectors, eig.values)
        catch e
            if e isa MethodError
                # Fallback: convert to Float64, compute, convert back
                H_f64 = convert.(Complex{Float64}, H)
                eig = eigen(Hermitian(H_f64))
                vectors = convert.(T, eig.vectors)
                values = convert.(real(T), eig.values)
                return (vectors, values)
            end
            rethrow(e)
        end
    end

    eig = eigen(Hermitian(H))
    return (eig.vectors, eig.values)
end

function _schur_diagonalisation(A::AbstractMatrix{T}) where {T}
    sch = schur(A)
    return (Matrix(sch.Z), sch.values)
end

function _vbd_gershgorin_intervals(H::BallMatrix{T, NT}; hermitian::Bool) where {T, NT}
    n = size(H, 1)
    midH = mid(H)
    radH = rad(H)

    intervals = hermitian ? Vector{Ball{T, T}}(undef, n) : Vector{Ball{T, Complex{T}}}(undef, n)

    absH = upper_abs(H)
    for i in 1:n
        diag_entry = Ball(midH[i, i], radH[i, i])
        diag_ball = hermitian ? _real_interval(diag_entry) : diag_entry

        row_sum = setrounding(T, RoundUp) do
            s = zero(T)
            for j in 1:n
                if j != i
                    s += absH[i, j]
                end
            end
            s
        end

        radius = setrounding(T, RoundUp) do
            rad(diag_ball) + row_sum
        end
        intervals[i] = Ball(mid(diag_ball), radius)
    end

    return intervals
end

_real_interval(x::Ball{T, T}) where {T} = x

function _real_interval(x::Ball{T, Complex{T}}) where {T}
    radius = setrounding(T, RoundUp) do
        rad(x) + abs(imag(mid(x)))
    end
    return Ball(real(mid(x)), radius)
end

function _interval_clusters(intervals::Vector{Ball{T, T}}) where {T}
    return _interval_clusters_generic(intervals)
end

function _interval_clusters(intervals::Vector{Ball{T, Complex{T}}}) where {T}
    return _interval_clusters_generic(intervals)
end

function _interval_clusters_generic(intervals)
    components = overlap_components(intervals)
    order = isempty(components) ? Int[] : vcat(components...)
    clusters = UnitRange{Int}[]
    start = 1
    for comp in components
        len = length(comp)
        push!(clusters, start:(start + len - 1))
        start += len
    end
    return clusters, (isempty(order) ? Int[] : order)
end

_balls_overlap(a::Ball{T, T}, b::Ball{T, T}) where {T} = intersect_ball(a, b) !== nothing

function _balls_overlap(a::Ball{T, Complex{T}}, b::Ball{T, Complex{T}}) where {T}
    # For conservative overlap detection (no false negatives) we need a
    # rigorous lower bound on the distance between centres.  RoundDown on
    # abs values followed by RoundDown hypot gives distance_down ≤ true
    # distance, so distance_down ≤ threshold_up guarantees detection of
    # every genuine overlap.
    distance = setrounding(T, RoundDown) do
        diff = mid(a) - mid(b)
        hypot(abs(real(diff)), abs(imag(diff)))
    end
    threshold = setrounding(T, RoundUp) do
        rad(a) + rad(b)
    end
    return distance <= threshold
end

function _block_diagonal_part(H::BallMatrix, clusters::Vector{UnitRange{Int}})
    midH = mid(H)
    radH = rad(H)
    block_mid = zeros(eltype(midH), size(midH))
    block_rad = zeros(eltype(radH), size(radH))

    for cluster in clusters
        block_mid[cluster, cluster] .= midH[cluster, cluster]
        block_rad[cluster, cluster] .= radH[cluster, cluster]
    end

    return BallMatrix(block_mid, block_rad)
end

function overlap_components(balls::AbstractVector{Ball{T, CT}}) where {T, CT}
    n = length(balls)
    components = Vector{Vector{Int}}()
    n == 0 && return components

    adjacency = [Int[] for _ in 1:n]
    for i in 1:n-1
        for j in i+1:n
            if _balls_overlap(balls[i], balls[j])
                push!(adjacency[i], j)
                push!(adjacency[j], i)
            end
        end
    end

    seen = falses(n)
    for s in 1:n
        seen[s] && continue
        stack = [s]
        seen[s] = true
        component = Int[s]
        while !isempty(stack)
            v = pop!(stack)
            for w in adjacency[v]
                if !seen[w]
                    seen[w] = true
                    push!(stack, w)
                    push!(component, w)
                end
            end
        end
        sort!(component)
        push!(components, component)
    end

    return components
end

function sep_clusters(ints, compA::UnitRange{Int}, compB::UnitRange{Int})
    hullA = reduce(ball_hull, ints[compA])
    hullB = reduce(ball_hull, ints[compB])
    T = radtype(typeof(hullA))

    cA, cB = mid(hullA), mid(hullB)
    rA, rB = rad(hullA), rad(hullB)

    # Rigorous lower bound on the distance between centres.
    distance_down = setrounding(T, RoundDown) do
        diff = cA - cB
        hypot(abs(real(diff)), abs(imag(diff)))
    end

    # Rigorous upper bound on the sum of the hull radii.
    radii_up = setrounding(T, RoundUp) do
        rA + rB
    end

    # Rigorous lower bound on the gap.
    gap = setrounding(T, RoundDown) do
        distance_down - radii_up
    end

    return gap <= zero(T) ? zero(T) : gap
end

function r2_infty_bound_by_blocks(H::BallMatrix{T}, intervals, clusters::Vector{UnitRange{Int}}) where {T}
    P = length(clusters)
    P == 0 && return zero(T)

    absH = upper_abs(H)
    row_sums = zeros(T, P)

    for p in 1:P
        for q in 1:P
            p == q && continue
            sep = sep_clusters(intervals, clusters[p], clusters[q])
            if sep == zero(T)
                return convert(T, Inf)
            end
            E∞ = block_infty_upper(absH, clusters[p], clusters[q])
            contribution = setrounding(T, RoundUp) do
                E∞ / sep
            end
            row_sums[p] = setrounding(T, RoundUp) do
                row_sums[p] + contribution
            end
        end
    end

    return maximum(row_sums)
end

function block_infty_upper(absH::AbstractMatrix{T}, rows, cols) where {T}
    smax = zero(T)
    for i in rows
        s = setrounding(T, RoundUp) do
            acc = zero(T)
            for j in cols
                acc += absH[i, j]
            end
            acc
        end
        smax = max(smax, s)
    end
    return smax
end

