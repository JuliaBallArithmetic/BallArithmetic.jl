# NJD-based Verified Block Diagonalisation
#
# Implements the Numerical Jordan Decomposition (NJD) route for verified
# block diagonalisation from:
#
#   Miyajima, S. (2021). "Verified computation of the matrix exponential",
#   J. Comput. Appl. Math. 396, 113614. (Section 4)
#
# The Jordan structure is computed via the RDEFL algorithm from:
#
#   Kågström, B. & Ruhe, A. (1980). "An algorithm for numerical
#   computation of the Jordan normal form of a complex matrix",
#   ACM Trans. Math. Softw. 6(3), 398–419. (Algorithm 560)
#
# Unlike the NSD-based `miyajima_vbd` (which relies on the eigenvector
# matrix X being nonsingular), the NJD route works when A is defective
# or near-defective.  The nilpotent part M_j = diag(N_{k1}, …, N_{ks})
# is block-diagonal with shift matrices, so the nilpotent index
# nmax = max(kᵢ) can be much smaller than the cluster size, yielding
# tighter Theorem 2 perturbation bounds.

using LinearAlgebra

# ── Types ────────────────────────────────────────────────────────────

"""
    JordanBlockInfo{T}

Information about the Jordan structure within a single eigenvalue cluster.
Produced by the NJD algorithm (SVD staircase + principal vector computation)
from [KagstromRuhe1980](@cite).
"""
struct JordanBlockInfo{T}
    """Sizes of the Jordan blocks in this cluster, e.g. [2, 2, 1]."""
    block_sizes::Vector{Int}
    """Nilpotent index pⱼ = max(kᵢ) for this cluster."""
    nilpotent_index::Int
    """Representative eigenvalue λ̂ⱼ (mean of cluster eigenvalues)."""
    representative_eigenvalue::Complex{T}
    """Small remainder Qⱼ in the Jordan basis (diag residual)."""
    diagonal_residual::Matrix{Complex{T}}
    """Local Jordan basis Zⱼ within this cluster."""
    local_transform::Matrix{Complex{T}}
end

"""
    NJDVBDResult

Container returned by [`miyajima_vbd_njd`](@ref) encapsulating the data
produced by the NJD-based verified block diagonalisation step
([Miyajima2021NJD](@cite), Section 4).  Field names are duck-type
compatible with [`MiyajimaVBDResult`](@ref) so that downstream consumers
(block Schur, spectral projectors) work unchanged.
"""
struct NJDVBDResult{MT, BT, IT, RT, ET, JT<:JordanBlockInfo}
    """Basis W = Q_reordered · blkdiag(Z₁,…,Zq) (generally NOT unitary)."""
    basis::MT
    """Interval matrix expressed in the chosen basis (W⁻¹ A W)."""
    transformed::BT
    """Block-diagonal truncation preserving the verified clusters."""
    block_diagonal::BT
    """Rigorous remainder satisfying transformed = block_diagonal + remainder."""
    remainder::BT
    """Index ranges identifying each spectral cluster."""
    clusters::Vector{UnitRange{Int}}
    """Gershgorin-type discs enclosing each diagonal entry."""
    cluster_intervals::Vector{IT}
    """Upper bound on ‖remainder‖₂ combining Collatz and block-separation bounds."""
    remainder_norm::RT
    """Eigenvalues associated with the diagonal of mid(transformed)."""
    eigenvalues::Vector{ET}
    """NJD-specific: Jordan structure per cluster."""
    jordan_info::Vector{JT}
    """Tolerance used for eigenvalue clustering."""
    clustering_tolerance::RT
    """Maximum nilpotent index across all clusters."""
    max_nilpotent_index::Int
end

# ── Phase 1: Schur + clustering + reorder ────────────────────────────

"""
    _njd_cluster_eigenvalues(eigenvalues, tol)

Cluster eigenvalues by proximity: build a graph where edge (i,j) exists
when |λᵢ - λⱼ| ≤ tol, then find connected components.  Returns a vector
of vectors of indices, one per cluster.
"""
function _njd_cluster_eigenvalues(eigenvalues::AbstractVector{<:Number}, tol::Real)
    n = length(eigenvalues)
    n == 0 && return Vector{Vector{Int}}()

    # Build proximity graph as Ball overlaps (reuse pattern from miyajima_vbd)
    T = typeof(float(real(tol)))
    half_tol = tol / 2
    balls = [Ball(complex(float(eigenvalues[i])), T(half_tol)) for i in 1:n]
    return overlap_components(balls)
end

"""
    _reorder_schur_by_clusters(F, cluster_assignments)

Given a Schur factorisation `F` and cluster assignments (vector of index
vectors), reorder the Schur form so that eigenvalues belonging to the
same cluster are contiguous on the diagonal.  Returns `(Q, T, eigenvalues,
cluster_ranges)` where cluster_ranges are UnitRange{Int} giving each
cluster's position in the reordered matrix.
"""
function _reorder_schur_by_clusters(F::Schur, cluster_assignments::Vector{Vector{Int}})
    n = size(F.T, 1)
    num_clusters = length(cluster_assignments)

    # Build the full ordering: clusters in order, indices within each cluster
    full_order = Int[]
    cluster_ranges = UnitRange{Int}[]
    pos = 1
    for component in cluster_assignments
        len = length(component)
        push!(cluster_ranges, pos:(pos + len - 1))
        append!(full_order, component)
        pos += len
    end

    # Use ordschur to reorder: process cluster by cluster.
    # We accumulate ordschur calls — each moves selected eigenvalues to
    # the top-left, so we process clusters in order from first to last.
    working_F = Schur(copy(Matrix(F.T)), copy(Matrix(F.Z)), copy(F.values))

    # We'll build up the reordering by selecting clusters from position 1 onward
    target_pos = 1
    for cl_idx in 1:num_clusters
        cluster_indices = cluster_assignments[cl_idx]
        cluster_size = length(cluster_indices)

        # Find which positions in the current working Schur form correspond
        # to eigenvalues that should be in this cluster.
        # Match by eigenvalue value (nearest unmatched).
        target_eigenvals = F.values[cluster_indices]

        select = falses(n)
        available = trues(n)
        # Don't re-select already-placed positions
        for i in 1:(target_pos - 1)
            available[i] = false
        end

        for tev in target_eigenvals
            best_idx = 0
            best_dist = Inf
            for j in 1:n
                if available[j] && !select[j]
                    d = abs(working_F.values[j] - tev)
                    if d < best_dist
                        best_dist = d
                        best_idx = j
                    end
                end
            end
            if best_idx > 0
                select[best_idx] = true
            end
        end

        # Also keep already-placed positions selected (they stay at top)
        for i in 1:(target_pos - 1)
            select[i] = true
        end

        working_F = ordschur(working_F, select)
        target_pos += cluster_size
    end

    return working_F.Z, working_F.T, working_F.values, cluster_ranges
end

# ── Phase 2: SVD staircase + Jordan chains ───────────────────────────

"""
    _svd_staircase(B0, tol)

SVD-based rank staircase for determining Jordan block sizes of a
nilpotent matrix B₀.

For k = 1, 2, …: compute svdvals(B₀^k) and count singular values below
`tol` to get d_k = dim(ker(B₀^k)).  Stop when d_k = n (nilpotent index).

Returns `(block_sizes, nilpotent_index)` where block_sizes[m] is the
number of Jordan blocks of size exactly m.
"""
function _svd_staircase(B0::AbstractMatrix, tol::Real)
    n = size(B0, 1)
    n == 0 && return (Int[], 0)

    # d_k = dim(ker(B0^k)) = number of singular values ≤ tol
    dims = Int[]  # dims[k] = d_k
    Bk = Matrix{eltype(B0)}(I, n, n)  # B0^0 = I

    nilpotent_index = 0
    for k in 1:n
        Bk = Bk * B0
        sv = svdvals(Bk)
        dk = count(s -> s ≤ tol, sv)
        push!(dims, dk)
        if dk == n
            nilpotent_index = k
            break
        end
    end

    # If we never reached d_k = n, the matrix isn't truly nilpotent within
    # tolerance.  Set nilpotent_index to the last k tested and treat
    # remaining dimension as a single block.
    if nilpotent_index == 0
        nilpotent_index = length(dims)
        # Force last entry to n for safety
        if !isempty(dims)
            dims[end] = n
        end
    end

    # Prepend d_0 = 0 for the differencing formula
    pushfirst!(dims, 0)
    # dims is now indexed 0:nilpotent_index via 1-based Julia as 1:(nilpotent_index+1)

    # Jordan block sizes from differences:
    # number of blocks of size ≥ m = d_m - d_{m-1}
    # number of blocks of size exactly m = (d_m - d_{m-1}) - (d_{m+1} - d_m)
    block_sizes = Int[]
    for m in 1:nilpotent_index
        geq_m = dims[m + 1] - dims[m]      # d_m - d_{m-1}
        if m < nilpotent_index
            geq_mp1 = dims[m + 2] - dims[m + 1]  # d_{m+1} - d_m
        else
            geq_mp1 = 0  # no blocks of size > nilpotent_index
        end
        exactly_m = geq_m - geq_mp1
        if exactly_m > 0
            append!(block_sizes, fill(m, exactly_m))
        end
    end

    # Sort descending (longest chains first) for chain computation
    sort!(block_sizes, rev=true)

    return block_sizes, nilpotent_index
end

"""
    _compute_jordan_chains(B0, block_sizes, nilpotent_index, tol)

Compute the Jordan chain vectors (principal vectors / generalised
eigenvectors) for the nilpotent matrix B₀ given the known Jordan
block sizes.

Returns a matrix Z whose columns form the Jordan basis, ordered by
chains from longest to shortest.  Within each chain the order is:
[B₀^{ℓ-1}·v, B₀^{ℓ-2}·v, …, B₀·v, v] (eigenvector first).
"""
function _compute_jordan_chains(B0::AbstractMatrix{CT}, block_sizes::Vector{Int},
                                 nilpotent_index::Int, tol::Real) where {CT}
    n = size(B0, 1)
    n == 0 && return Matrix{CT}(undef, 0, 0)

    # Pre-compute null space bases at each level
    null_bases = Vector{Matrix{CT}}(undef, nilpotent_index)
    Bk = Matrix{CT}(I, n, n)
    for k in 1:nilpotent_index
        Bk = Bk * B0
        F = svd(Bk)
        # Null space = columns of V corresponding to singular values ≤ tol
        null_cols = findall(s -> s ≤ tol, F.S)
        if isempty(null_cols)
            null_bases[k] = zeros(CT, n, 0)
        else
            null_bases[k] = F.Vt[null_cols, :]'
        end
    end

    # Build chains from longest to shortest
    Z_columns = Vector{Vector{CT}}()
    used_space = zeros(CT, n, 0)  # columns already used

    # block_sizes is sorted descending
    for chain_length in block_sizes
        # Find a chain starter v ∈ ker(B₀^ℓ) that is NOT in ker(B₀^{ℓ-1})
        # and is independent of previously used chain vectors at each level.

        # Get ker(B₀^ℓ) basis
        Nℓ = null_bases[chain_length]
        if size(Nℓ, 2) == 0
            # Fallback: use a random vector (shouldn't happen with correct block_sizes)
            v = randn(CT, n)
            v ./= norm(v)
        else
            # Find v in ker(B₀^ℓ) \ ker(B₀^{ℓ-1})
            if chain_length == 1
                # ker(B₀^0) = {0}, so any v in ker(B₀) works
                # but must be orthogonal to used_space
                v = _find_independent_vector(Nℓ, used_space)
            else
                # Project out ker(B₀^{ℓ-1}) from ker(B₀^ℓ) to find candidates
                Nℓm1 = null_bases[chain_length - 1]
                v = _find_chain_starter(Nℓ, Nℓm1, used_space)
            end
        end

        # Build the chain: [B₀^{ℓ-1}·v, …, B₀·v, v]
        chain = Vector{Vector{CT}}(undef, chain_length)
        chain[chain_length] = v  # highest-grade vector (chain starter)
        w = v
        for i in (chain_length - 1):-1:1
            w = B0 * w
            chain[i] = w
        end

        # Orthogonalise chain vectors for numerical stability
        for c in chain
            # Project out previously used space
            if size(used_space, 2) > 0
                c .-= used_space * (used_space' * c)
            end
            nrm = norm(c)
            if nrm > tol
                c ./= nrm
            end
        end

        # Add to result and update used space
        for c in chain
            push!(Z_columns, copy(c))
            used_space = hcat(used_space, c)
        end
    end

    # Assemble columns into matrix
    if isempty(Z_columns)
        return Matrix{CT}(I, n, n)
    end
    Z = hcat(Z_columns...)

    # If we don't have enough columns (shouldn't happen), pad with random orthogonal vectors
    if size(Z, 2) < n
        for _ in 1:(n - size(Z, 2))
            v = randn(CT, n)
            v .-= Z * (Z' * v)
            nrm = norm(v)
            if nrm > tol
                v ./= nrm
            end
            Z = hcat(Z, v)
        end
    end

    return Z
end

"""
    _find_independent_vector(basis, used_space)

Find a vector in the column span of `basis` that is independent of
`used_space`.  Returns a unit vector.
"""
function _find_independent_vector(basis::AbstractMatrix{CT}, used_space::AbstractMatrix{CT}) where {CT}
    if size(used_space, 2) == 0
        v = basis[:, 1]
        return v / norm(v)
    end

    # Project each basis column out of used_space, pick the one with largest residual
    best_v = zeros(CT, size(basis, 1))
    best_norm = 0.0
    for j in 1:size(basis, 2)
        v = basis[:, j] - used_space * (used_space' * basis[:, j])
        nrm = norm(v)
        if nrm > best_norm
            best_norm = nrm
            best_v = v
        end
    end

    return best_norm > 0 ? best_v / best_norm : basis[:, 1] / norm(basis[:, 1])
end

"""
    _find_chain_starter(Nℓ, Nℓm1, used_space)

Find a vector in ker(B₀^ℓ) that is NOT in ker(B₀^{ℓ-1}), and that is
independent of previously used chain vectors.
"""
function _find_chain_starter(Nℓ::AbstractMatrix{CT}, Nℓm1::AbstractMatrix{CT},
                              used_space::AbstractMatrix{CT}) where {CT}
    n = size(Nℓ, 1)

    # Project Nℓ columns orthogonally to both Nℓm1 and used_space
    # to find vectors in ker(B₀^ℓ) \ ker(B₀^{ℓ-1})
    combined = if size(Nℓm1, 2) > 0 && size(used_space, 2) > 0
        hcat(Nℓm1, used_space)
    elseif size(Nℓm1, 2) > 0
        Nℓm1
    elseif size(used_space, 2) > 0
        used_space
    else
        zeros(CT, n, 0)
    end

    best_v = zeros(CT, n)
    best_norm = 0.0

    if size(combined, 2) > 0
        # Orthogonal projector onto complement of combined
        # Use QR for numerical stability
        Q_comb = if size(combined, 2) > 0
            qr(combined).Q[:, 1:min(size(combined, 2), n)]
        else
            zeros(CT, n, 0)
        end

        for j in 1:size(Nℓ, 2)
            v = Nℓ[:, j]
            if size(Q_comb, 2) > 0
                v = v - Q_comb * (Q_comb' * v)
            end
            nrm = norm(v)
            if nrm > best_norm
                best_norm = nrm
                best_v = v
            end
        end

        # If projection killed everything, try random combinations
        if best_norm < 1e-14
            for _ in 1:10
                coeffs = randn(CT, size(Nℓ, 2))
                v = Nℓ * coeffs
                if size(Q_comb, 2) > 0
                    v = v - Q_comb * (Q_comb' * v)
                end
                nrm = norm(v)
                if nrm > best_norm
                    best_norm = nrm
                    best_v = v
                end
            end
        end
    else
        best_v = Nℓ[:, 1]
        best_norm = norm(best_v)
    end

    return best_norm > 0 ? best_v / best_norm : Nℓ[:, 1] / norm(Nℓ[:, 1])
end

# ── Phase 3: VBD transformation ──────────────────────────────────────

"""
    _build_canonical_nilpotent(block_sizes, n)

Build the canonical nilpotent matrix M_j = diag(N_{k1}, …, N_{ks})
where each N_k is a k×k shift matrix (ones on the superdiagonal).
Returns a sparse matrix.
"""
function _build_canonical_nilpotent(block_sizes::Vector{Int}, n::Int)
    M = zeros(Float64, n, n)
    pos = 1
    for k in block_sizes
        for i in 1:(k - 1)
            M[pos + i - 1, pos + i] = 1.0
        end
        pos += k
    end
    return M
end

# ── Phase 4: Theorem 2 perturbation bound ────────────────────────────

"""
    _theorem2_bound(jordan_info::JordanBlockInfo, R_j::AbstractMatrix, alpha::Real)

Compute the Theorem 2 perturbation bound for matrix power A^α,
exploiting the block-diagonal nilpotent structure of M_j.

From [Miyajima2021NJD](@cite), Section 4.2, Theorem 2:
```
T_j := (sin(απ)/(απ)) · (Σ_{i=0}^{nmax-1} Σ_{k=0}^{nmax-1}
        ζ_{i+k+2} · M_j^i · R_j · M_j^k) · (I + v·wᵀ)
```

Returns a scalar upper bound.
"""
function _theorem2_bound(jordan_info::JordanBlockInfo{T}, R_j::AbstractMatrix,
                          alpha::Real) where {T}
    nmax = jordan_info.nilpotent_index
    n_j = sum(jordan_info.block_sizes)
    λ = jordan_info.representative_eigenvalue

    abs_λ = abs(λ)

    # If eigenvalue is zero or alpha is non-positive, fall back
    if abs_λ < eps(T) || alpha <= 0
        return convert(T, Inf)
    end

    # Compute ζ_k coefficients (Eq. 6 of Miyajima 2021)
    # ζ_k = |α| · |α-1| · … · |α-k+1| / (k! · |λ|^k) for the binomial-type terms
    # Simplified: for integer α, many terms vanish.  For general α:
    ζ = zeros(T, 2 * nmax)
    for k in 1:(2 * nmax)
        # ζ_k = |binomial(α, k-1)| / |λ|^(k-1)  (shifted index)
        binom_coeff = one(T)
        for j in 0:(k - 2)
            binom_coeff *= abs(alpha - j) / (j + 1)
        end
        ζ[k] = setrounding(T, RoundUp) do
            binom_coeff / abs_λ^(k - 1)
        end
    end

    # Build M_j as canonical nilpotent
    M = _build_canonical_nilpotent(jordan_info.block_sizes, n_j)

    # Pre-compute M^i for i = 0, …, nmax-1
    M_powers = Vector{Matrix{Float64}}(undef, nmax)
    M_powers[1] = Matrix{Float64}(I, n_j, n_j)  # M^0 = I
    for i in 2:nmax
        M_powers[i] = M_powers[i - 1] * M
    end

    # Compute the double sum: S = Σ_{i=0}^{nmax-1} Σ_{k=0}^{nmax-1} ζ_{i+k+2} · |M^i · R_j · M^k|
    abs_R = abs.(R_j)
    S = zeros(T, n_j, n_j)
    for i in 0:(nmax - 1)
        Mi = M_powers[i + 1]
        for k in 0:(nmax - 1)
            Mk = M_powers[k + 1]
            idx = i + k + 2
            if idx <= length(ζ)
                term = setrounding(T, RoundUp) do
                    ζ[idx] .* abs.(Mi * abs_R * Mk)
                end
                S .= setrounding(T, RoundUp) do
                    S .+ term
                end
            end
        end
    end

    # sin(α·π) / (α·π) factor
    sinc_factor = setrounding(T, RoundUp) do
        abs(sin(T(alpha) * T(π))) / (abs(T(alpha)) * T(π))
    end

    # Upper bound: sinc_factor · ‖S‖_∞ · (1 + correction)
    # For simplicity, use ‖S‖_∞ (maximum absolute row sum)
    S_norm = setrounding(T, RoundUp) do
        maximum(sum(abs.(S), dims=2))
    end

    bound = setrounding(T, RoundUp) do
        sinc_factor * S_norm * 2  # factor 2 accounts for (I + v·w^T) norm bound
    end

    return isfinite(bound) ? bound : convert(T, Inf)
end

# ── Main entry point ─────────────────────────────────────────────────

"""
    miyajima_vbd_njd(A::BallMatrix; tol=nothing)

Perform NJD-based verified block diagonalisation on the square ball
matrix `A`.  This method works for defective and near-defective matrices
where the standard NSD-based [`miyajima_vbd`](@ref) may fail because
the eigenvector matrix X is singular.

The algorithm follows Miyajima 2021, Section 4:
1. Complex Schur form of `mid(A)` with eigenvalue clustering
2. SVD staircase (RDEFL) to determine Jordan block structure
3. Principal vector computation for Jordan chains
4. Transformation to Jordan-like basis with rigorous remainder bounds

# Arguments
- `A::BallMatrix`: Square ball matrix to decompose.
- `tol::Real = nothing`: Clustering tolerance.  Default: `√eps · ‖A‖`.

# Returns
[`NJDVBDResult`](@ref) duck-type compatible with [`MiyajimaVBDResult`](@ref).

# Example
```julia
# Exactly defective matrix
A = BallMatrix([1.0 1.0; 0.0 1.0])
result = miyajima_vbd_njd(A)
@assert result.max_nilpotent_index == 2  # single 2×2 Jordan block
```

# References

* [Miyajima2021NJD](@cite) Miyajima, S. (2021). "Verified computation of the matrix exponential",
  J. Comput. Appl. Math. 396, 113614. doi:10.1016/j.cam.2021.113614
* [KagstromRuhe1980](@cite) Kågström, B. & Ruhe, A. (1980). "An algorithm for numerical
  computation of the Jordan normal form of a complex matrix", ACM Trans.
  Math. Softw. 6(3), 398–419. doi:10.1145/355900.355917 (Algorithm 560)
"""
function miyajima_vbd_njd(A::BallMatrix{T, NT}; tol=nothing) where {T, NT}
    m, n = size(A)
    m == n || throw(ArgumentError("miyajima_vbd_njd expects a square matrix"))

    A_mid = mid(A)

    # Default tolerance
    if tol === nothing
        norm_A = upper_bound_L2_opnorm(A)
        tol = sqrt(eps(T)) * norm_A
    end
    tol = T(tol)

    # ── Phase 1: Complex Schur + clustering + reorder ────────────
    F = schur(complex(A_mid))

    # Cluster eigenvalues by proximity
    cluster_assignments = _njd_cluster_eigenvalues(F.values, tol)

    # Reorder Schur form to make clusters contiguous
    Q_reord, T_reord, eigenvals_reord, cluster_ranges = _reorder_schur_by_clusters(F, cluster_assignments)

    # ── Phase 2: Jordan structure per cluster ────────────────────
    CT = Complex{T}
    jordan_infos = JordanBlockInfo{T}[]
    local_transforms = Matrix{CT}[]

    svd_tol = tol  # tolerance for SVD rank determination

    for cl_range in cluster_ranges
        n_j = length(cl_range)
        T_jj = T_reord[cl_range, cl_range]

        # Representative eigenvalue: mean of cluster eigenvalues
        λ_hat = mean(eigenvals_reord[cl_range])

        if n_j == 1
            # Singleton cluster: trivial Jordan structure
            residual = zeros(CT, 1, 1)
            residual[1, 1] = T_jj[1, 1] - λ_hat
            push!(jordan_infos, JordanBlockInfo{T}(
                [1], 1, λ_hat, residual, ones(CT, 1, 1)
            ))
            push!(local_transforms, ones(CT, 1, 1))
            continue
        end

        # B₀ = triu(T_jj - λ̂·I, 1) — strictly upper triangular, exactly nilpotent
        B0 = triu(T_jj - λ_hat * I, 1)

        # SVD staircase to find Jordan block sizes
        block_sizes, nilpotent_idx = _svd_staircase(B0, svd_tol)

        # Handle edge case: if staircase finds no blocks
        if isempty(block_sizes)
            block_sizes = fill(1, n_j)
            nilpotent_idx = 1
        end

        # Ensure block sizes sum to n_j
        total = sum(block_sizes)
        if total < n_j
            append!(block_sizes, fill(1, n_j - total))
            sort!(block_sizes, rev=true)
        elseif total > n_j
            # Trim excess (shouldn't happen with correct algorithm)
            block_sizes = block_sizes[1:min(length(block_sizes), n_j)]
            while sum(block_sizes) > n_j && !isempty(block_sizes)
                pop!(block_sizes)
            end
            if sum(block_sizes) < n_j
                append!(block_sizes, fill(1, n_j - sum(block_sizes)))
            end
        end

        # Compute Jordan chains (principal vectors)
        Z_j = _compute_jordan_chains(B0, block_sizes, nilpotent_idx, svd_tol)

        # Diagonal residual: Z_j⁻¹ · T_jj · Z_j - λ̂·I - M_j
        # This is the small remainder Q_j
        M_j = Matrix(_build_canonical_nilpotent(block_sizes, n_j))
        if cond(Z_j) < 1 / eps(T)
            Q_j = Z_j \ T_jj * Z_j - λ_hat * I - M_j
        else
            # Z_j may be ill-conditioned, use pinv
            Q_j = pinv(Z_j) * T_jj * Z_j - λ_hat * I - M_j
        end

        push!(jordan_infos, JordanBlockInfo{T}(
            block_sizes, nilpotent_idx, λ_hat, Q_j, Z_j
        ))
        push!(local_transforms, Z_j)
    end

    # ── Phase 3: Assemble NJD basis and run VBD ──────────────────

    # Full basis: W = Q_reordered · blkdiag(Z₁, …, Zq)
    Z_blkdiag = zeros(CT, n, n)
    for (cl_idx, cl_range) in enumerate(cluster_ranges)
        Z_blkdiag[cl_range, cl_range] = local_transforms[cl_idx]
    end

    W = Q_reord * Z_blkdiag

    # Verify nonsingularity
    W_cond = cond(W)
    if W_cond > 1 / (10 * eps(T))
        @warn "NJD basis W is ill-conditioned (cond = $(W_cond)). Results may be unreliable."
    end

    # Transform in ball arithmetic
    W_ball = BallMatrix(W)
    W_inv = inv(W)
    W_inv_ball = BallMatrix(W_inv)
    transformed = W_inv_ball * A * W_ball

    # Gershgorin intervals (reuse from miyajima_vbd.jl)
    intervals = _vbd_gershgorin_intervals(transformed; hermitian=false)

    # Cluster intervals and reorder if needed
    clusters_final, order = _interval_clusters(intervals)

    identity_order = collect(1:n)
    if order != identity_order
        # Need to permute — update basis and re-transform
        W_perm = W[:, order]
        W_perm_ball = BallMatrix(W_perm)
        W_perm_inv = inv(W_perm)
        W_perm_inv_ball = BallMatrix(W_perm_inv)
        transformed = W_perm_inv_ball * A * W_perm_ball
        intervals = _vbd_gershgorin_intervals(transformed; hermitian=false)
        clusters_final, _ = _interval_clusters(intervals)
        W = W_perm
    end

    # Block diagonal extraction
    block = _block_diagonal_part(transformed, clusters_final)
    remainder = transformed - block

    # Remainder norm: min(Collatz bound, block-separation bound)
    collatz_bound = collatz_upper_bound_L2_opnorm(remainder)
    block_bound = r2_infty_bound_by_blocks(transformed, intervals, clusters_final)
    remainder_norm = isfinite(block_bound) ? min(collatz_bound, block_bound) : collatz_bound

    # Eigenvalues from the diagonal of mid(transformed)
    eigenvalues = [mid(transformed)[i, i] for i in 1:n]

    max_nilp = isempty(jordan_infos) ? 1 : maximum(ji.nilpotent_index for ji in jordan_infos)

    return NJDVBDResult(
        W, transformed, block, remainder,
        clusters_final, intervals, remainder_norm, eigenvalues,
        jordan_infos, tol, max_nilp
    )
end

# Helper: mean of a collection
function mean(v::AbstractVector)
    return sum(v) / length(v)
end
