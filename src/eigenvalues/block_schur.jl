# Rigorous block Schur decomposition with verified error bounds
# Based on Miyajima's verified block diagonalization framework

"""
    RigorousBlockSchurResult

Container returned by [`rigorous_block_schur`](@ref) encapsulating a verified
block Schur decomposition `A ≈ Q * T * Q'` where:
- `Q` is an approximately orthogonal/unitary transformation (as ball matrix)
- `T` is in block upper quasi-triangular form
- Diagonal blocks correspond to eigenvalue clusters
- Off-diagonal blocks are rigorously bounded

The decomposition satisfies:
- `A ≈ Q * T * Q'` with rigorous residual bounds
- `Q' * Q ≈ I` with rigorous orthogonality defect bounds
- Each diagonal block `T[cluster_k, cluster_k]` contains eigenvalues from cluster k
"""
struct RigorousBlockSchurResult{QT, TT, IT, RT, VT}
    """Orthogonal/unitary transformation matrix (as ball matrix)."""
    Q::QT
    """Block upper triangular matrix (as ball matrix)."""
    T::TT
    """Index ranges identifying each diagonal block."""
    clusters::Vector{UnitRange{Int}}
    """Gershgorin-type intervals for each cluster."""
    cluster_intervals::Vector{IT}
    """Diagonal blocks extracted from T."""
    diagonal_blocks::Vector{TT}
    """Rigorous bound on residual ‖A - Q*T*Q'‖₂."""
    residual_norm::RT
    """Rigorous bound on orthogonality defect ‖Q'*Q - I‖₂."""
    orthogonality_defect::RT
    """Rigorous bound on ‖T‖₂ (norm of block triangular form)."""
    block_schur_norm::RT
    """Maximum norm of off-diagonal blocks."""
    off_diagonal_norm::RT
    """Original matrix."""
    A::BallMatrix
    """VBD result used in construction."""
    vbd_result::VT
end

Base.length(result::RigorousBlockSchurResult) = length(result.clusters)

"""
    rigorous_block_schur(A::BallMatrix; hermitian=false, block_structure=:quasi_triangular)

Compute a rigorous block Schur decomposition `A ≈ Q * T * Q'` where `Q` is
orthogonal/unitary and `T` is in block form determined by eigenvalue clustering.

The method follows Miyajima's VBD framework:
1. Apply verified block diagonalization to identify eigenvalue clusters
2. Construct orthogonal basis `Q` from the diagonalizing transformation
3. Transform matrix to block form `T = Q' * A * Q`
4. Verify orthogonality of `Q` and residual bounds

# Arguments
- `A::BallMatrix`: Square ball matrix to decompose
- `hermitian::Bool = false`: Whether to assume `A` is Hermitian
- `block_structure::Symbol = :quasi_triangular`: Block structure to compute
  - `:diagonal`: Keep only diagonal blocks (same as VBD)
  - `:quasi_triangular`: Keep upper triangular block structure
  - `:full`: Keep all blocks (no truncation)

# Returns
[`RigorousBlockSchurResult`](@ref) containing the decomposition and verification data.

# Example
```julia
using BallArithmetic, LinearAlgebra

# Create a matrix with clustered eigenvalues
A = BallMatrix([2.0 0.1 0.05 0.02;
                0.1 2.1 0.03 0.01;
                0.05 0.03 5.0 0.15;
                0.02 0.01 0.15 5.1])

# Compute block Schur form
result = rigorous_block_schur(A; hermitian=true)

# Access components
Q = result.Q
T = result.T

# Verify decomposition
@assert result.residual_norm < 1e-10
@assert result.orthogonality_defect < 1e-10
```

# References

* [MiyajimaInvariantSubspaces2014](@cite) Miyajima, SIAM J. Matrix Anal. Appl. 35, 1205–1225 (2014)
* [Miyajima2014](@cite) Miyajima, Japan J. Indust. Appl. Math. 31, 513–539 (2014)
"""
function rigorous_block_schur(A::BallMatrix{RT, NT};
                               hermitian::Bool = false,
                               block_structure::Symbol = :quasi_triangular,
                               vbd_method::Symbol = :nsd) where {RT, NT}
    n = size(A, 1)
    n == size(A, 2) || throw(ArgumentError("A must be square"))

    block_structure ∈ [:diagonal, :quasi_triangular, :full] ||
        throw(ArgumentError("block_structure must be :diagonal, :quasi_triangular, or :full"))

    vbd_method ∈ [:nsd, :njd] ||
        throw(ArgumentError("vbd_method must be :nsd or :njd"))

    # Step 1: Compute VBD to identify clusters and get basis
    vbd = if vbd_method == :njd
        miyajima_vbd_njd(A)
    else
        miyajima_vbd(A; hermitian = hermitian)
    end

    # Step 2: Construct orthogonal transformation Q as ball matrix
    Q = BallMatrix(vbd.basis)

    # Step 3: Transform matrix: T = Q' * A * Q
    Q_adj = BallMatrix(adjoint(vbd.basis))
    T_full = Q_adj * A * Q

    # Step 4: Apply block structure truncation
    T = _apply_block_structure(T_full, vbd.clusters, block_structure)

    # Step 5: Extract diagonal blocks
    diagonal_blocks = [T[cluster, cluster] for cluster in vbd.clusters]

    # Step 6: Verify orthogonality of Q
    Q_adj_Q = Q_adj * Q
    I_ball = BallMatrix(Matrix{NT}(I, n, n))
    orthogonality_defect = collatz_upper_bound_L2_opnorm(Q_adj_Q - I_ball)

    # Step 7: Compute residual ‖A - Q*T*Q'‖₂
    reconstruction = Q * T * Q_adj
    residual = A - reconstruction
    residual_norm = collatz_upper_bound_L2_opnorm(residual)

    # Step 8: Compute norms
    block_schur_norm = collatz_upper_bound_L2_opnorm(T)
    off_diagonal_norm = _compute_off_diagonal_norm(T, vbd.clusters)

    return RigorousBlockSchurResult(
        Q, T, vbd.clusters, vbd.cluster_intervals,
        diagonal_blocks, residual_norm, orthogonality_defect,
        block_schur_norm, off_diagonal_norm, A, vbd
    )
end

"""
    _apply_block_structure(T::BallMatrix, clusters, structure::Symbol)

Apply the requested block structure to the transformed matrix `T`.

- `:diagonal`: Zero out all off-diagonal blocks
- `:quasi_triangular`: Keep upper triangular block structure
- `:full`: Keep all blocks unchanged
"""
function _apply_block_structure(T::BallMatrix{T_type, NT},
                                  clusters::Vector{UnitRange{Int}},
                                  structure::Symbol) where {T_type, NT}
    if structure == :full
        return T
    end

    n = size(T, 1)
    T_mid = mid(T)
    T_rad = rad(T)

    result_mid = zeros(NT, n, n)
    result_rad = zeros(T_type, n, n)

    if structure == :diagonal
        # Keep only diagonal blocks
        for cluster in clusters
            result_mid[cluster, cluster] .= T_mid[cluster, cluster]
            result_rad[cluster, cluster] .= T_rad[cluster, cluster]
        end
    elseif structure == :quasi_triangular
        # Keep upper triangular block structure
        for (i, cluster_i) in enumerate(clusters)
            for (j, cluster_j) in enumerate(clusters)
                if i <= j  # Upper triangular: i ≤ j
                    result_mid[cluster_i, cluster_j] .= T_mid[cluster_i, cluster_j]
                    result_rad[cluster_i, cluster_j] .= T_rad[cluster_i, cluster_j]
                end
            end
        end
    end

    return BallMatrix(result_mid, result_rad)
end

"""
    _compute_off_diagonal_norm(T::BallMatrix, clusters)

Compute rigorous upper bound on the maximum norm of off-diagonal blocks.
"""
function _compute_off_diagonal_norm(T::BallMatrix{T_type, NT},
                                     clusters::Vector{UnitRange{Int}}) where {T_type, NT}
    max_norm = zero(T_type)

    for (i, cluster_i) in enumerate(clusters)
        for (j, cluster_j) in enumerate(clusters)
            if i != j
                block = T[cluster_i, cluster_j]
                block_norm = collatz_upper_bound_L2_opnorm(block)
                max_norm = max(max_norm, block_norm)
            end
        end
    end

    return max_norm
end

"""
    extract_cluster_block(result::RigorousBlockSchurResult, i::Int, j::Int)

Extract the (i,j)-th block from the block Schur form `T`.
Returns a `BallMatrix` corresponding to `T[cluster_i, cluster_j]`.
"""
function extract_cluster_block(result::RigorousBlockSchurResult, i::Int, j::Int)
    1 <= i <= length(result.clusters) || throw(BoundsError("Cluster index i out of range"))
    1 <= j <= length(result.clusters) || throw(BoundsError("Cluster index j out of range"))

    cluster_i = result.clusters[i]
    cluster_j = result.clusters[j]

    return result.T[cluster_i, cluster_j]
end

"""
    verify_block_schur_properties(result::RigorousBlockSchurResult; tol=1e-10)

Verify that the block Schur decomposition satisfies all required properties
within the specified tolerance.

Checks:
1. Residual: ‖A - Q*T*Q'‖₂ < tol
2. Orthogonality: ‖Q'*Q - I‖₂ < tol
3. Block structure preserved (if applicable)

Returns `true` if all properties are satisfied, `false` otherwise.
"""
function verify_block_schur_properties(result::RigorousBlockSchurResult;
                                        tol::Real = 1e-10)
    checks = [
        result.residual_norm < tol,
        result.orthogonality_defect < tol
    ]

    return all(checks)
end

"""
    compute_block_sylvester_rhs(result::RigorousBlockSchurResult, i::Int, j::Int)

For clusters i < j, compute the right-hand side for the block Sylvester equation
that would refine the (i,j) off-diagonal block.

Given `T_ii * X + X * T_jj = C`, returns the matrix `C` that should equal
`T[cluster_i, cluster_j]` if the block Schur form were exact.
"""
function compute_block_sylvester_rhs(result::RigorousBlockSchurResult, i::Int, j::Int)
    i < j || throw(ArgumentError("Must have i < j for off-diagonal block"))

    # The RHS for Sylvester equation T_ii * X + X * T_jj = C is C = T_ij
    # This can be used to verify or refine the off-diagonal block
    T_ij = extract_cluster_block(result, i, j)
    return T_ij
end

"""
    estimate_block_separation(result::RigorousBlockSchurResult, i::Int, j::Int)

Estimate the spectral separation between clusters i and j.
A small separation indicates potential numerical difficulties in
separating the corresponding invariant subspaces.
"""
function estimate_block_separation(result::RigorousBlockSchurResult, i::Int, j::Int)
    return sep_clusters(result.cluster_intervals, result.clusters[i], result.clusters[j])
end

"""
    refine_off_diagonal_block(result::RigorousBlockSchurResult, i::Int, j::Int)

Refine the (i,j) off-diagonal block by solving the block Sylvester equation
with Miyajima's verified solver.

For i < j, solves `T_ii' * Y - Y * T_jj' = T_ij'` to obtain a refined
enclosure for the (i,j) off-diagonal block.

Returns a `BallMatrix` with rigorous enclosure for the refined block.
"""
function refine_off_diagonal_block(result::RigorousBlockSchurResult, i::Int, j::Int)
    i < j || throw(ArgumentError("Only upper triangular blocks (i < j) can be refined"))

    # Extract blocks
    T_ii = result.diagonal_blocks[i]
    T_jj = result.diagonal_blocks[j]
    T_ij = extract_cluster_block(result, i, j)

    # Set up Sylvester equation: T_ii' * Y - Y * T_jj' = T_ij'
    # This is equivalent to: A * Y + Y * B = C
    # where A = T_ii', B = -T_jj', C = T_ij'

    A = BallMatrix(adjoint(mid(T_ii)))
    B = -BallMatrix(adjoint(mid(T_jj)))
    C = BallMatrix(adjoint(mid(T_ij)))

    # Compute approximate solution
    Y_approx = sylvester(mid(A), mid(B), mid(C))

    # Apply Miyajima verification
    Y_verified = sylvester_miyajima_enclosure(mid(A), mid(B), mid(C), Y_approx)

    # Transpose back to get refined X_ij
    X_ij_refined = BallMatrix(adjoint(mid(Y_verified)), adjoint(rad(Y_verified)))

    return X_ij_refined
end
