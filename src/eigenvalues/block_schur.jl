# A block form of a matrix in the basis of a verified block diagonalisation, with the two residuals
# that say how far it is from a similarity.

"""
    RigorousBlockSchurResult

Returned by [`rigorous_block_schur`](@ref). For the matrix `A`, a basis `Q`, a matrix
`Q_inv ≈ Q⁻¹` computed in floating point, and a block matrix `T`, it holds the two residuals

    R₁ = Q_inv (A Q − Q T),        R₂ = Q_inv Q − I,

through upper bounds of their spectral norms, `projected_residual_norm` and
`inverse_defect_norm`. When `inverse_defect_norm < 1`, `Q` is invertible,
`Q⁻¹ = (I + R₂)⁻¹ Q_inv`, and

    Q⁻¹ A Q = T + S,        S = (I + R₂)⁻¹ R₁,        ‖S‖₂ ≤ ‖R₁‖₂ / (1 − ‖R₂‖₂),

an identity; the last bound is `perturbation_norm` (`Inf` when `inverse_defect_norm ≥ 1`). So `A`
is similar to a matrix within `perturbation_norm` of `T`, and no inverse is enclosed. The
statements hold for every `A`, `Q` and `T` in the respective balls.

`Q_inv` is the adjoint of `Q` when the basis is unitary and `inv` of its midpoint otherwise; it is
NOT `Q'` in general.
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
    """Upper bound of ‖Q_inv (A Q − Q T)‖₂."""
    projected_residual_norm::RT
    """Upper bound of ‖Q_inv Q − I‖₂."""
    inverse_defect_norm::RT
    """Upper bound of ‖S‖₂ in Q⁻¹AQ = T + S; Inf when the inverse defect is not below 1."""
    perturbation_norm::RT
    """The floating-point approximate inverse of Q the residuals were computed with."""
    Q_inv::QT
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
    rigorous_block_schur(A::BallMatrix; hermitian = false, block_structure = :quasi_triangular,
                         vbd_method = :auto)

A block form `T` of `A` in the basis `Q` of a verified block diagonalisation, with the residuals
that bound its distance from a similarity. See [`RigorousBlockSchurResult`](@ref) for the
statement that the result carries.

# Steps

1. The verified block diagonalisation gives the clusters, their intervals and the basis:
   [`miyajima2014a_schurnewton`](@ref) (Schur form followed by a Newton step for the decoupling)
   for a non-Hermitian matrix, [`schur_gershgorin_enclosure`](@ref) for a Hermitian one
   (`vbd_method = :schur_newton` or `:nsd`; `:auto` chooses by `hermitian`). The clusters and
   intervals of the result are the certified ones of that routine.
2. `Q` is that basis and `Q_inv` an approximate inverse in floating point (the adjoint for a
   unitary basis).
3. `T = Q_inv A Q` in ball arithmetic, with the blocks that `block_structure` drops set to zero:
   `:diagonal` keeps the diagonal blocks, `:quasi_triangular` the blocks on and above the
   diagonal, `:full` everything.
4. `R₁ = Q_inv (A Q − Q T)` and `R₂ = Q_inv Q − I` are computed in ball arithmetic and their
   spectral norms bounded from above; the dropped blocks are therefore in `R₁`.

The device of an arbitrary `Y ≈ X⁻¹` with the two residuals `Y(AX − XD)` and `YX − I` in place of
an enclosed inverse is that of Theorem 2 of

S. Miyajima, *Fast enclosure for all eigenvalues and invariant subspaces in generalized
eigenvalue problems*, SIAM J. Matrix Anal. Appl. 35 (2014), 1205–1225,
doi:10.1137/140953150 ([Miyajima2014a](@cite)),

used here with a block matrix `T` in place of the diagonal one.
"""
function rigorous_block_schur(A::BallMatrix{RT, NT};
                               hermitian::Bool = false,
                               block_structure::Symbol = :quasi_triangular,
                               vbd_method::Symbol = :auto) where {RT, NT}
    n = size(A, 1)
    n == size(A, 2) || throw(ArgumentError("A must be square"))

    block_structure ∈ [:diagonal, :quasi_triangular, :full] ||
        throw(ArgumentError("block_structure must be :diagonal, :quasi_triangular, or :full"))

    vbd_method ∈ [:auto, :nsd, :schur_newton, :njd] ||
        throw(ArgumentError("vbd_method must be :auto, :nsd, or :schur_newton"))
    if vbd_method == :njd
        @warn "vbd_method = :njd is deprecated; use :schur_newton (the O(n³) Schur+Newton VBD)." maxlog=1
        vbd_method = :schur_newton
    end
    # :auto — NSD only diagonalizes a NORMAL matrix; for a non-normal (non-hermitian) matrix
    # a unitary basis leaves all the non-normality in the off-diagonal (huge coupling), so
    # default to Schur+Newton, which decouples between clusters and confines the
    # non-normality to the diagonal blocks.  Hermitian ⇒ NSD (a unitary basis diagonalizes).
    vbd_method == :auto && (vbd_method = hermitian ? :nsd : :schur_newton)

    # Step 1: Compute VBD to identify clusters and get basis
    vbd = if vbd_method == :schur_newton
        miyajima2014a_schurnewton(A)
    else
        schur_gershgorin_enclosure(A; hermitian = hermitian)
    end

    # Step 2: Construct the basis Q and its inverse.  For a unitary (NSD) basis the
    # inverse is the adjoint; for the block-orthonormal Schur+Newton basis it is not,
    # so use inv(W) — otherwise T = Qinv*A*Q is not a similarity and the result is wrong.
    is_unitary = _vbd_unitary_basis(vbd)
    Q = BallMatrix(vbd.basis)
    Q_inv = is_unitary ? BallMatrix(adjoint(vbd.basis)) : BallMatrix(inv(vbd.basis))

    # Step 3: Transform matrix: T = Q⁻¹ * A * Q
    T_full = Q_inv * A * Q

    # Step 4: Apply block structure truncation
    T = _apply_block_structure(T_full, vbd.clusters, block_structure)

    # Step 5: Extract diagonal blocks
    diagonal_blocks = [T[cluster, cluster] for cluster in vbd.clusters]

    # Step 6: the two residuals. R₂ = Q_inv Q − I says how far Q_inv is from the inverse of Q,
    # R₁ = Q_inv (A Q − Q T) how far T is from Q_inv A Q; with ‖R₂‖ < 1, Q⁻¹AQ = T + (I + R₂)⁻¹R₁.
    I_ball = BallMatrix(Matrix{NT}(I, n, n))
    inverse_defect_norm = collatz_upper_bound_L2_opnorm(Q_inv * Q - I_ball)
    projected_residual_norm = collatz_upper_bound_L2_opnorm(Q_inv * (A * Q - Q * T))
    perturbation_norm = inverse_defect_norm < 1 ?
                        div_up(projected_residual_norm, sub_down(one(RT), inverse_defect_norm)) :
                        RT(Inf)

    # Step 7: norms of T
    block_schur_norm = collatz_upper_bound_L2_opnorm(T)
    off_diagonal_norm = _compute_off_diagonal_norm(T, vbd.clusters)

    return RigorousBlockSchurResult(
        Q, T, vbd.clusters, vbd.cluster_intervals,
        diagonal_blocks, projected_residual_norm, inverse_defect_norm, perturbation_norm,
        Q_inv, block_schur_norm, off_diagonal_norm, A, vbd
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
    verify_block_schur_properties(result::RigorousBlockSchurResult; tol = 1e-10)

Whether `projected_residual_norm` and `inverse_defect_norm` are both below `tol`.
"""
verify_block_schur_properties(result::RigorousBlockSchurResult; tol::Real = 1e-10) =
    result.projected_residual_norm < tol && result.inverse_defect_norm < tol

"""
    estimate_block_separation(result::RigorousBlockSchurResult, i::Int, j::Int)

Estimate the spectral separation between clusters i and j.
A small separation indicates potential numerical difficulties in
separating the corresponding invariant subspaces.
"""
function estimate_block_separation(result::RigorousBlockSchurResult, i::Int, j::Int)
    return sep_clusters(result.cluster_intervals, result.clusters[i], result.clusters[j])
end
