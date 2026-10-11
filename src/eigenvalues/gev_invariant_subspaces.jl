# Deflating subspaces of a pencil A x = λ B x from the Schur-Newton block diagonalisation,
# after S. Miyajima, "Fast enclosure for all eigenvalues and invariant subspaces in generalized
# eigenvalue problems", SIAM J. Matrix Anal. Appl. 35 (2014) 1205-1225.
#
# The spectral projectors that this file used to return (`miyajima_spectral_projectors`, and the
# field `projector` here) were products with a floating-point inverse and enclosed nothing; they
# were removed on 2026-10-11. For the spectral projector of a matrix see
# `compute_spectral_projector_schur`, which bounds the distance to the projector through the
# resolvent on a contour.

"""
    GEVInvariantSubspace

One cluster of the block diagonalisation of the pencil `A x = λ B x`, returned by
[`gev_invariant_subspaces`](@ref).

# Fields
- `indices`: the columns of the cluster in the basis `W`;
- `basis`: `W[:, indices]`, an approximate basis of the deflating subspace of the cluster. It is
  not an enclosure of that subspace: no bound of the distance between the two is computed;
- `block`: the diagonal block `Λ_CC` of the transformed matrix, a `BallMatrix`;
- `discs`: the discs of the cluster from [`miyajima2014a_schurnewton`](@ref);
- `residual`: an upper bound of `‖A basis − B basis block‖₂`, valid for every `A` and `B` in the
  given balls.
"""
struct GEVInvariantSubspace{MT, BT, IT, RT}
    indices::UnitRange{Int}
    basis::MT
    block::BT
    discs::Vector{IT}
    residual::RT
end

Base.size(s::GEVInvariantSubspace) = size(s.basis)
dim(s::GEVInvariantSubspace) = length(s.indices)

"""
    gev_invariant_subspaces(A::BallMatrix, B::BallMatrix)

For each cluster `C` of the block diagonalisation [`miyajima2014a_schurnewton`](@ref)`(A, B)` of
the pencil `A x = λ B x`: the columns `W[:, C]` of its basis, the diagonal block `Λ_CC`, the discs
of the cluster, and an upper bound of the residual `‖A W[:, C] − B W[:, C] Λ_CC‖₂`. One
[`GEVInvariantSubspace`](@ref) per cluster.

What is verified is the residual: it is evaluated in ball arithmetic on the given balls, so it
holds for every `A` and `B` in them. The columns `W[:, C]` are an approximate basis of the
deflating subspace. An enclosure of the subspace itself, which is what Section 3.3 of the paper
below gives (Lemmas 3.9 and 3.10, Theorem 3.15), is not computed here.

# Reference

S. Miyajima, "Fast enclosure for all eigenvalues and invariant subspaces in generalized
eigenvalue problems", SIAM Journal on Matrix Analysis and Applications 35 (2014), 1205-1225,
doi 10.1137/140953150.
"""
function gev_invariant_subspaces(A::BallMatrix{T, NT}, B::BallMatrix) where {T, NT}
    size(A) == size(B) ||
        throw(DimensionMismatch("A and B must have the same size"))

    vbd = miyajima2014a_schurnewton(A, B)
    n = size(A, 1)
    W = BallMatrix(vbd.basis)
    CTm = eltype(mid(vbd.transformed))
    Acx = BallMatrix(Matrix{CTm}(mid(A)), rad(A))
    Bcx = BallMatrix(Matrix{CTm}(mid(B)), rad(B))

    out = GEVInvariantSubspace[]
    for cluster in vbd.clusters
        Wc = W[:, cluster]
        Λc = vbd.block_diagonal[cluster, cluster]
        # deflating-subspace residual: A·W_C − B·W_C·Λ_CC
        residual = upper_bound_L2_opnorm(Acx * Wc - (Bcx * Wc) * Λc)
        push!(out,
            GEVInvariantSubspace(cluster, Wc, Λc, vbd.cluster_intervals[cluster],
                residual))
    end
    return identity.(out)
end
