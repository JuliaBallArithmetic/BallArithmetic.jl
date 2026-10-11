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

One certified deflating (invariant) subspace of the pencil `Ax = λBx`, returned
by [`gev_invariant_subspaces`](@ref).

# Fields
- `indices`: the cluster's column range in the certified basis
- `basis`: `W[:, indices]` — the columns of the VBD basis that **span** the
  subspace.
- `block`: the certified block `Λ_CC` (a `BallMatrix`); its spectrum is the
  cluster's eigenvalues, so `dim = length(indices)` counts them with multiplicity
- `discs`: the β-inflated Gershgorin discs of the cluster
- `residual`: rigorous bound on `‖A·basis − B·basis·block‖₂`, the deflating-
  subspace residual whose smallness certifies the subspace
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

Enclose **all** eigenvalues and deflating (invariant) subspaces of the pencil
`Ax = λBx`, one entry per certified cluster.

The subspaces are produced by the certified basis `W` of
[`miyajima2014a_schurnewton`](@ref)`(A, B)`: cluster `C` spans `W[:, C]`. Since
`Y·B·W = I + R₂` gives `W⁻¹(B⁻¹A)W = (I+R₂)⁻¹Ã = M`, the basis block-diagonalises
`B⁻¹A`, so each `W[:, C]` spans a deflating subspace of the pencil and carries
exactly `length(C)` eigenvalues (with multiplicity) — the ones enclosed by the
cluster's β-inflated discs.

What makes it rigorous is the residual `‖A·W[:, C] − B·W[:, C]·Λ_CC‖₂`, evaluated
in ball arithmetic against the *input balls*, so input uncertainty and every
rounding error are included. This is the cluster restriction of Miyajima's `R₁`,
before pre-multiplication by `Y`.

This is the invariant-subspace enclosure of Miyajima (2014) §3.3 obtained through
the Rump–Miyajima Schur–Newton frame, so it needs neither the Brouwer fixed point
on `ℂ^{n×k}` nor the parameter matrices of Lemmas 3.9–3.10 and Theorem 3.15.
`B` is assumed neither Hermitian nor positive definite; `‖R₂‖_∞ < 1` proves its
nonsingularity.

# Example
```julia
A = BallMatrix(randn(6, 6));  B = BallMatrix(Matrix(randn(6, 6)' * randn(6, 6) + 6I))
subs = gev_invariant_subspaces(A, B)
subs[1].basis        # spans the first deflating subspace
subs[1].residual     # rigorous ‖A·W_C − B·W_C·Λ_C‖₂
```
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
