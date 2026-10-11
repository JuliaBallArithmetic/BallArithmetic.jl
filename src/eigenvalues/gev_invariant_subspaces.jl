# The eigenvalues and deflating subspaces of a pencil A x = λ B x, cluster by cluster: a view of
# the result of Algorithms 1 and 2 of
#
#   S. Miyajima, "Fast enclosure for all eigenvalues and invariant subspaces in generalized
#   eigenvalue problems", SIAM Journal on Matrix Analysis and Applications 35 (2014), 1205-1225,
#   doi 10.1137/140953150,
#
# which are implemented in `miyajima_2014a.jl` and reached through `verifyeigall(A, B)`.
#
# Until 2026-10-12 this file returned columns of the Schur-Newton basis with a bound of their
# residual, which is not an enclosure of the subspace; and, before 2026-10-11, spectral projectors
# formed with a floating-point inverse (`miyajima_spectral_projectors`), which enclosed nothing.
# For the spectral projector of a matrix see `compute_spectral_projector_schur`.

"""
    GEVInvariantSubspace

One cluster of eigenvalues of the pencil `A x = λ B x` with its deflating subspace, returned by
[`gev_invariant_subspaces`](@ref).

# Fields
- `indices`: the indices of the cluster among the `n` approximate eigenvalues;
- `certified`: whether the subspace was enclosed (Theorem 3.5 for one index, Theorem 3.11
  through Algorithm 2 for several);
- `center`, `radius`: a disc containing exactly `length(indices)` eigenvalues of the pencil,
  counted with multiplicity;
- `basis`: when `certified`, a ball matrix containing a basis of the deflating subspace of those
  eigenvalues; its radii are infinite otherwise;
- `block`: when `certified`, a ball matrix containing the matrix `Λ` with
  `A X = B X Λ` for that basis `X`; its radii are infinite otherwise.
"""
struct GEVInvariantSubspace{T, CT}
    indices::Vector{Int}
    certified::Bool
    center::CT
    radius::T
    basis::BallMatrix{T, CT}
    block::BallMatrix{T, CT}
end

Base.size(s::GEVInvariantSubspace) = size(s.basis)
dim(s::GEVInvariantSubspace) = length(s.indices)

"""
    gev_invariant_subspaces(A::BallMatrix, B::BallMatrix)

Enclosures of all the eigenvalues of the pencil `A x = λ B x` and of their deflating subspaces,
one [`GEVInvariantSubspace`](@ref) for each cluster of eigenvalues, valid for every `A` and `B`
in the given balls. The vector returned is empty when the method fails, which includes the case
of a singular `B`: its nonsingularity is proved, not assumed.

This is [`verifyeigall`](@ref)`(A, B; method = :miyajima2014a)` read cluster by cluster, that
is Algorithms 1 and 2 of the paper below. With `X̃` the matrix of approximate eigenvectors
(`similarity` in the result of `verifyeigall`):

  - the discs of Corollary 3.2 are grouped into connected components, and a component of `k`
    discs contains exactly `k` eigenvalues (Remark 3.3);
  - for an isolated disc, Theorem 3.5 gives an eigenvector `x` with
    `|x − X̃ e_i| ≤ |X̃| J⁽ⁱ⁾ q`, normalised by `(X̃⁻¹ x)_i = 1`;
  - for a component `v` of `k > 1` discs, Theorem 3.11 gives a basis `X` of the deflating
    subspace in `⟨X̃ V, |X̃| U Uᵀ P̄_ε⟩`, normalised by `Vᵀ X̃⁻¹ X = I_k`, with `A X = B X Λ` and
    `Λ ∈ ⟨λ̃ I_k, Vᵀ P̄_ε⟩`, the `k` eigenvalues lying in the disc `⟨λ̃, ρ(Vᵀ P̄_ε)⟩`. Here `V`
    and `U` are the columns of the identity with indices in `v` and outside it.

Where Algorithm 2 declines on a component, the entry has `certified = false`, the disc is the
smallest one about the mean `λ̃` that contains the discs of the component, and `basis` and
`block` have infinite radii.

# Reference

S. Miyajima, "Fast enclosure for all eigenvalues and invariant subspaces in generalized
eigenvalue problems", SIAM Journal on Matrix Analysis and Applications 35 (2014), 1205-1225,
doi 10.1137/140953150.
"""
function gev_invariant_subspaces(A::BallMatrix{T}, B::BallMatrix{T}) where {T}
    size(A) == size(B) || throw(DimensionMismatch("A and B must have the same size"))
    r = verifyeigall(A, B; method = :miyajima2014a)
    CT = complex(T)
    out = GEVInvariantSubspace{T, CT}[]
    r.spectrum_covered || return out
    for i in eachindex(r.clusters)
        push!(out,
            GEVInvariantSubspace{T, CT}(r.clusters[i], r.certified[i], r.centers[i],
                r.radii[i], r.subspaces[i], r.blocks[i]))
    end
    return out
end
