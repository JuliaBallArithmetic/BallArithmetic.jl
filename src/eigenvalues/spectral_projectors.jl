# Rigorous spectral projectors from verified block diagonalization
# Based on Miyajima, S. "Fast enclosure for all eigenvalues and invariant
# subspaces in generalized eigenvalue problems", SIAM J. Matrix Anal. Appl.
# 35, 1205–1225 (2014)

"""
    RigorousSpectralProjectorsResult

Container returned by [`miyajima_spectral_projectors`](@ref) encapsulating
the rigorously computed spectral projectors for each eigenvalue cluster
identified by verified block diagonalization (VBD).

Each projector `P_k` is a ball matrix satisfying:
- `P_k^2 ≈ P_k` (idempotency)
- `∑_k P_k ≈ I` (resolution of identity)
- `P_i * P_j ≈ 0` for `i ≠ j` (orthogonality)
- `A * P_k ≈ P_k * A * P_k` (invariance)

The projectors are constructed from the basis that block-diagonalizes the
matrix, restricted to each spectral cluster.
"""
struct RigorousSpectralProjectorsResult{PT, IT, RT, VT}
    """Vector of ball matrix projectors, one per cluster."""
    projectors::Vector{PT}
    """Index ranges identifying each spectral cluster."""
    clusters::Vector{UnitRange{Int}}
    """Gershgorin-type intervals for each cluster."""
    cluster_intervals::Vector{IT}
    """Upper bound on idempotency defect max_k ‖P_k^2 - P_k‖₂."""
    idempotency_defect::RT
    """Upper bound on orthogonality defect max_{i≠j} ‖P_i * P_j‖₂."""
    orthogonality_defect::RT
    """Upper bound on resolution defect ‖∑_k P_k - I‖₂."""
    resolution_defect::RT
    """Upper bound on invariance defect max_k ‖A*P_k - P_k*A*P_k‖₂."""
    invariance_defect::RT
    """Original matrix for verification."""
    A::BallMatrix
    """VBD result used to construct projectors."""
    vbd_result::VT
end

Base.length(result::RigorousSpectralProjectorsResult) = length(result.projectors)
Base.getindex(result::RigorousSpectralProjectorsResult, i::Int) = result.projectors[i]
Base.iterate(result::RigorousSpectralProjectorsResult) = iterate(result.projectors)
function Base.iterate(result::RigorousSpectralProjectorsResult, state)
    iterate(result.projectors, state)
end

"""
    miyajima_spectral_projectors(A::BallMatrix; hermitian=false, verify_invariance=true)

Compute rigorous enclosures for spectral projectors corresponding to each
eigenvalue cluster identified by Miyajima's verified block diagonalization (VBD).

The method follows the approach from Ref. [Miyajima2014a](@cite):
1. Apply VBD to obtain basis `V` that block-diagonalizes `A`
2. For each cluster `k`, extract columns `V[:, cluster_k]`
3. Construct projector `P_k = V[:, cluster_k] * V[:, cluster_k]'` as ball matrix
4. Verify idempotency, orthogonality, and resolution of identity

When `hermitian = true`, the basis is computed via eigendecomposition and
projectors are Hermitian. Otherwise, the Schur basis is used.

When `verify_invariance = true`, additionally verifies that `A * P_k ≈ P_k * A * P_k`
for each projector, confirming that the columns of `P_k` span an invariant subspace.

# Arguments
- `A::BallMatrix`: Square ball matrix whose spectral projectors to compute
- `hermitian::Bool = false`: Whether to assume `A` is Hermitian
- `verify_invariance::Bool = true`: Whether to verify invariant subspace property

# Returns
[`RigorousSpectralProjectorsResult`](@ref) containing the projectors and verification data.

# Example
```julia
using BallArithmetic, LinearAlgebra

# Create a matrix with clustered eigenvalues
A = BallMatrix(Diagonal([1.0, 1.1, 5.0, 5.1]))

# Compute projectors
result = miyajima_spectral_projectors(A; hermitian=true)

# Access projectors
P1 = result[1]  # Projector for first cluster (eigenvalues ≈ 1.0, 1.1)
P2 = result[2]  # Projector for second cluster (eigenvalues ≈ 5.0, 5.1)

# Verify properties
@assert result.idempotency_defect < 1e-10
@assert result.orthogonality_defect < 1e-10
```

# References

* [Miyajima2014a](@cite) Miyajima, SIAM J. Matrix Anal. Appl. 35, 1205–1225 (2014)
"""
function miyajima_spectral_projectors(A::BallMatrix{T, NT};
        hermitian::Bool = false,
        verify_invariance::Bool = true,
        vbd_method::Symbol = :auto) where {T, NT}
    vbd_method ∈ [:auto, :nsd, :schur_newton, :njd] ||
        throw(ArgumentError("vbd_method must be :auto, :nsd, or :schur_newton"))
    if vbd_method == :njd
        @warn "vbd_method = :njd is deprecated; use :schur_newton (the O(n³) Schur+Newton VBD)." maxlog=1
        vbd_method = :schur_newton
    end
    # :auto — NSD (unitary basis) only diagonalizes a NORMAL matrix; for a non-normal
    # (non-hermitian) matrix it leaves the non-normality in the off-diagonal, so default to
    # Schur+Newton (decouples between clusters, non-normality confined to the blocks).
    vbd_method == :auto && (vbd_method = hermitian ? :nsd : :schur_newton)

    # Step 1: Compute VBD
    vbd = if vbd_method == :schur_newton
        schur_newton_vbd(A)
    else
        schur_gershgorin_enclosure(A; hermitian = hermitian)
    end

    # Step 2: Extract basis and its inverse/adjoint
    V = BallMatrix(vbd.basis)
    n = size(A, 1)
    num_clusters = length(vbd.clusters)

    # For unitary bases (NSD), P_k = V_k * V_k^† is correct.
    # For non-unitary bases (Schur+Newton), P_k = W_k * (W⁻¹)_k where
    # W_k = W[:, cluster_k] and (W⁻¹)_k = (W⁻¹)[cluster_k, :].
    is_unitary_basis = _vbd_unitary_basis(vbd)

    V_inv_ball = if is_unitary_basis
        nothing  # not needed — use adjoint instead
    else
        BallMatrix(inv(vbd.basis))
    end

    # Step 3: Construct projector for each cluster
    projectors_list = BallMatrix[]

    for cluster in vbd.clusters
        V_k = V[:, cluster]

        P_k = if is_unitary_basis
            # Unitary basis: P_k = V_k * V_k^†
            V_k_adj = BallMatrix(adjoint(vbd.basis[:, cluster]))
            V_k * V_k_adj
        else
            # Non-unitary basis: P_k = W[:, cluster] * (W⁻¹)[cluster, :]
            V_inv_k = V_inv_ball[cluster, :]
            V_k * V_inv_k
        end

        push!(projectors_list, P_k)
    end

    # Convert to typed vector
    projectors = identity.(projectors_list)

    # Step 4: Verify projector properties

    # Verify idempotency: P_k^2 ≈ P_k
    idempotency_defect = zero(T)
    for P in projectors
        P_squared = P * P
        defect = upper_bound_L2_opnorm(P_squared - P)
        idempotency_defect = max(idempotency_defect, defect)
    end

    # Verify orthogonality: P_i * P_j ≈ 0 for i ≠ j
    orthogonality_defect = zero(T)
    for i in 1:num_clusters
        for j in (i + 1):num_clusters
            product = projectors[i] * projectors[j]
            defect = upper_bound_L2_opnorm(product)
            orthogonality_defect = max(orthogonality_defect, defect)
        end
    end

    # Verify resolution of identity: ∑ P_k ≈ I
    sum_projectors = sum(projectors)
    # Match identity element type to projector element type (Schur+Newton may produce complex)
    proj_eltype = eltype(mid(projectors[1]))
    I_ball = BallMatrix(Matrix{proj_eltype}(I, n, n))
    resolution_defect = upper_bound_L2_opnorm(sum_projectors - I_ball)

    # Verify invariance: A * P_k ≈ P_k * A * P_k
    invariance_defect = zero(T)
    if verify_invariance
        for P in projectors
            AP = A * P
            PAP = P * AP
            defect = upper_bound_L2_opnorm(AP - PAP)
            invariance_defect = max(invariance_defect, defect)
        end
    end

    return RigorousSpectralProjectorsResult(
        projectors,
        vbd.clusters,
        vbd.cluster_intervals,
        idempotency_defect,
        orthogonality_defect,
        resolution_defect,
        invariance_defect,
        A,
        vbd
    )
end

"""
    compute_invariant_subspace_basis(proj_result::RigorousSpectralProjectorsResult, k::Int)

Extract an orthonormal basis for the invariant subspace corresponding to
cluster `k` from the spectral projector result.

Returns a `BallMatrix` whose columns span the invariant subspace associated
with the k-th eigenvalue cluster.
"""
function compute_invariant_subspace_basis(proj_result::RigorousSpectralProjectorsResult, k::Int)
    cluster = proj_result.clusters[k]
    V = BallMatrix(proj_result.vbd_result.basis)
    return V[:, cluster]
end

"""
    verify_projector_properties(proj_result::RigorousSpectralProjectorsResult; tol=1e-10)

Verify that all projector properties hold within the specified tolerance.
Returns `true` if all properties are satisfied, `false` otherwise.

Checks:
1. Idempotency: ‖P_k^2 - P_k‖₂ < tol for all k
2. Orthogonality: ‖P_i * P_j‖₂ < tol for all i ≠ j
3. Resolution: ‖∑_k P_k - I‖₂ < tol
4. Invariance: ‖A*P_k - P_k*A*P_k‖₂ < tol for all k (if computed)
"""
function verify_projector_properties(proj_result::RigorousSpectralProjectorsResult;
        tol::Real = 1e-10)
    checks = [
        proj_result.idempotency_defect < tol,
        proj_result.orthogonality_defect < tol,
        proj_result.resolution_defect < tol
    ]

    # Only check invariance if it was computed (non-zero value)
    if proj_result.invariance_defect > 0
        push!(checks, proj_result.invariance_defect < tol)
    end

    return all(checks)
end

"""
    projector_condition_number(proj_result::RigorousSpectralProjectorsResult, k::Int)

Estimate the condition number of the k-th spectral projector based on
the gap between eigenvalue clusters.

A small gap indicates potential ill-conditioning of the projector.
"""
function projector_condition_number(proj_result::RigorousSpectralProjectorsResult, k::Int)
    vbd = proj_result.vbd_result
    clusters = vbd.clusters
    intervals = vbd.cluster_intervals

    # Find minimum separation to other clusters
    min_sep = Inf
    for j in 1:length(clusters)
        if j != k
            sep = sep_clusters(intervals, clusters[k], clusters[j])
            min_sep = min(min_sep, sep)
        end
    end

    # Condition number scales inversely with separation
    # κ(P) ∼ ‖A‖ / gap
    A = proj_result.A
    norm_A = upper_bound_L2_opnorm(A)

    return min_sep > 0 ? norm_A / min_sep : convert(radtype(typeof(A)), Inf)
end

"""
    miyajima_spectral_projectors(A::BallMatrix, B::BallMatrix; verify_invariance = true,
                                 vbd_method = :schur_newton)

Rigorous spectral projectors — equivalently, enclosures of the **deflating
(invariant) subspaces** — for the pencil `Ax = λBx`.

Built on [`schur_newton_vbd`](@ref)`(A, B)`. Because `Y·B·W = I + R₂`, we have
`(BW)⁻¹ = (I+R₂)⁻¹Y` and hence

    W⁻¹(B⁻¹A)W = (I+R₂)⁻¹·Y·A·W = (I+R₂)⁻¹Ã = M ,

so the certified basis `W` block-diagonalises `B⁻¹A` and the projector for a
cluster `C` is the *same* expression as for the standard problem,

    P_C = W[:, C] · (W⁻¹)[C, :] ,

with `W⁻¹` well defined because `‖R₂‖ < 1` proves `W` nonsingular. `B⁻¹` is never
formed.

Idempotency, mutual orthogonality and resolution of the identity are checked as
in the one-argument method. Invariance is reported through the pencil residual

    max_C ‖A·W[:, C] − B·W[:, C]·Λ_CC‖₂ ,

`Λ` being the certified block-diagonal candidate: this is the cluster restriction
of Miyajima's `R₁` (before pre-multiplication by `Y`), and its smallness is what
certifies `span(W[:, C])` as a deflating subspace. Every quantity is a rigorous
ball bound.

This realises the invariant-subspace enclosure of Miyajima (2014) §3.3 through
the Rump–Miyajima Schur–Newton frame, avoiding the Brouwer fixed point on
`ℂ^{n×k}` and its parameterised linear systems.

`B` need be neither Hermitian nor positive definite; its nonsingularity is proved
by `‖R₂‖_∞ < 1` rather than assumed.
"""
function miyajima_spectral_projectors(A::BallMatrix{T, NT}, B::BallMatrix;
        verify_invariance::Bool = true,
        vbd_method::Symbol = :schur_newton) where {T, NT}
    size(A) == size(B) ||
        throw(DimensionMismatch("A and B must have the same size"))
    vbd_method ∈ (:schur_newton, :njd, :auto) ||
        throw(ArgumentError("only :schur_newton is available for the generalized problem"))

    vbd = schur_newton_vbd(A, B)
    n = size(A, 1)

    W = BallMatrix(vbd.basis)
    Winv = BallMatrix(inv(vbd.basis))          # ‖R₂‖<1 proved W nonsingular

    projectors = BallMatrix[]
    for cluster in vbd.clusters
        push!(projectors, W[:, cluster] * Winv[cluster, :])
    end
    projectors = identity.(projectors)

    idempotency_defect = zero(T)
    for P in projectors
        idempotency_defect = max(idempotency_defect,
            upper_bound_L2_opnorm(P * P - P))
    end

    orthogonality_defect = zero(T)
    for i in 1:length(projectors), j in (i + 1):length(projectors)

        orthogonality_defect = max(orthogonality_defect,
            upper_bound_L2_opnorm(projectors[i] * projectors[j]))
    end

    proj_eltype = eltype(mid(projectors[1]))
    resolution_defect = upper_bound_L2_opnorm(sum(projectors) -
                                              BallMatrix(Matrix{proj_eltype}(I, n, n)))

    # Deflating-subspace residual ‖A·W_C − B·W_C·Λ_CC‖₂ per cluster.
    invariance_defect = zero(T)
    if verify_invariance
        Acx = BallMatrix(Matrix{proj_eltype}(mid(A)), rad(A))
        Bcx = BallMatrix(Matrix{proj_eltype}(mid(B)), rad(B))
        for cluster in vbd.clusters
            Wc = W[:, cluster]
            Λc = vbd.block_diagonal[cluster, cluster]
            res = Acx * Wc - (Bcx * Wc) * Λc
            invariance_defect = max(invariance_defect, upper_bound_L2_opnorm(res))
        end
    end

    return RigorousSpectralProjectorsResult(projectors, vbd.clusters,
        vbd.cluster_intervals, idempotency_defect, orthogonality_defect,
        resolution_defect, invariance_defect, A, vbd)
end

"""
    GEVInvariantSubspace

One certified deflating (invariant) subspace of the pencil `Ax = λBx`, returned
by [`gev_invariant_subspaces`](@ref).

# Fields
- `indices`: the cluster's column range in the certified basis
- `basis`: `W[:, indices]` — the columns of the VBD basis that **span** the
  subspace. This is the subspace itself; the projector below is derived from it.
- `block`: the certified block `Λ_CC` (a `BallMatrix`); its spectrum is the
  cluster's eigenvalues, so `dim = length(indices)` counts them with multiplicity
- `discs`: the β-inflated Gershgorin discs of the cluster
- `residual`: rigorous bound on `‖A·basis − B·basis·block‖₂`, the deflating-
  subspace residual whose smallness certifies the subspace
- `projector`: the spectral projector `W[:, C]·(W⁻¹)[C, :]` onto it, or `nothing`
  when `gev_invariant_subspaces` was called with `projectors = false`
"""
struct GEVInvariantSubspace{MT, BT, IT, RT, PT}
    indices::UnitRange{Int}
    basis::MT
    block::BT
    discs::Vector{IT}
    residual::RT
    """Spectral projector onto the subspace, or `nothing` when `projectors = false`."""
    projector::PT
end

Base.size(s::GEVInvariantSubspace) = size(s.basis)
dim(s::GEVInvariantSubspace) = length(s.indices)

"""
    gev_invariant_subspaces(A::BallMatrix, B::BallMatrix; projectors = true)

Enclose **all** eigenvalues and deflating (invariant) subspaces of the pencil
`Ax = λBx`, one entry per certified cluster.

The subspaces are produced by the certified basis `W` of
[`schur_newton_vbd`](@ref)`(A, B)`: cluster `C` spans `W[:, C]`. Since
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

Pass `projectors = false` to skip forming the `n×n` projectors when only the
subspace bases and blocks are wanted.

# Example
```julia
A = BallMatrix(randn(6, 6));  B = BallMatrix(Matrix(randn(6, 6)' * randn(6, 6) + 6I))
subs = gev_invariant_subspaces(A, B)
subs[1].basis        # spans the first deflating subspace
subs[1].residual     # rigorous ‖A·W_C − B·W_C·Λ_C‖₂
```
"""
function gev_invariant_subspaces(A::BallMatrix{T, NT}, B::BallMatrix;
        projectors::Bool = true) where {T, NT}
    size(A) == size(B) ||
        throw(DimensionMismatch("A and B must have the same size"))

    vbd = schur_newton_vbd(A, B)
    n = size(A, 1)
    W = BallMatrix(vbd.basis)
    CTm = eltype(mid(vbd.transformed))
    Acx = BallMatrix(Matrix{CTm}(mid(A)), rad(A))
    Bcx = BallMatrix(Matrix{CTm}(mid(B)), rad(B))
    Winv = projectors ? BallMatrix(inv(vbd.basis)) : nothing

    out = GEVInvariantSubspace[]
    for cluster in vbd.clusters
        Wc = W[:, cluster]
        Λc = vbd.block_diagonal[cluster, cluster]
        # deflating-subspace residual: A·W_C − B·W_C·Λ_CC
        residual = upper_bound_L2_opnorm(Acx * Wc - (Bcx * Wc) * Λc)
        P = projectors ? Wc * Winv[cluster, :] : nothing
        push!(out,
            GEVInvariantSubspace(cluster, Wc, Λc, vbd.cluster_intervals[cluster],
                residual, P))
    end
    return identity.(out)
end
