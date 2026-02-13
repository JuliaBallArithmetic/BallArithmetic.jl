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
Base.iterate(result::RigorousSpectralProjectorsResult, state) = iterate(result.projectors, state)

"""
    miyajima_spectral_projectors(A::BallMatrix; hermitian=false, verify_invariance=true)

Compute rigorous enclosures for spectral projectors corresponding to each
eigenvalue cluster identified by Miyajima's verified block diagonalization (VBD).

The method follows the approach from Ref. [MiyajimaInvariantSubspaces2014](@cite):
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

* [MiyajimaInvariantSubspaces2014](@cite) Miyajima, SIAM J. Matrix Anal. Appl. 35, 1205–1225 (2014)
"""
function miyajima_spectral_projectors(A::BallMatrix{T, NT};
                                       hermitian::Bool = false,
                                       verify_invariance::Bool = true,
                                       vbd_method::Symbol = :nsd) where {T, NT}
    vbd_method ∈ [:nsd, :njd] ||
        throw(ArgumentError("vbd_method must be :nsd or :njd"))

    # Step 1: Compute VBD
    vbd = if vbd_method == :njd
        miyajima_vbd_njd(A)
    else
        miyajima_vbd(A; hermitian = hermitian)
    end

    # Step 2: Extract basis and its inverse/adjoint
    V = BallMatrix(vbd.basis)
    n = size(A, 1)
    num_clusters = length(vbd.clusters)

    # For unitary bases (NSD), P_k = V_k * V_k^† is correct.
    # For non-unitary bases (NJD), P_k = W_k * (W⁻¹)_k where
    # W_k = W[:, cluster_k] and (W⁻¹)_k = (W⁻¹)[cluster_k, :].
    is_unitary_basis = !(vbd isa NJDVBDResult)

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
        for j in (i+1):num_clusters
            product = projectors[i] * projectors[j]
            defect = upper_bound_L2_opnorm(product)
            orthogonality_defect = max(orthogonality_defect, defect)
        end
    end

    # Verify resolution of identity: ∑ P_k ≈ I
    sum_projectors = sum(projectors)
    # Match identity element type to projector element type (NJD may produce complex)
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
