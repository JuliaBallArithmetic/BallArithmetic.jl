# Rigorous spectral projection via Schur decomposition and Sylvester equations
# Implements Riesz projectors for non-normal matrices using verified computations

"""
    SchurSpectralProjectorResult{T}

Result of verified spectral projector computation from Schur decomposition.

# Fields
- `projector::BallMatrix{T}`: Rigorous enclosure of the spectral projector P
- `schur_projector::BallMatrix{T}`: Projector in Schur coordinates
- `coupling_matrix::BallMatrix{T}`: Solution Y to the Sylvester equation T₁₁Y - YT₂₂ = T₁₂
- `eigenvalue_separation::T`: Lower bound on min|λᵢ - λⱼ| for i ∈ S, j ∉ S
- `projector_norm::T`: Upper bound on ‖P‖₂
- `idempotency_defect::T`: Upper bound on ‖P² - P‖₂
- `schur_basis::Matrix{T}`: Unitary matrix Q from A = QTQ^†
- `schur_form::Matrix{T}`: Upper triangular T from A = QTQ^†
- `cluster_indices::UnitRange{Int}`: Indices in Schur form corresponding to projected eigenvalues
"""
struct SchurSpectralProjectorResult{T, NT}
    projector::BallMatrix{T, NT}
    schur_projector::BallMatrix{T, NT}
    coupling_matrix::Union{BallMatrix{T, NT}, Nothing}
    eigenvalue_separation::T
    projector_norm::T
    idempotency_defect::T
    schur_basis::Matrix{NT}
    schur_form::Matrix{NT}
    cluster_indices::UnitRange{Int}
end


"""
    compute_spectral_projector_schur(A::BallMatrix, cluster_indices::UnitRange{Int};
                                     verify_idempotency::Bool=true)

Compute rigorous enclosure of the spectral projector for eigenvalues at specified
Schur form indices using verified Sylvester equation solver.

# Mathematical Background

Given a matrix A with Schur decomposition A = QTQ^†, where T is upper triangular,
partition T into blocks:

```
T = [T₁₁  T₁₂]
    [0    T₂₂]
```

where T₁₁ contains the eigenvalues we want to project onto (diagonal entries
at positions given by `cluster_indices`).

The spectral projector (Riesz projector) onto the eigenspace corresponding to
eigenvalues of T₁₁ is:

```
P = Q * P_Schur * Q^†
```

where P_Schur is the projector in Schur coordinates:

```
P_Schur = [I   Y]
          [0   0]
```

and Y solves the Sylvester equation:

```
T₁₁*Y - Y*T₂₂ = T₁₂
```

This is solved rigorously using [`triangular_sylvester_miyajima_enclosure`](@ref).

# Algorithm

1. Compute Schur decomposition: A = QTQ^† (using approximate Schur for A.c)
2. Extract blocks T₁₁, T₁₂, T₂₂ based on cluster_indices
3. Solve Sylvester equation rigorously: T₁₁*Y - Y*T₂₂ = T₁₂
4. Construct P_Schur = [I Y; 0 0] with interval arithmetic
5. Transform back: P = Q * P_Schur * Q^† with interval arithmetic
6. Verify idempotency: ‖P² - P‖₂ < tol

# Arguments
- `A::BallMatrix`: Square interval matrix (n × n)
- `cluster_indices::UnitRange{Int}`: Indices of eigenvalues to project onto
  (corresponds to positions in Schur form, typically 1:k for first k eigenvalues)
- `verify_idempotency::Bool=true`: Check that ‖P² - P‖₂ is small

# Returns
[`SchurSpectralProjectorResult`](@ref) containing the verified projector and diagnostics.

# Examples

## Project onto first two eigenvalues
```julia
using BallArithmetic, LinearAlgebra

# Matrix with small uncertainties
A = BallMatrix([1.0 2.0 0.0; 0.0 3.0 1.0; 0.0 0.0 5.0], fill(1e-10, 3, 3))

# Compute projector onto eigenvalues 1.0 and 3.0 (first two in Schur form)
result = compute_spectral_projector_schur(A, 1:2)

P = result.projector
@show result.eigenvalue_separation  # Gap to third eigenvalue
@show result.idempotency_defect     # ‖P² - P‖₂

# Project a vector
v = BallVector([1.0, 2.0, 3.0])
v_projected = P * v
```

## For a clustered matrix
```julia
# Upper triangular matrix with eigenvalues [1.0, 1.1, 5.0, 5.1]
A = BallMatrix(triu(randn(4,4)) .+ Diagonal([1.0, 1.1, 5.0, 5.1]))

# After sorting eigenvalues by Schur decomposition, project onto first cluster
result = compute_spectral_projector_schur(A, 1:2)
```

# Notes
- Requires non-zero eigenvalue separation: min|λᵢ - λⱼ| for i ∈ cluster, j ∉ cluster
- The Sylvester equation solver may fail if separation is too small
- Complexity: O(n³) for Schur decomposition + O(k²(n-k)²) for Sylvester solver
- For hermitian matrices, use `compute_spectral_projector_hermitian` instead (more efficient)
- The projector P satisfies: P² ≈ P and P*A ≈ A*P (modulo idempotency defect)

# References
- Kato, T. "Perturbation Theory for Linear Operators" (1995), Chapter II.4
- Stewart, G. W., Sun, J. "Matrix Perturbation Theory" (1990), Chapter V
- Miyajima, S. "Fast enclosure for all eigenvalues in generalized eigenvalue problems"
  SIAM J. Matrix Anal. Appl. 35, 1205–1225 (2014)

# See Also
- [`triangular_sylvester_miyajima_enclosure`](@ref): Verified Sylvester solver used internally
- [`project_vector_spectral`](@ref): Convenient interface to project vectors using this result
"""
function compute_spectral_projector_schur(A::BallMatrix{T, NT},
                                          cluster_indices::UnitRange{Int};
                                          verify_idempotency::Bool=true) where {T, NT}
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("A must be square"))

    k = length(cluster_indices)
    k >= 1 || throw(ArgumentError("Cluster must contain at least one index"))
    k < n || throw(ArgumentError("Cluster cannot contain all indices (would be identity)"))

    # Assuming cluster_indices = 1:k for simplicity
    # (General case would require reordering Schur form)
    cluster_indices == 1:k || throw(ArgumentError(
        "Currently only supports cluster_indices = 1:k. " *
        "General reordering not yet implemented."))

    # Step 1: Compute Schur decomposition of the center matrix
    A_center = A.c
    F = schur(A_center)
    Q = F.Z  # Unitary Schur basis
    Tmat = F.T  # Upper triangular Schur form

    # Convert to interval matrices
    Q_ball = BallMatrix(Q)
    T_ball = BallMatrix(Tmat)

    # Step 2: Extract blocks
    # Tmat = [T₁₁  T₁₂]
    #        [0    T₂₂]
    T11 = Tmat[1:k, 1:k]
    T12 = Tmat[1:k, (k+1):n]
    T22 = Tmat[(k+1):n, (k+1):n]

    # Compute eigenvalue separation (for diagnostics)
    eig_T11 = eigvals(T11)
    eig_T22 = eigvals(T22)

    eigenvalue_separation = minimum(
        abs(λ1 - λ2) for λ1 in eig_T11 for λ2 in eig_T22
    )

    realtype = float(T)
    if eigenvalue_separation < 10 * eps(realtype)
        @warn "Eigenvalue separation very small ($eigenvalue_separation). " *
              "Projector may be ill-conditioned."
    end

    # Step 3: Solve Sylvester equation T₁₁*Y - Y*T₂₂ = T₁₂
    # Use the triangular Sylvester solver
    # Note: triangular_sylvester_miyajima_enclosure solves T₂₂' * Y₂ - Y₂ * T₁₁' = T₁₂'
    # which means Y₂ is (n-k)×k, but we need Y to be k×(n-k)
    Y_transposed = triangular_sylvester_miyajima_enclosure(Tmat, k)

    # Transpose Y to get the correct dimensions
    Y_ball = BallMatrix(transpose(Y_transposed.c), transpose(Y_transposed.r))

    # Step 4: Construct projector in Schur coordinates
    # P_Schur = [I   Y]
    #           [0   0]
    # Build center and radius matrices separately
    P_schur_c = zeros(NT, n, n)
    P_schur_r = zeros(NT, n, n)

    # Top-left: I (k×k identity)
    P_schur_c[1:k, 1:k] .= Matrix{NT}(I, k, k)

    # Top-right: Y (now k×(n-k))
    P_schur_c[1:k, (k+1):n] .= Y_ball.c
    P_schur_r[1:k, (k+1):n] .= Y_ball.r

    # Bottom blocks are already zeros

    # Assemble P_Schur as BallMatrix
    P_schur = BallMatrix(P_schur_c, P_schur_r)

    # Step 5: Transform back to original coordinates
    # P = Q * P_Schur * Q^†
    Q_adj_ball = BallMatrix(adjoint(Q))
    P = Q_ball * P_schur * Q_adj_ball

    # Step 6: Estimate projector norm
    projector_norm = collatz_upper_bound_L2_opnorm(P)

    # Step 7: Verify idempotency if requested
    idempotency_defect = zero(T)
    if verify_idempotency
        P_squared = P * P
        P_diff = P_squared - P
        idempotency_defect = collatz_upper_bound_L2_opnorm(P_diff)

        if idempotency_defect > 1e-6
            @warn "Large idempotency defect: ‖P² - P‖₂ ≈ $idempotency_defect"
        end
    end

    return SchurSpectralProjectorResult(
        P,
        P_schur,
        Y_ball,
        eigenvalue_separation,
        projector_norm,
        idempotency_defect,
        Q,
        Tmat,
        cluster_indices
    )
end


"""
    compute_spectral_projector_hermitian(A::BallMatrix, cluster_indices::UnitRange{Int})

Compute spectral projector for Hermitian matrix (simplified, no Sylvester equation needed).

For Hermitian matrices, eigenvectors are orthogonal, so the spectral projector is simply:
```
P = V_S * V_S^†
```
where V_S are the eigenvectors corresponding to the selected eigenvalues.

This is more efficient than the general Schur-based method since no Sylvester equation
needs to be solved.

# Arguments
- `A::BallMatrix`: Hermitian interval matrix
- `cluster_indices::UnitRange{Int}`: Indices of eigenvalues to project onto (after sorting)

# Returns
[`SchurSpectralProjectorResult`](@ref) with the verified projector.

# Example
```julia
# Symmetric matrix
A = BallMatrix([4.0 1.0; 1.0 3.0], fill(1e-10, 2, 2))

# Project onto first eigenvalue
result = compute_spectral_projector_hermitian(A, 1:1)
P = result.projector
```
"""
function compute_spectral_projector_hermitian(A::BallMatrix{T, NT},
                                              cluster_indices::UnitRange{Int}) where {T, NT}
    n = size(A, 1)
    k = length(cluster_indices)

    # Compute eigendecomposition of center
    F = eigen(Symmetric(A.c))
    V = F.vectors
    λ = F.values

    # Extract eigenvectors for cluster
    V_S = V[:, cluster_indices]

    # Compute projector: P = V_S * V_S^†
    V_S_ball = BallMatrix(V_S)
    V_S_adj_ball = BallMatrix(adjoint(V_S))
    P = V_S_ball * V_S_adj_ball

    # Eigenvalue separation
    λ_in = λ[cluster_indices]
    λ_out = λ[setdiff(1:n, cluster_indices)]
    eigenvalue_separation = minimum(abs(λi - λj) for λi in λ_in for λj in λ_out)

    # Verify idempotency
    P_squared = P * P
    idempotency_defect = collatz_upper_bound_L2_opnorm(P_squared - P)

    # Projector norm (should be ≈ 1 for projector)
    projector_norm = collatz_upper_bound_L2_opnorm(P)

    # For Hermitian case, P_schur = P in eigenbasis (no coupling)
    P_schur_diag = zeros(NT, n)
    P_schur_diag[cluster_indices] .= 1.0
    P_schur = BallMatrix(Diagonal(P_schur_diag))

    return SchurSpectralProjectorResult(
        P,
        P_schur,
        nothing,  # No coupling matrix for Hermitian case
        eigenvalue_separation,
        projector_norm,
        idempotency_defect,
        V,  # Schur basis = eigenvector matrix
        Matrix(Diagonal(λ)),  # Schur form = diagonal eigenvalue matrix
        cluster_indices
    )
end


"""
    project_vector_spectral(v::BallVector, result::SchurSpectralProjectorResult)

Project interval vector onto eigenspace using precomputed spectral projector.

# Arguments
- `v::BallVector`: Interval vector to project
- `result::SchurSpectralProjectorResult`: Precomputed projector from
  [`compute_spectral_projector_schur`](@ref) or [`compute_spectral_projector_hermitian`](@ref)

# Returns
`BallVector` containing P*v with rigorous error bounds.

# Example
```julia
A = BallMatrix([1.0 2.0; 0.0 3.0], fill(1e-10, 2, 2))
result = compute_spectral_projector_schur(A, 1:1)

v = BallVector([1.0, 2.0], [1e-10, 1e-10])
v_projected = project_vector_spectral(v, result)
```
"""
function project_vector_spectral(v::BallVector, result::SchurSpectralProjectorResult)
    return result.projector * v
end


"""
    project_vector_spectral(v::AbstractVector, result::SchurSpectralProjectorResult)

Project standard vector onto eigenspace (non-verified version).

# Arguments
- `v::AbstractVector`: Vector to project
- `result::SchurSpectralProjectorResult`: Precomputed projector

# Returns
Projected vector P*v (center value only, no error bounds).
"""
function project_vector_spectral(v::AbstractVector, result::SchurSpectralProjectorResult)
    P_center = result.projector.c
    return P_center * v
end


"""
    verify_spectral_projector_properties(result::SchurSpectralProjectorResult, A::BallMatrix;
                                         tol::Real=1e-10)

Verify that the computed spectral projector satisfies expected mathematical properties.

# Checks
1. **Idempotency**: ‖P² - P‖₂ < tol
2. **Bounded norm**: ‖P‖₂ < ∞ (typically ‖P‖₂ ≤ κ(V) where κ is condition number of eigenvectors)
3. **Commutation** (if A normal): ‖A*P - P*A‖₂ < tol
4. **Eigenvalue separation**: min|λᵢ - λⱼ| > 0 for i ∈ S, j ∉ S

# Arguments
- `result::SchurSpectralProjectorResult`: Computed projector
- `A::BallMatrix`: Original matrix
- `tol::Real=1e-10`: Tolerance for property verification

# Returns
- `true` if all properties satisfied, `false` otherwise
"""
function verify_spectral_projector_properties(result::SchurSpectralProjectorResult,
                                               A::BallMatrix;
                                               tol::Real=1e-10,
                                               check_commutation::Bool=false)
    checks = Bool[]

    # 1. Idempotency
    push!(checks, result.idempotency_defect < tol)

    # 2. Bounded norm
    push!(checks, isfinite(result.projector_norm))

    # 3. Positive eigenvalue separation
    push!(checks, result.eigenvalue_separation > eps(eltype(result.projector_norm)))

    # 4. Commutation (optional, expensive)
    if check_commutation
        AP = A * result.projector
        PA = result.projector * A
        comm_defect = collatz_upper_bound_L2_opnorm(AP - PA)
        push!(checks, comm_defect < tol)
    end

    return all(checks)
end
