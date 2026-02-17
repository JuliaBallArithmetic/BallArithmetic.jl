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
- **Caveat**: The Schur decomposition of the midpoint matrix is not itself rigorously
  enclosed (Q and T carry zero radii). The Sylvester coupling Y is rigorously bounded,
  but the final projector P = Q·P_Schur·Q† does not account for the Schur approximation
  error. The idempotency and invariance checks verify self-consistency of the result,
  but do not guarantee that the true spectral projector lies within the computed ball.
  For a fully rigorous enclosure, use [`miyajima_spectral_projectors`](@ref) instead.

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
                                          verify_idempotency::Bool=true,
                                          schur_data=nothing,
                                          ordered_schur_data=nothing) where {T, NT}
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("A must be square"))

    k = length(cluster_indices)
    k >= 1 || throw(ArgumentError("Cluster must contain at least one index"))
    k < n || throw(ArgumentError("Cluster cannot contain all indices (would be identity)"))

    # Priority: ordered_schur_data > schur_data > compute from A
    if ordered_schur_data !== nothing
        Q_ord, T_ord, k_ord = ordered_schur_data
        k_ord == k || throw(ArgumentError("ordered_schur_data k=$k_ord does not match cluster size $k"))
        return _spectral_projector_from_ordered_schur(Q_ord, T_ord, k, T, NT, n,
                                                       verify_idempotency)
    end

    # Step 1: Compute Schur decomposition of the center matrix (or use provided)
    if schur_data !== nothing
        Q, Tmat = schur_data
    else
        A_center = A.c
        F = schur(A_center)
        Q = F.Z  # Unitary Schur basis
        Tmat = F.T  # Upper triangular Schur form
    end

    # Step 1b: If cluster_indices ≠ 1:k, reorder via ordschur
    if cluster_indices != 1:k
        select = falses(n)
        select[cluster_indices] .= true
        F_schur = Schur(Matrix(Tmat), Matrix(Q), diag(Tmat))
        F_ord = ordschur(F_schur, BitVector(select))
        Q = F_ord.Z
        Tmat = F_ord.T
    end

    # From here on, the selected eigenvalues are at positions 1:k
    _spectral_projector_from_ordered_schur(Q, Tmat, k, T, NT, n,
                                           verify_idempotency)
end

"""
    compute_spectral_projector_schur(A::BallMatrix, target_indices::AbstractVector{Int};
                                     verify_idempotency::Bool=true,
                                     schur_data=nothing)

Compute rigorous spectral projector for eigenvalues at arbitrary Schur form positions.

Unlike the `UnitRange` method, this accepts any subset of eigenvalue indices. Internally,
the Schur form is reordered (via `ordschur`) so the target eigenvalues occupy positions
`1:k`, then the standard Sylvester-based projector pipeline is applied.

# Arguments
- `A::BallMatrix`: Square interval matrix (n × n)
- `target_indices::AbstractVector{Int}`: Positions of eigenvalues in Schur form to project onto
- `verify_idempotency::Bool=true`: Check ‖P² - P‖₂
- `schur_data=nothing`: Optional pre-computed `(Q, T)` Schur factors to skip internal Schur

# Returns
[`SchurSpectralProjectorResult`](@ref) with the reordered Schur factors and verified projector.

# Example
```julia
A = BallMatrix(randn(5, 5))
# Project onto 2nd and 4th eigenvalues (arbitrary positions)
result = compute_spectral_projector_schur(A, [2, 4])
```
"""
function compute_spectral_projector_schur(A::BallMatrix{T, NT},
                                          target_indices::AbstractVector{Int};
                                          verify_idempotency::Bool=true,
                                          schur_data=nothing,
                                          ordered_schur_data=nothing) where {T, NT}
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("A must be square"))

    k = length(target_indices)
    k >= 1 || throw(ArgumentError("Must target at least one eigenvalue"))
    k < n || throw(ArgumentError("Cannot target all eigenvalues (would be identity)"))
    all(1 .<= target_indices .<= n) || throw(ArgumentError("Indices must be in 1:$n"))

    # Priority: ordered_schur_data > schur_data > compute from A
    if ordered_schur_data !== nothing
        Q_ord, T_ord, k_ord = ordered_schur_data
        k_ord == k || throw(ArgumentError("ordered_schur_data k=$k_ord does not match cluster size $k"))
        return _spectral_projector_from_ordered_schur(Q_ord, T_ord, k, T, NT, n,
                                                       verify_idempotency)
    end

    # Step 1: Compute Schur decomposition (or use provided)
    if schur_data !== nothing
        Q, Tmat = schur_data
    else
        A_center = A.c
        F = schur(A_center)
        Q = F.Z
        Tmat = F.T
    end

    # Step 2: Reorder so target eigenvalues are at positions 1:k
    select = falses(n)
    select[target_indices] .= true
    F_schur = Schur(Matrix(Tmat), Matrix(Q), diag(Tmat))
    F_ord = ordschur(F_schur, BitVector(select))
    Q = F_ord.Z
    Tmat = F_ord.T

    _spectral_projector_from_ordered_schur(Q, Tmat, k, T, NT, n,
                                           verify_idempotency)
end

# Internal helper: given Schur factors (Q, Tmat) where the target eigenvalues
# are already at positions 1:k, compute the spectral projector via Sylvester.
function _spectral_projector_from_ordered_schur(Q::AbstractMatrix, Tmat::AbstractMatrix,
                                                k::Int, ::Type{T}, ::Type{NT}, n::Int,
                                                verify_idempotency::Bool) where {T, NT}
    # Convert to interval matrices
    Q_ball = BallMatrix(Q)
    T_ball = BallMatrix(Tmat)

    # Extract blocks for eigenvalue separation diagnostic
    T11 = Tmat[1:k, 1:k]
    T22 = Tmat[(k+1):n, (k+1):n]

    eig_T11 = diag(T11)
    eig_T22 = diag(T22)

    eigenvalue_separation = minimum(
        abs(λ1 - λ2) for λ1 in eig_T11 for λ2 in eig_T22
    )

    realtype = float(T)
    if eigenvalue_separation < 10 * eps(realtype)
        @warn "Eigenvalue separation very small ($eigenvalue_separation). " *
              "Projector may be ill-conditioned."
    end

    # Solve Sylvester equation T₁₁*Y - Y*T₂₂ = T₁₂
    # triangular_sylvester_miyajima_enclosure solves T₂₂^H X - X T₁₁^H = T₁₂^H
    # Taking adjoint: T₁₁ (-X^H) - (-X^H) T₂₂ = T₁₂, so Y = -X^H = -adjoint(X)
    Y_transposed = triangular_sylvester_miyajima_enclosure(Tmat, k)
    Y_ball = BallMatrix(-adjoint(Y_transposed.c), transpose(Y_transposed.r))

    # Construct projector in Schur coordinates: P_Schur = [I Y; 0 0]
    P_schur_c = zeros(NT, n, n)
    P_schur_r = zeros(realtype, n, n)
    P_schur_c[1:k, 1:k] .= Matrix{NT}(I, k, k)
    P_schur_c[1:k, (k+1):n] .= Y_ball.c
    P_schur_r[1:k, (k+1):n] .= Y_ball.r
    P_schur = BallMatrix(P_schur_c, P_schur_r)

    # Transform back: P = Q * P_Schur * Q^†
    Q_adj_ball = BallMatrix(adjoint(Q))
    P = Q_ball * P_schur * Q_adj_ball

    # Estimate projector norm
    projector_norm = upper_bound_L2_opnorm(P)

    # Verify idempotency
    idempotency_defect = zero(T)
    if verify_idempotency
        P_diff = P * P - P
        idempotency_defect = upper_bound_L2_opnorm(P_diff)
        if idempotency_defect > 1e-6
            @warn "Large idempotency defect: ‖P² - P‖₂ ≈ $idempotency_defect"
        end
    end

    return SchurSpectralProjectorResult(
        P, P_schur, Y_ball, eigenvalue_separation, projector_norm,
        idempotency_defect, Matrix(Q), Matrix(Tmat), 1:k
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
    idempotency_defect = upper_bound_L2_opnorm(P_squared - P)

    # Projector norm (should be ≈ 1 for projector)
    projector_norm = upper_bound_L2_opnorm(P)

    # For Hermitian case, P_schur = P in eigenbasis (no coupling)
    P_schur_diag = zeros(NT, n)
    P_schur_diag[cluster_indices] .= one(NT)
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
        comm_defect = upper_bound_L2_opnorm(AP - PA)
        push!(checks, comm_defect < tol)
    end

    return all(checks)
end


"""
    SpectralCoefficientResult{T, NT}

Result of computing spectral coefficients without forming the full projector.

# Fields
- `coefficients::BallVector{T, NT}`: k-vector of spectral coefficients `[I_k  Y] * Q^H v`
- `left_eigenvector_schur::BallMatrix{T, NT}`: The `[I_k  Y]` block in Schur basis (k × n)
- `coupling_matrix::Union{BallMatrix{T, NT}, Nothing}`: Solution Y of Sylvester equation
- `eigenvalue_separation::T`: Lower bound on min|λᵢ - λⱼ| for i ∈ cluster, j ∉ cluster
- `schur_basis::Matrix{NT}`: Unitary matrix Q from A = QTQ^†
- `schur_form::Matrix{NT}`: Upper triangular T from A = QTQ^†
"""
struct SpectralCoefficientResult{T, NT}
    coefficients::BallVector{T, NT}
    left_eigenvector_schur::BallMatrix{T, NT}
    coupling_matrix::Union{BallMatrix{T, NT}, Nothing}
    eigenvalue_separation::T
    schur_basis::Matrix{NT}
    schur_form::Matrix{NT}
end

"""
    compute_spectral_coefficient(A::BallMatrix, v::AbstractVector,
                                  cluster_indices;
                                  schur_data=nothing,
                                  ordered_schur_data=nothing)

Compute spectral coefficients `P * v` restricted to the cluster subspace, without
forming the full n × n projector. This is O(n²) per eigenvalue instead of O(n³).

# Algorithm
Given ordered Schur `A = Q T Q^H` with target eigenvalues at positions 1:k:
1. Solve the Sylvester equation for the k × (n-k) coupling matrix Y
2. Compute `q = Q^H v` (matrix-vector multiply, O(n²))
3. Return `coefficients = q[1:k] + Y * q[(k+1):n]` (O(nk))

# Arguments
- `A::BallMatrix`: Square interval matrix (n × n)
- `v::AbstractVector`: Vector to project (length n), can be `BallVector` or plain vector
- `cluster_indices`: Eigenvalue indices — `UnitRange{Int}` or `AbstractVector{Int}`
- `schur_data=nothing`: Optional `(Q, T)` Schur factors
- `ordered_schur_data=nothing`: Optional `(Q_ord, T_ord, k)` — pre-ordered, skips ordschur

# Returns
[`SpectralCoefficientResult`](@ref) with the k-vector of coefficients and diagnostics.
"""
function compute_spectral_coefficient(A::BallMatrix{T, NT},
                                       v::AbstractVector,
                                       cluster_indices;
                                       schur_data=nothing,
                                       ordered_schur_data=nothing) where {T, NT}
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("A must be square"))
    length(v) == n || throw(DimensionMismatch("v must have length $n"))

    k = length(cluster_indices)
    k >= 1 || throw(ArgumentError("Cluster must contain at least one index"))
    k < n || throw(ArgumentError("Cluster cannot contain all indices"))

    # Get ordered Schur data
    if ordered_schur_data !== nothing
        Q, Tmat, k_ord = ordered_schur_data
        k_ord == k || throw(ArgumentError("ordered_schur_data k=$k_ord does not match cluster size $k"))
    else
        # Compute or use provided Schur decomposition
        if schur_data !== nothing
            Q, Tmat = schur_data
        else
            A_center = A.c
            F = schur(A_center)
            Q = F.Z
            Tmat = F.T
        end

        # Reorder if needed
        if !(cluster_indices isa UnitRange && cluster_indices == 1:k)
            select = falses(n)
            if cluster_indices isa UnitRange
                select[cluster_indices] .= true
            else
                for idx in cluster_indices
                    select[idx] = true
                end
            end
            F_schur = Schur(Matrix(Tmat), Matrix(Q), diag(Tmat))
            F_ord = ordschur(F_schur, BitVector(select))
            Q = F_ord.Z
            Tmat = F_ord.T
        end
    end

    # Eigenvalue separation
    T11 = Tmat[1:k, 1:k]
    T22 = Tmat[(k+1):n, (k+1):n]
    eig_T11 = diag(T11)
    eig_T22 = diag(T22)

    eigenvalue_separation = minimum(
        abs(λ1 - λ2) for λ1 in eig_T11 for λ2 in eig_T22
    )

    realtype = float(T)
    if eigenvalue_separation < 10 * eps(realtype)
        @warn "Eigenvalue separation very small ($eigenvalue_separation). " *
              "Coefficients may be ill-conditioned."
    end

    # Solve Sylvester equation for Y (k × (n-k))
    Y_transposed = triangular_sylvester_miyajima_enclosure(Tmat, k)
    Y_ball = BallMatrix(-adjoint(Y_transposed.c), transpose(Y_transposed.r))

    # Construct [I_k  Y] block (k × n) in Schur basis
    IY_c = zeros(NT, k, n)
    IY_r = zeros(realtype, k, n)
    IY_c[1:k, 1:k] .= Matrix{NT}(I, k, k)
    IY_c[1:k, (k+1):n] .= Y_ball.c
    IY_r[1:k, (k+1):n] .= Y_ball.r
    IY_ball = BallMatrix(IY_c, IY_r)

    # Compute q = Q^H * v (O(n^2) matrix-vector product)
    Q_adj = adjoint(Q)
    if v isa BallVector
        q = BallMatrix(Q_adj) * v
    else
        q_mid = Q_adj * v
        q = BallVector(q_mid)
    end

    # Compute coefficients = [I_k  Y] * q = q[1:k] + Y * q[(k+1):n] (O(nk))
    coefficients = IY_ball * q

    return SpectralCoefficientResult(
        coefficients, IY_ball, Y_ball,
        eigenvalue_separation,
        Matrix(Q), Matrix(Tmat)
    )
end
