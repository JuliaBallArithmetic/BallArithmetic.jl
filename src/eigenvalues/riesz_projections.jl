# Riesz projection interfaces for eigenspaces and invariant subspaces
# Provides simplified interfaces for projecting vectors onto eigenspaces
# after eigenvalue or Schur decomposition.

"""
    project_onto_eigenspace(v::AbstractVector, V::AbstractMatrix, indices::UnitRange{Int};
                            hermitian::Bool=false, left_eigenvectors::Union{Nothing, AbstractMatrix}=nothing)

Project vector `v` onto the eigenspace spanned by eigenvectors at the specified indices.

# Mathematical Background

For a diagonalizable matrix A = VΛV⁻¹, the Riesz projection onto the eigenspace
corresponding to eigenvalues λ_k (k ∈ indices) is:

**Hermitian/Normal case** (hermitian=true or V orthogonal):
```
P_S v = V_S (V_S^† V_S)⁻¹ V_S^† v = V_S V_S^† v
```
where V_S = V[:, indices] and V^† is the conjugate transpose.

**Non-normal case** (requires left eigenvectors):
```
P_S v = V_S W_S^† v
```
where W_S contains the corresponding left eigenvectors (rows of V⁻¹).

# Arguments
- `v::AbstractVector`: Vector to project (length n)
- `V::AbstractMatrix`: Matrix of right eigenvectors (n × n)
- `indices::UnitRange{Int}`: Column indices of eigenvectors to project onto
- `hermitian::Bool=false`: If true, assumes V is orthogonal (V^† V = I)
- `left_eigenvectors::Union{Nothing, AbstractMatrix}=nothing`: Left eigenvectors W = V⁻¹
  (required for non-normal matrices when hermitian=false)

# Returns
Projected vector P_S v of length n.

# Examples

## Symmetric/Hermitian case
```julia
using LinearAlgebra, BallArithmetic

# Symmetric matrix with clustered eigenvalues
A = [4.0 1.0 0.0; 1.0 4.0 0.0; 0.0 0.0 10.0]
F = eigen(Symmetric(A))

# Project onto first two eigenvectors (eigenvalues ≈ 3, 5)
v = [1.0, 2.0, 3.0]
v_proj = project_onto_eigenspace(v, F.vectors, 1:2; hermitian=true)
```

## Non-normal case (requires left eigenvectors)
```julia
# Non-normal matrix
A = [1.0 1.0; 0.0 2.0]  # Upper triangular, non-symmetric
F = eigen(A)

# Compute left eigenvectors
V_inv = inv(F.vectors)  # Rows are left eigenvectors

# Project onto first eigenspace
v = [1.0, 1.0]
v_proj = project_onto_eigenspace(v, F.vectors, 1:1;
                                  hermitian=false, left_eigenvectors=V_inv)
```

# Notes
- For hermitian=true, uses the simple formula P = V_S V_S^†
- For hermitian=false without left eigenvectors, attempts V⁻¹ computation (may fail)
- For non-normal matrices, providing left_eigenvectors avoids computing V⁻¹
- The projection is exact in exact arithmetic but subject to rounding errors
- The result is a floating-point approximation with no error bound

# References
- Kato, T. "Perturbation theory for linear operators" (1995), Chapter II
- Stewart, G. W., Sun, J. "Matrix Perturbation Theory" (1990), Section V.3
"""
function project_onto_eigenspace(v::AbstractVector,
                                  V::AbstractMatrix,
                                  indices::UnitRange{Int};
                                  hermitian::Bool=false,
                                  left_eigenvectors::Union{Nothing, AbstractMatrix}=nothing)
    n = length(v)
    size(V, 1) == n || throw(DimensionMismatch("Eigenvector matrix must have same row dimension as vector"))
    maximum(indices) <= size(V, 2) || throw(ArgumentError("Indices out of bounds"))

    # Extract relevant eigenvectors
    V_S = V[:, indices]

    if hermitian
        # Hermitian/normal case: P_S v = V_S (V_S^† V_S)^{-1} V_S^† v
        # If orthonormal, this simplifies to V_S V_S^† v
        G = V_S' * V_S  # Gram matrix
        coeffs = G \ (V_S' * v)  # Solve for coefficients
        return V_S * coeffs
    else
        # Non-normal case: need left eigenvectors
        if left_eigenvectors === nothing
            # Attempt to compute V^{-1} (may fail if ill-conditioned)
            @warn "Non-normal projection without left eigenvectors: computing V^{-1} (may be unstable)"
            local V_inv
            try
                V_inv = inv(V)
            catch e
                error("Failed to compute eigenvector inverse for non-normal projection. " *
                      "Provide left_eigenvectors explicitly or use hermitian=true if applicable.")
            end
            W_S = V_inv[indices, :]  # Extract rows corresponding to eigenspace
        else
            size(left_eigenvectors) == size(V') || throw(DimensionMismatch("Left eigenvectors must be n × n"))
            W_S = left_eigenvectors[indices, :]
        end

        # P_S v = V_S W_S^† v (in real case, W_S^† = W_S^T)
        coeffs = W_S * v
        return V_S * coeffs
    end
end


"""
    project_onto_schur_subspace(v::AbstractVector, Q::AbstractMatrix, indices::UnitRange{Int})

Project vector `v` onto the invariant subspace spanned by Schur vectors at specified indices.

# Mathematical Background

For a Schur decomposition A = QTQ^†, the Schur vectors (columns of Q) span nested
invariant subspaces. The projection onto the subspace spanned by Q[:, indices] is:

```
P_S v = Q_S Q_S^† v
```

where Q_S = Q[:, indices]. Since Q is unitary (Q^† Q = I), this is always well-defined.

# Arguments
- `v::AbstractVector`: Vector to project (length n)
- `Q::AbstractMatrix`: Unitary Schur matrix (n × n, with Q^† Q = I)
- `indices::UnitRange{Int}`: Column indices of Schur vectors to project onto

# Returns
Projected vector P_S v of length n.

# Example
```julia
using LinearAlgebra, BallArithmetic

# Matrix with complex eigenvalues
A = [0.0 1.0; -1.0 0.0]  # Rotation matrix
F = schur(A)

# Project onto first Schur vector
v = [1.0, 1.0]
v_proj = project_onto_schur_subspace(v, F.Z, 1:1)
```

# Notes
- Schur vectors are always orthonormal, so projection is straightforward
- The subspace spanned by Q[:, 1:k] is an invariant subspace of A
- For real Schur form, complex eigenvalue pairs share a 2D invariant subspace
- This is simpler than eigenspace projection for non-normal matrices

# References
- Golub, G. H., Van Loan, C. F. "Matrix Computations" (2013), Section 7.6
"""
function project_onto_schur_subspace(v::AbstractVector,
                                      Q::AbstractMatrix,
                                      indices::UnitRange{Int})
    n = length(v)
    size(Q, 1) == n || throw(DimensionMismatch("Schur matrix must have same row dimension as vector"))
    maximum(indices) <= size(Q, 2) || throw(ArgumentError("Indices out of bounds"))

    # Extract Schur vectors
    Q_S = Q[:, indices]

    # Project: P_S v = Q_S Q_S^† v
    coeffs = Q_S' * v
    return Q_S * coeffs
end


"""
    compute_eigenspace_projector(V::AbstractMatrix, indices::UnitRange{Int};
                                hermitian::Bool=false,
                                left_eigenvectors::Union{Nothing, AbstractMatrix}=nothing)

Compute the projection matrix P_S onto the eigenspace spanned by specified eigenvectors.

Returns the n × n projection matrix P_S such that P_S v projects any vector v onto
the specified eigenspace.

# Arguments
- `V::AbstractMatrix`: Matrix of right eigenvectors (n × n)
- `indices::UnitRange{Int}`: Column indices of eigenvectors defining the subspace
- `hermitian::Bool=false`: If true, assumes V is orthogonal
- `left_eigenvectors::Union{Nothing, AbstractMatrix}=nothing`: Left eigenvectors W = V⁻¹

# Returns
n × n projection matrix P_S with P_S² = P_S.

# Example
```julia
F = eigen(A)
P = compute_eigenspace_projector(F.vectors, 1:2; hermitian=true)

# Now can project multiple vectors
v1_proj = P * v1
v2_proj = P * v2
```

# See Also
- [`project_onto_eigenspace`](@ref): Project single vector without forming matrix
"""
function compute_eigenspace_projector(V::AbstractMatrix,
                                      indices::UnitRange{Int};
                                      hermitian::Bool=false,
                                      left_eigenvectors::Union{Nothing, AbstractMatrix}=nothing)
    n = size(V, 1)
    maximum(indices) <= size(V, 2) || throw(ArgumentError("Indices out of bounds"))

    V_S = V[:, indices]

    if hermitian
        # P = V_S (V_S^† V_S)^{-1} V_S^†
        G = V_S' * V_S
        return V_S * (G \ V_S')
    else
        if left_eigenvectors === nothing
            @warn "Computing V⁻¹ for non-normal projector (may be unstable)"
            V_inv = inv(V)
        else
            V_inv = left_eigenvectors
        end

        W_S = V_inv[indices, :]

        # P = V_S W_S
        return V_S * W_S
    end
end


"""
    compute_schur_projector(Q::AbstractMatrix, indices::UnitRange{Int})

Compute the projection matrix onto the invariant subspace spanned by Schur vectors.

Returns the n × n projection matrix P_S = Q_S Q_S^† where Q_S = Q[:, indices].

# Arguments
- `Q::AbstractMatrix`: Unitary Schur matrix (n × n)
- `indices::UnitRange{Int}`: Column indices of Schur vectors

# Returns
n × n projection matrix P_S with P_S² = P_S.

# Example
```julia
F = schur(A)
P = compute_schur_projector(F.Z, 1:3)  # Project onto first 3-dimensional invariant subspace
```
"""
function compute_schur_projector(Q::AbstractMatrix, indices::UnitRange{Int})
    maximum(indices) <= size(Q, 2) || throw(ArgumentError("Indices out of bounds"))

    Q_S = Q[:, indices]
    return Q_S * Q_S'
end


# Convenience constructors for BallVector if not already defined
# (These may already exist in the main BallArithmetic module)
if !@isdefined(BallVector)
    """
        BallVector{T, NT}

    Vector of balls representing interval vector [v - r, v + r] componentwise.
    """
    struct BallVector{T, NT}
        c::Vector{NT}  # Centers
        r::Vector{NT}  # Radii
    end

    BallVector(c::Vector{T}, r::Vector{T}) where T = BallVector{T, T}(c, r)
    BallVector(c::Vector{T}) where T = BallVector(c, zeros(T, length(c)))

    Base.length(v::BallVector) = length(v.c)
    Base.getindex(v::BallVector, i::Int) = Ball(v.c[i], v.r[i])
end
