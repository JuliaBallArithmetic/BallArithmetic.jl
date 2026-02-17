# Fast eigenvector computation for triangular matrices
# Eigenvectors of triangular matrices are computable in O(n^2) via back/forward substitution,
# since V is unit triangular and V⁻¹ is also unit triangular.

"""
    _has_distinct_diagonal(A::AbstractMatrix, tol)

Check whether all diagonal entries of `A` are pairwise distinct, i.e.
`min_{i≠j} |A[i,i] - A[j,j]| > tol`.
"""
function _has_distinct_diagonal(A::AbstractMatrix, tol)
    n = size(A, 1)
    for i in 1:n
        for j in (i+1):n
            if abs(A[i, i] - A[j, j]) <= tol
                return false
            end
        end
    end
    return true
end

"""
    triangular_eigenvectors(L::AbstractMatrix{T}; tol=nothing) where T

Compute eigenvectors and their inverse for a triangular matrix in O(n²).

For a **lower triangular** matrix `L`, the eigenvector for eigenvalue `λ_i = L[i,i]`
has `v[j] = 0` for `j < i`, `v[i] = 1`, and entries for `j > i` by forward substitution.
The resulting `V` is **unit lower triangular**, so `W = V⁻¹` is also unit lower triangular.

For an **upper triangular** matrix `U`, the eigenvector for eigenvalue `λ_i = U[i,i]`
has `v[j] = 0` for `j > i`, `v[i] = 1`, and entries for `j < i` by back-substitution.
The resulting `V` is **unit upper triangular**, so `W = V⁻¹` is also unit upper triangular.

# Fallback
If `min_{i≠j} |L[i,i] - L[j,j]| < tol`, falls back to `eigen()` + `inv()`.

# Arguments
- `L::AbstractMatrix`: A triangular matrix (upper or lower).
- `tol`: Tolerance for detecting repeated eigenvalues. Defaults to `n * eps(real(eltype(L)))`.

# Returns
`(V, W, eigenvalues)` where `V` is the eigenvector matrix, `W = V⁻¹`, and
`eigenvalues = diag(L)`.
"""
function triangular_eigenvectors(L::AbstractMatrix{T}; tol=nothing) where T
    n = size(L, 1)
    n == size(L, 2) || throw(DimensionMismatch("Matrix must be square"))

    RT = real(float(T))
    if tol === nothing
        tol = n * eps(RT)
    end

    eigenvalues = diag(L)

    if !_has_distinct_diagonal(L, tol)
        # Fallback: repeated eigenvalues
        F = eigen(Matrix(L))
        V = Matrix(F.vectors)
        W = inv(V)
        return V, W, F.values
    end

    CT = promote_type(float(T), Complex{RT})

    if istril(L)
        V, W = _triangular_eigenvectors_lower(L, eigenvalues, CT)
    elseif istriu(L)
        V, W = _triangular_eigenvectors_upper(L, eigenvalues, CT)
    else
        throw(ArgumentError("Matrix must be upper or lower triangular"))
    end

    return V, W, eigenvalues
end

# Lower triangular: eigenvector i has V[j,i] = 0 for j<i, V[i,i] = 1,
# and V[j,i] for j>i by forward substitution.
function _triangular_eigenvectors_lower(L::AbstractMatrix, eigenvalues, ::Type{CT}) where CT
    n = size(L, 1)
    V = zeros(CT, n, n)

    for i in 1:n
        V[i, i] = one(CT)
        λi = eigenvalues[i]
        # Forward substitution for j = i+1, ..., n
        for j in (i+1):n
            # (L[j,j] - λi) * V[j,i] = -sum_{m=i}^{j-1} L[j,m] * V[m,i]
            s = zero(CT)
            for m in i:(j-1)
                s += L[j, m] * V[m, i]
            end
            V[j, i] = -s / (L[j, j] - λi)
        end
    end

    # W = V⁻¹ is also unit lower triangular; compute by forward substitution on V*W = I
    W = zeros(CT, n, n)
    for i in 1:n
        W[i, i] = one(CT)
    end
    for i in 1:n
        for j in (i+1):n
            # V[j,j]*W[j,i] + sum_{m=i}^{j-1} V[j,m]*W[m,i] = 0
            # Since V[j,j] = 1: W[j,i] = -sum_{m=i}^{j-1} V[j,m]*W[m,i]
            s = zero(CT)
            for m in i:(j-1)
                s += V[j, m] * W[m, i]
            end
            W[j, i] = -s
        end
    end

    return V, W
end

# Upper triangular: eigenvector i has V[j,i] = 0 for j>i, V[i,i] = 1,
# and V[j,i] for j<i by back-substitution.
function _triangular_eigenvectors_upper(U::AbstractMatrix, eigenvalues, ::Type{CT}) where CT
    n = size(U, 1)
    V = zeros(CT, n, n)

    for i in 1:n
        V[i, i] = one(CT)
        λi = eigenvalues[i]
        # Back-substitution for j = i-1, ..., 1
        for j in (i-1):-1:1
            # (U[j,j] - λi) * V[j,i] = -sum_{m=j+1}^{i} U[j,m] * V[m,i]
            s = zero(CT)
            for m in (j+1):i
                s += U[j, m] * V[m, i]
            end
            V[j, i] = -s / (U[j, j] - λi)
        end
    end

    # W = V⁻¹ is also unit upper triangular; compute by back-substitution on V*W = I
    W = zeros(CT, n, n)
    for i in 1:n
        W[i, i] = one(CT)
    end
    for i in 1:n
        for j in (i-1):-1:1
            # V[j,j]*W[j,i] + sum_{m=j+1}^{i} V[j,m]*W[m,i] = 0
            # Since V[j,j] = 1: W[j,i] = -sum_{m=j+1}^{i} V[j,m]*W[m,i]
            s = zero(CT)
            for m in (j+1):i
                s += V[j, m] * W[m, i]
            end
            W[j, i] = -s
        end
    end

    return V, W
end
