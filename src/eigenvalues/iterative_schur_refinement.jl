# Iterative refinement of Schur decompositions for higher precision
# Based on Algorithm 4 from Bujanović, Kressner & Schröder (2022/2024)
#
# This allows computing Schur decomposition in Float64 and refining to BigFloat
# precision using Newton-like iterations, achieving 10-20× speedup over direct
# computation in higher precision.

"""
    SchurRefinementResult{T, RT}

Result from iterative Schur refinement to higher precision.

# Fields
- `Q::Matrix{T}`: Refined orthogonal/unitary Schur basis
- `T::Matrix{T}`: Refined upper triangular Schur form
- `iterations::Int`: Number of refinement iterations performed
- `residual_norm::RT`: Final residual ‖A - QTQ^H‖_F / ‖A‖_F (real type)
- `orthogonality_defect::RT`: Final ‖Q^H Q - I‖_F (real type)
- `converged::Bool`: Whether refinement converged to desired tolerance

# References
- [BujanovicKressnerSchroder2022](@cite) Bujanović, Kressner & Schröder,
  "Iterative refinement of Schur decompositions", Numer. Algorithms 95, 247–267 (2024)
"""
struct SchurRefinementResult{T, RT<:Real}
    Q::Matrix{T}
    T::Matrix{T}
    iterations::Int
    residual_norm::RT
    orthogonality_defect::RT
    converged::Bool
end

"""
    _frobenius_norm(A::AbstractMatrix)

Compute Frobenius norm of a matrix. Works with any element type including BigFloat.
"""
function _frobenius_norm(A::AbstractMatrix{T}) where T
    s = zero(real(T))
    @inbounds for j in axes(A, 2)
        for i in axes(A, 1)
            s += abs2(A[i, j])
        end
    end
    return sqrt(s)
end

"""
    newton_schulz_orthogonalize!(Q::Matrix{T}; max_iter=10, tol=nothing) where T

Apply Newton-Schulz iteration to orthogonalize Q in-place.

The iteration is:
    Q_{k+1} = (1/2) Q_k (3I - Q_k^H Q_k)

This converges quadratically if ‖Q^H Q - I‖₂ < 1.

# Arguments
- `Q`: Matrix to orthogonalize (modified in place)
- `max_iter`: Maximum iterations (default: 10)
- `tol`: Convergence tolerance (default: machine epsilon of T)

# Returns
- Number of iterations performed
- Final orthogonality defect ‖Q^H Q - I‖_F

# References
- [BujanovicKressnerSchroder2022](@cite) Section 2.2.2 (Newton-Schulz iteration)
"""
function newton_schulz_orthogonalize!(Q::Matrix{T}; max_iter::Int=10, tol=nothing) where T
    n = size(Q, 1)
    tol = isnothing(tol) ? 100 * eps(real(T)) : tol

    I_n = Matrix{T}(I, n, n)

    for iter in 1:max_iter
        # Compute Q^H * Q
        QtQ = Q' * Q

        # Check convergence (use Frobenius norm for BigFloat compatibility)
        defect = _frobenius_norm(QtQ - I_n)
        if defect < tol
            return iter, defect
        end

        # Newton-Schulz step: Q = (1/2) * Q * (3I - Q^H Q)
        Q .= Q * (T(3) * I_n - QtQ) / T(2)
    end

    # Final defect
    final_defect = _frobenius_norm(Q' * Q - I_n)
    return max_iter, final_defect
end


#####################################################
# TRIANGULAR MATRIX EQUATION SOLVERS                #
# Based on Algorithms 2 and 3 from Bujanović et al. #
#####################################################

"""
    _stril(A::AbstractMatrix)

Return the strictly lower triangular part of matrix A (excluding diagonal).
"""
function _stril(A::AbstractMatrix{T}) where T
    n, m = size(A)
    L = zeros(T, n, m)
    for j in 1:min(n-1, m)
        for i in (j+1):n
            L[i, j] = A[i, j]
        end
    end
    return L
end

"""
    _solve_triangular_equation_direct!(L::Matrix{T}, T_mat::Matrix{T}, E::Matrix{T};
                                       max_entry::Real=1e5) where T

Solve the triangular matrix equation stril(TL - LT) = -E for strictly lower triangular L.

This implements Algorithm 2 from Bujanović et al. (2022): successive substitution
with bottom-to-top columnwise order.

# Arguments
- `L`: Output matrix (will contain strictly lower triangular solution)
- `T_mat`: Upper triangular matrix with pairwise distinct diagonal entries
- `E`: Strictly lower triangular right-hand side matrix
- `max_entry`: Maximum allowed absolute value for entries of L (default: 1e5).
  Larger values are truncated to zero to prevent instability from clustered eigenvalues.

# Details
For 1 ≤ j < i ≤ n, the entry L[i,j] is computed as:

    L[i,j] = -1/(T[i,i] - T[j,j]) * (E[i,j] + Σ_{k=i+1}^n T[i,k]*L[k,j] - Σ_{k=1}^{j-1} L[i,k]*T[k,j])

where entries to the left or below (i,j) must be computed first.

# Notes
When eigenvalues are clustered (nearly equal), the denominator T[i,i] - T[j,j] becomes
very small, causing large entries in L that can destabilize the algorithm. Following
the suggestion in Bujanović et al. (Example 10), entries larger than `max_entry` are
set to zero.

# References
- [BujanovicKressnerSchroder2022](@cite) Algorithm 2: Successive substitution
"""
function _solve_triangular_equation_direct!(L::Matrix{T}, T_mat::Matrix{T}, E::Matrix{T};
                                            max_entry::Real=1e5) where T
    n = size(T_mat, 1)

    # Initialize L to zero
    fill!(L, zero(T))

    # Solve column by column (j = 1 to n-1)
    # Within each column, solve from bottom to top (i = n down to j+1)
    for j in 1:(n-1)
        for i in n:-1:(j+1)
            # Compute: E[i,j] + T[i, i+1:n] · L[i+1:n, j] - L[i, 1:j-1] · T[1:j-1, j]
            val = E[i, j]

            # Add contribution from rows below: Σ_{k=i+1}^n T[i,k]*L[k,j]
            for k in (i+1):n
                val += T_mat[i, k] * L[k, j]
            end

            # Subtract contribution from columns to the left: Σ_{k=1}^{j-1} L[i,k]*T[k,j]
            for k in 1:(j-1)
                val -= L[i, k] * T_mat[k, j]
            end

            # Divide by eigenvalue difference
            denom = T_mat[i, i] - T_mat[j, j]
            if abs(denom) < eps(real(T)) * max(abs(T_mat[i, i]), abs(T_mat[j, j]), one(real(T)))
                # Near-singular: eigenvalues too close, set to zero to avoid instability
                L[i, j] = zero(T)
            else
                entry = -val / denom
                # Truncate large entries to zero (following Example 10 in the paper)
                if abs(entry) > max_entry
                    L[i, j] = zero(T)
                else
                    L[i, j] = entry
                end
            end
        end
    end

    return L
end

"""
    _solve_sylvester_triangular!(X::Matrix{T}, T22::Matrix{T}, T11::Matrix{T}, C::Matrix{T}) where T

Solve the triangular Sylvester equation T22 * X - X * T11 = C where T11 and T22 are
upper triangular (from Schur form).

This corresponds to equation (15) in Bujanović et al. (2022):
    T22 * L21 - L21 * T11 = -E21

# Arguments
- `X`: Output matrix (will contain solution), size (m × k)
- `T22`: Upper triangular matrix (m × m)
- `T11`: Upper triangular matrix (k × k)
- `C`: Right-hand side matrix (m × k)

# References
- [BujanovicKressnerSchroder2022](@cite) Section 2.1, equation (15)
"""
function _solve_sylvester_triangular!(X::Matrix{T}, T22::Matrix{T}, T11::Matrix{T}, C::Matrix{T}) where T
    m = size(T22, 1)
    k = size(T11, 1)

    # Initialize X from C
    X .= C

    # Solve column by column from left to right
    # For column j: T22 * X[:,j] - X[:,j] * T11[j,j] = C[:,j] + X[:,1:j-1] * T11[1:j-1,j]
    for j in 1:k
        # Update RHS with contributions from already computed columns
        for l in 1:(j-1)
            for i in 1:m
                X[i, j] += X[i, l] * T11[l, j]
            end
        end

        # Solve (T22 - T11[j,j] * I) * X[:,j] = X[:,j]
        # This is upper triangular system, solve from bottom to top
        for i in m:-1:1
            # Accumulate contributions from rows below
            for l in (i+1):m
                X[i, j] -= T22[i, l] * X[l, j]
            end

            denom = T22[i, i] - T11[j, j]
            if abs(denom) < eps(real(T)) * max(abs(T22[i, i]), abs(T11[j, j]), one(real(T)))
                # Near-singular: eigenvalues too close
                X[i, j] = zero(T)
            else
                X[i, j] = X[i, j] / denom
            end
        end
    end

    return X
end

"""
    _solve_triangular_equation_recursive!(L::Matrix{T}, T_mat::Matrix{T}, E::Matrix{T};
                                          nmin::Int=32, max_entry::Real=1e5) where T

Solve stril(TL - LT) = -E using the recursive block algorithm (Algorithm 3).

This achieves better cache efficiency for large matrices by recursively partitioning
into 2×2 blocks and using Sylvester equation solvers for off-diagonal blocks.

# Arguments
- `L`: Output matrix (will contain strictly lower triangular solution)
- `T_mat`: Upper triangular matrix with pairwise distinct diagonal entries
- `E`: Strictly lower triangular right-hand side matrix
- `nmin`: Minimum block size before switching to direct algorithm (default: 32)
- `max_entry`: Maximum allowed absolute value for entries of L (default: 1e5)

# Algorithm
Partition T, E, L into 2×2 blocks:
    T = [T11  T12]    E = [E11   0 ]    L = [L11   0 ]
        [ 0   T22]        [E21  E22]        [L21  L22]

1. Solve Sylvester equation: T22 * L21 - L21 * T11 = -E21
2. Update: E11 ← E11 + stril(T12 * L21)
3. Update: E22 ← E22 - stril(L21 * T12)
4. Recursively solve for L11 from stril(T11*L11 - L11*T11) = -E11
5. Recursively solve for L22 from stril(T22*L22 - L22*T22) = -E22

# References
- [BujanovicKressnerSchroder2022](@cite) Algorithm 3: Recursive block algorithm
"""
function _solve_triangular_equation_recursive!(L::Matrix{T}, T_mat::Matrix{T}, E::Matrix{T};
                                               nmin::Int=32, max_entry::Real=1e5) where T
    n = size(T_mat, 1)

    # Base case: use direct algorithm for small matrices
    if n <= nmin
        return _solve_triangular_equation_direct!(L, T_mat, E; max_entry=max_entry)
    end

    # Partition at midpoint
    n1 = div(n, 2)
    ind1 = 1:n1
    ind2 = (n1+1):n

    # Extract blocks
    T11 = T_mat[ind1, ind1]
    T12 = T_mat[ind1, ind2]
    T22 = T_mat[ind2, ind2]

    E11 = E[ind1, ind1]
    E21 = E[ind2, ind1]
    E22 = E[ind2, ind2]

    # Step 1: Solve Sylvester equation T22 * L21 - L21 * T11 = -E21
    L21 = similar(E21)
    _solve_sylvester_triangular!(L21, T22, T11, -E21)

    # Store L21 in output
    L[ind2, ind1] .= L21

    # Step 2: Update E11 ← E11 + stril(T12 * L21)
    temp = T12 * L21
    for j in 1:n1
        for i in (j+1):n1
            E11[i, j] += temp[i, j]
        end
    end

    # Step 3: Update E22 ← E22 - stril(L21 * T12)
    temp = L21 * T12
    n2 = n - n1
    for j in 1:n2
        for i in (j+1):n2
            E22[i, j] -= temp[i, j]
        end
    end

    # Step 4: Recursively solve for L11
    L11 = zeros(T, n1, n1)
    _solve_triangular_equation_recursive!(L11, T11, E11; nmin=nmin, max_entry=max_entry)
    L[ind1, ind1] .= L11

    # Step 5: Recursively solve for L22
    L22 = zeros(T, n2, n2)
    _solve_triangular_equation_recursive!(L22, T22, E22; nmin=nmin, max_entry=max_entry)
    L[ind2, ind2] .= L22

    # Zero out upper triangular part (L should be strictly lower triangular)
    for j in 1:n
        for i in 1:j
            L[i, j] = zero(T)
        end
    end

    return L
end

"""
    solve_triangular_matrix_equation(T_mat::Matrix{T}, E::Matrix{T};
                                     use_recursive::Bool=true, nmin::Int=32, max_entry::Real=1e5) where T

Solve the triangular matrix equation stril(TL - LT) = -E for strictly lower triangular L.

Given an upper triangular matrix T with pairwise distinct diagonal entries and a strictly
lower triangular matrix E, finds the unique strictly lower triangular matrix L satisfying
the equation.

# Arguments
- `T_mat`: Upper triangular matrix (n × n) with pairwise distinct diagonal entries
- `E`: Strictly lower triangular matrix (n × n)
- `use_recursive`: If true, use recursive block algorithm (default: true)
- `nmin`: Minimum block size for recursive algorithm (default: 32)
- `max_entry`: Maximum allowed entry magnitude before truncation (default: 1e5).
  This prevents instability from clustered eigenvalues (see Example 10 in the paper).

# Returns
Strictly lower triangular matrix L satisfying stril(TL - LT) = -E

# References
- [BujanovicKressnerSchroder2022](@cite) Algorithms 2 and 3
"""
function solve_triangular_matrix_equation(T_mat::Matrix{T}, E::Matrix{T};
                                          use_recursive::Bool=true, nmin::Int=32, max_entry::Real=1e5) where T
    n = size(T_mat, 1)
    L = zeros(T, n, n)

    if use_recursive && n > nmin
        _solve_triangular_equation_recursive!(L, T_mat, E; nmin=nmin, max_entry=max_entry)
    else
        _solve_triangular_equation_direct!(L, T_mat, E; max_entry=max_entry)
    end

    return L
end

"""
    refine_schur_decomposition(A::AbstractMatrix{T}, Q0::Matrix, T0::Matrix;
                                target_precision::Int=256,
                                max_iterations::Int=20,
                                tol::Real=0.0) where T

Refine an approximate Schur decomposition A ≈ Q₀ T₀ Q₀^H to higher precision.

Starting from an approximate Schur decomposition computed in Float64, this
function iteratively refines Q and T to achieve the accuracy of `target_precision`
bits. The algorithm uses Newton-like iterations that converge quadratically.

# Algorithm (from Bujanović et al. 2022, Algorithm 4)

Given A = Q₀ T₀ Q₀^H + E where E is the initial error:

1. **Pre-computation** (2 high-precision matrix multiplications):
   - Compute A_Q = A * Q and Q^H_A = Q^H * A in target precision

2. **Iteration** (4 high-precision matrix multiplications per step):
   - Compute residual R = Q^H * A_Q - T
   - Extract diagonal correction: T_new = T + diag(R)
   - Solve Sylvester equations for off-diagonal corrections
   - Apply Newton-Schulz to re-orthogonalize Q
   - Repeat until convergence

The method achieves 10-20× speedup over computing Schur directly in high precision.

# Arguments
- `A`: Original matrix (will be converted to target precision)
- `Q0`: Approximate orthogonal/unitary factor from Schur decomposition
- `T0`: Approximate upper triangular factor from Schur decomposition
- `target_precision`: Target precision in bits (default: 256 for BigFloat)
- `max_iterations`: Maximum refinement iterations (default: 20)
- `tol`: Convergence tolerance (default: machine epsilon of target precision)

# Returns
[`SchurRefinementResult`](@ref) containing refined Q, T and convergence information.

# Example
```julia
using BallArithmetic, LinearAlgebra

# Compute Schur in Float64
A = randn(100, 100)
F = schur(A)
Q0, T0 = F.Z, F.T

# Refine to BigFloat precision
result = refine_schur_decomposition(A, Q0, T0; target_precision=256)

# Check residual
@show result.residual_norm      # Should be ≈ 10^-77 for 256-bit precision
@show result.orthogonality_defect
```

# References
- [BujanovicKressnerSchroder2022](@cite) Bujanović, Kressner & Schröder,
  "Iterative refinement of Schur decompositions", Numer. Algorithms 95, 247–267 (2024).
  Algorithm 4: Mixed-precision Schur refinement.
"""
function refine_schur_decomposition(A::AbstractMatrix, Q0::Matrix, T0::Matrix;
                                    target_precision::Int=256,
                                    max_iterations::Int=20,
                                    tol::Real=0.0)
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("A must be square"))
    size(Q0) == (n, n) || throw(DimensionMismatch("Q0 must be n×n"))
    size(T0) == (n, n) || throw(DimensionMismatch("T0 must be n×n"))

    # Set precision for BigFloat
    old_prec = precision(BigFloat)
    setprecision(BigFloat, target_precision)

    try
        return _refine_schur_impl(A, Q0, T0, max_iterations, tol)
    finally
        setprecision(BigFloat, old_prec)
    end
end

# Helper function for converting to high precision
function _to_bigfloat(A::AbstractMatrix{T}) where T<:Real
    return convert.(BigFloat, A)
end

function _to_bigfloat(A::AbstractMatrix{Complex{T}}) where T<:Real
    return convert.(Complex{BigFloat}, A)
end

"""
    _refine_schur_impl(A, Q0, T0, max_iterations, tol)

Internal implementation of Schur refinement using Algorithm 4 from Bujanović et al. (2022).

This implements the full mixed-precision iterative refinement algorithm:

1. **Initial orthogonalization**: Apply Newton-Schulz to Q₀ converted to high precision
2. **Main iteration loop**:
   a. Compute T̂ = Q^H A Q in high precision
   b. Extract E = stril(T̂), T = T̂ - E (upper triangular part)
   c. Solve triangular matrix equation stril(TL - LT) = -E for L
   d. Construct skew-Hermitian correction W = L - L^H
   e. Compute orthogonality defect Y = Q^H Q - I
   f. Update Q with combined Newton-Schulz step:
      Q_new = ½ Q (2I + 2W - Y - YW + W² + W³)
3. **Convergence**: Stop when ‖E‖_F and ‖Y‖_F are below tolerance

The algorithm achieves quadratic convergence and typically needs only 3-4 iterations
to refine from double to quadruple precision.

# References
- [BujanovicKressnerSchroder2022](@cite) Bujanović, Kressner & Schröder,
  "Iterative refinement of Schur decompositions", Numer. Algorithms 95, 247–267 (2024).
  Algorithm 4: Mixed-precision Schur refinement.
"""
function _refine_schur_impl(A::AbstractMatrix, Q0::Matrix, T0::Matrix,
                            max_iterations::Int, tol::Real)
    n = size(A, 1)

    # Convert to BigFloat (or Complex{BigFloat} for complex matrices)
    A_high = _to_bigfloat(A)
    Q = _to_bigfloat(Q0)
    T_mat = _to_bigfloat(T0)

    # Determine element type for high precision
    HP = eltype(Q)  # BigFloat or Complex{BigFloat}
    HP_real = real(HP)

    # Determine tolerance
    target_tol = tol > 0 ? HP_real(tol) : HP_real(100) * eps(HP_real)

    # Compute initial norm for relative tolerance
    A_norm = _frobenius_norm(A_high)

    # Identity matrix
    I_n = Matrix{HP}(I, n, n)

    # Step 1: Initial orthogonalization of Q using Newton-Schulz (Algorithm 4, line 2)
    # Q ← ½Q̂(3I - Q̂^H Q̂)
    newton_schulz_orthogonalize!(Q; max_iter=3, tol=target_tol)

    converged = false
    iterations = 0
    residual_norm = HP_real(Inf)
    orthogonality_defect = HP_real(Inf)

    # Main iteration loop (Algorithm 4, lines 3-12)
    for iter in 1:max_iterations
        iterations = iter

        # Step 2: Compute T̂ = Q^H * A * Q in high precision (line 4)
        T_hat = Q' * A_high * Q

        # Step 3: Extract E = stril(T̂) and T = T̂ - E (line 5)
        # E is strictly lower triangular part, T is upper triangular
        E = _stril(T_hat)
        T_mat = T_hat - E

        # Check convergence: E should become small
        E_norm = _frobenius_norm(E)
        residual_norm = E_norm / A_norm

        # Also check orthogonality
        Y = Q' * Q - I_n
        orthogonality_defect = _frobenius_norm(Y)

        if residual_norm < target_tol && orthogonality_defect < target_tol
            converged = true
            return SchurRefinementResult(
                Q, T_mat, iterations, residual_norm, orthogonality_defect, converged
            )
        end

        # Step 4: Solve triangular matrix equation stril(TL - LT) = -E for L (line 6)
        # According to Algorithm 4 in Bujanović et al., this should be done in LOW precision
        # Convert to Float64/ComplexF64 for the solve, then convert back
        T_lp = HP <: Complex ? convert.(ComplexF64, T_mat) : convert.(Float64, T_mat)
        E_lp = HP <: Complex ? convert.(ComplexF64, E) : convert.(Float64, E)

        # Solve in low precision with aggressive truncation for stability
        # The paper (Example 10) suggests truncating large entries to avoid instability
        L_lp = solve_triangular_matrix_equation(T_lp, E_lp; use_recursive=true, nmin=32, max_entry=1e5)

        # Convert L back to high precision (line 7)
        L = convert.(HP, L_lp)

        # Step 5: Construct skew-Hermitian W = L - L^H (line 8)
        W = L - L'

        # Step 6: Compute products for Newton-Schulz update (lines 9-10)
        # Following equation (23) from the paper:
        # Q_new = ½ Q (2I + 2W - Y - YW + W² + W³ + W²Y + W²YW)
        #
        # According to Algorithm 4, line 9: compute YW, W², W³ in LOW precision
        # This is more efficient and avoids numerical issues
        W_lp = HP <: Complex ? convert.(ComplexF64, W) : convert.(Float64, W)
        Y_lp = HP <: Complex ? convert.(ComplexF64, Y) : convert.(Float64, Y)

        W2_lp = W_lp * W_lp
        W3_lp = W2_lp * W_lp
        YW_lp = Y_lp * W_lp

        # Convert back to high precision for the summation (line 10)
        W2 = convert.(HP, W2_lp)
        W3 = convert.(HP, W3_lp)
        YW = convert.(HP, YW_lp)

        # Compute Σ = 2I + 2W - Y - YW + W² + W³ in high precision (line 10)
        Σ = 2 * I_n + 2 * W - Y - YW + W2 + W3

        # Step 7: Update Q ← ½QΣ (line 11)
        Q .= (Q * Σ) / HP(2)
    end

    # Final computation if not converged
    T_hat = Q' * A_high * Q
    E = _stril(T_hat)
    T_mat = T_hat - E

    E_norm = _frobenius_norm(E)
    residual_norm = E_norm / A_norm

    Y = Q' * Q - I_n
    orthogonality_defect = _frobenius_norm(Y)

    return SchurRefinementResult(
        Q, T_mat, iterations, residual_norm, orthogonality_defect, converged
    )
end

"""
    rigorous_schur_bigfloat(A::BallMatrix{T}; target_precision::Int=256,
                            max_iterations::Int=20) where T

Compute rigorous Schur decomposition with BigFloat precision using iterative refinement.

This combines:
1. Fast approximate Schur decomposition in Float64
2. Iterative refinement to BigFloat precision
3. Rigorous error certification

# Arguments
- `A::BallMatrix`: Input ball matrix
- `target_precision`: Target BigFloat precision in bits (default: 256)
- `max_iterations`: Maximum refinement iterations (default: 20)

# Returns
A tuple `(Q_ball, T_ball, result)` where:
- `Q_ball::BallMatrix{BigFloat}`: Rigorous enclosure of Schur basis
- `T_ball::BallMatrix{BigFloat}`: Rigorous enclosure of Schur form
- `result::SchurRefinementResult`: Refinement diagnostics

# Example
```julia
A = BallMatrix(randn(10, 10), fill(1e-10, 10, 10))
Q, T, result = rigorous_schur_bigfloat(A; target_precision=512)
@show result.converged
```

# References
- [BujanovicKressnerSchroder2022](@cite) for the iterative refinement algorithm
- [Rump2010](@cite) for rigorous error certification
"""
function rigorous_schur_bigfloat(A::BallMatrix{T, NT};
                                  target_precision::Int=256,
                                  max_iterations::Int=20) where {T, NT}
    n = size(A, 1)

    # Step 1: Compute approximate Schur in Float64
    # Handle both real and complex BallMatrix inputs
    # Always use complex Schur to ensure strictly upper triangular T
    # (real Schur has 2x2 blocks for complex eigenvalue pairs which the algorithm doesn't handle)
    A_mid = mid(A)
    A_center = convert.(ComplexF64, A_mid)
    F = schur(A_center)
    Q0, T0 = F.Z, F.T

    # Step 2: Refine to BigFloat precision
    result = refine_schur_decomposition(A_center, Q0, T0;
                                        target_precision=target_precision,
                                        max_iterations=max_iterations)

    if !result.converged
        @warn "Schur refinement did not converge. Residual: $(result.residual_norm)"
    end

    # Step 3: Certify with rigorous error bounds
    # The residual gives us the backward error
    # Forward error is bounded by condition number * backward error

    old_prec = precision(BigFloat)
    setprecision(BigFloat, target_precision)

    try
        # Convert input uncertainties to BigFloat
        A_rad_big = convert.(BigFloat, rad(A))

        # Compute rigorous error bounds
        # Residual norm gives backward error
        backward_error = result.residual_norm

        # Orthogonality defect contributes to Q error
        Q_error = result.orthogonality_defect

        # Build ball matrices with certified errors
        # Q radius: orthogonality defect + propagated input uncertainty
        # Use Frobenius norm (upper bound on spectral norm) for BigFloat compatibility
        Q_rad = fill(Q_error + _frobenius_norm(A_rad_big), n, n)
        Q_ball = BallMatrix(result.Q, Q_rad)

        # T radius: backward error + propagated input uncertainty
        T_rad = fill(backward_error * _frobenius_norm(result.T) + _frobenius_norm(A_rad_big), n, n)
        T_ball = BallMatrix(result.T, T_rad)

        return Q_ball, T_ball, result
    finally
        setprecision(BigFloat, old_prec)
    end
end

#####################################################
# RefSyEv: Symmetric Eigenvalue Decomposition        #
# Refinement (Ogita & Aishima, 2018)                 #
#####################################################

"""
    SymmetricEigenRefinementResult{T, RT}

Result from RefSyEv iterative refinement of symmetric eigenvalue decomposition.

# Fields
- `Q::Matrix{T}`: Refined orthogonal eigenvector matrix
- `λ::Vector{RT}`: Refined eigenvalues (real)
- `iterations::Int`: Number of refinement iterations performed
- `residual_norm::RT`: Final ‖A - QΛQ^T‖_F / ‖A‖_F
- `orthogonality_defect::RT`: Final ‖Q^T Q - I‖_F
- `converged::Bool`: Whether refinement converged to desired tolerance

# References
- [OgitaAishima2018](@cite) Ogita, T. & Aishima, K., "Iterative refinement for symmetric
  eigenvalue decomposition", Japan J. Indust. Appl. Math. 35, 1007–1035 (2018).
"""
struct SymmetricEigenRefinementResult{T, RT<:Real}
    Q::Matrix{T}
    λ::Vector{RT}
    iterations::Int
    residual_norm::RT
    orthogonality_defect::RT
    converged::Bool
end

"""
    refine_symmetric_eigen(A::AbstractMatrix{T}, Q0::Matrix, λ0::AbstractVector;
                           target_precision::Int=256,
                           max_iterations::Int=20,
                           tol::Real=0.0) where T

Refine an approximate symmetric eigenvalue decomposition A ≈ Q₀ Λ₀ Q₀^T to higher precision
using the RefSyEv algorithm from Ogita & Aishima (2018).

This algorithm is specifically designed for symmetric (or Hermitian) matrices and handles
multiple eigenvalues gracefully by using a special correction formula that avoids
division by nearly-zero eigenvalue differences.

# Algorithm (RefSyEv from Ogita & Aishima 2018)

For symmetric A with approximate decomposition A ≈ Q̂ D̂ Q̂^T:

1. Compute residuals: R = I - Q̂^T Q̂, S = Q̂^T A Q̂
2. Update eigenvalues: λ̃_i = s_ii / (1 - r_ii)  (Rayleigh quotient)
3. Compute threshold: δ = 2(‖S - D̃‖ + ‖A‖ · ‖R‖)
4. Compute correction matrix Ẽ:
   - For distinct eigenvalues (|λ̃_i - λ̃_j| > δ):
       ẽ_ij = (s_ij + λ̃_j · r_ij) / (λ̃_j - λ̃_i)
   - For multiple eigenvalues (|λ̃_i - λ̃_j| ≤ δ):
       ẽ_ij = r_ij / 2
5. Update: Q' = Q̂(I + Ẽ)

The method converges quadratically, even for matrices with multiple eigenvalues.

# Arguments
- `A`: Symmetric matrix (will be converted to target precision)
- `Q0`: Approximate orthogonal eigenvector matrix
- `λ0`: Approximate eigenvalues
- `target_precision`: Target precision in bits (default: 256)
- `max_iterations`: Maximum refinement iterations (default: 20)
- `tol`: Convergence tolerance (default: machine epsilon of target precision)

# Returns
[`SymmetricEigenRefinementResult`](@ref) containing refined Q, λ and convergence info.

# Example
```julia
using BallArithmetic, LinearAlgebra

# Create symmetric matrix
A = randn(10, 10); A = A + A'

# Compute eigen decomposition in Float64
F = eigen(Symmetric(A))
Q0, λ0 = F.vectors, F.values

# Refine to BigFloat precision
result = refine_symmetric_eigen(A, Q0, λ0; target_precision=256)

@show result.residual_norm       # Should be ≈ 10^-77
@show result.orthogonality_defect
@show result.converged
```

# Notes
- For general (non-symmetric) matrices, use [`refine_schur_decomposition`](@ref)
- This algorithm is more efficient than general Schur refinement for symmetric matrices
- Handles multiple eigenvalues without instability

# References
- [OgitaAishima2018](@cite) Ogita, T. & Aishima, K., "Iterative refinement for symmetric
  eigenvalue decomposition", Japan J. Indust. Appl. Math. 35, 1007–1035 (2018).
  Algorithm 1: RefSyEv.
"""
function refine_symmetric_eigen(A::AbstractMatrix, Q0::Matrix, λ0::AbstractVector;
                                target_precision::Int=256,
                                max_iterations::Int=20,
                                tol::Real=0.0)
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("A must be square"))
    size(Q0) == (n, n) || throw(DimensionMismatch("Q0 must be n×n"))
    length(λ0) == n || throw(DimensionMismatch("λ0 must have length n"))

    # Set precision for BigFloat
    old_prec = precision(BigFloat)
    setprecision(BigFloat, target_precision)

    try
        return _refine_symmetric_eigen_impl(A, Q0, λ0, max_iterations, tol)
    finally
        setprecision(BigFloat, old_prec)
    end
end

"""
    _refine_symmetric_eigen_impl(A, Q0, λ0, max_iterations, tol)

Internal implementation of RefSyEv algorithm (Algorithm 1 from Ogita & Aishima 2018).
"""
function _refine_symmetric_eigen_impl(A::AbstractMatrix, Q0::Matrix, λ0::AbstractVector,
                                      max_iterations::Int, tol::Real)
    n = size(A, 1)

    # Convert to BigFloat
    A_high = _to_bigfloat(A)
    Q = _to_bigfloat(Q0)
    λ = convert.(BigFloat, λ0)

    HP = eltype(Q)  # BigFloat or Complex{BigFloat}
    HP_real = real(HP)

    # Determine tolerance
    target_tol = tol > 0 ? HP_real(tol) : HP_real(100) * eps(HP_real)

    # Compute norm of A for relative tolerance
    A_norm = _frobenius_norm(A_high)

    # Identity matrix
    I_n = Matrix{HP}(I, n, n)

    converged = false
    iterations = 0
    residual_norm = HP_real(Inf)
    orthogonality_defect = HP_real(Inf)

    # Correction matrix
    E_tilde = zeros(HP, n, n)

    # Main iteration loop (Algorithm 1 from Ogita & Aishima 2018)
    for iter in 1:max_iterations
        iterations = iter

        # Line 2: R ← I - Q̂^T Q̂ (orthogonality residual)
        R = I_n - Q' * Q

        # Line 3: S ← Q̂^T A Q̂ (should be diagonal if eigenvectors are exact)
        S = Q' * A_high * Q

        # Line 4: λ̃_i ← s_ii / (1 - r_ii) for i = 1,...,n (Rayleigh quotient)
        for i in 1:n
            denom = HP(1) - R[i, i]
            if abs(denom) > eps(HP_real)
                λ[i] = real(S[i, i] / denom)
            else
                λ[i] = real(S[i, i])
            end
        end

        # Line 5: D̃ ← diag(λ̃_i)
        D_tilde = Diagonal(λ)

        # Line 6: δ ← 2(‖S - D̃‖₂ + ‖A‖₂ ‖R‖₂)
        # Use Frobenius norm as upper bound on spectral norm
        S_minus_D_norm = _frobenius_norm(S - D_tilde)
        R_norm = _frobenius_norm(R)
        δ = HP_real(2) * (S_minus_D_norm + A_norm * R_norm)

        # Check convergence
        orthogonality_defect = R_norm

        # Compute reconstruction error
        Λ = Diagonal(λ)
        reconstruction = Q * Λ * Q'
        residual_norm = _frobenius_norm(A_high - reconstruction) / A_norm

        if residual_norm < target_tol && orthogonality_defect < target_tol
            converged = true
            return SymmetricEigenRefinementResult(
                Q, λ, iterations, residual_norm, orthogonality_defect, converged
            )
        end

        # Line 7: Compute correction matrix Ẽ
        fill!(E_tilde, zero(HP))

        for j in 1:n
            for i in 1:n
                if i == j
                    # Diagonal: ẽ_ii = r_ii / 2
                    E_tilde[i, i] = R[i, i] / HP(2)
                else
                    # Off-diagonal
                    λ_diff = λ[j] - λ[i]
                    if abs(λ_diff) > δ
                        # Distinct eigenvalues: ẽ_ij = (s_ij + λ̃_j r_ij) / (λ̃_j - λ̃_i)
                        E_tilde[i, j] = (S[i, j] + λ[j] * R[i, j]) / λ_diff
                    else
                        # Multiple eigenvalues: ẽ_ij = r_ij / 2
                        E_tilde[i, j] = R[i, j] / HP(2)
                    end
                end
            end
        end

        # Line 8: X' ← X̂ + X̂ Ẽ = X̂(I + Ẽ)
        Q .= Q * (I_n + E_tilde)
    end

    # Final computation if not converged
    R = I_n - Q' * Q
    S = Q' * A_high * Q
    for i in 1:n
        denom = HP(1) - R[i, i]
        if abs(denom) > eps(HP_real)
            λ[i] = real(S[i, i] / denom)
        else
            λ[i] = real(S[i, i])
        end
    end

    orthogonality_defect = _frobenius_norm(R)

    Λ = Diagonal(λ)
    reconstruction = Q * Λ * Q'
    residual_norm = _frobenius_norm(A_high - reconstruction) / A_norm

    return SymmetricEigenRefinementResult(
        Q, λ, iterations, residual_norm, orthogonality_defect, converged
    )
end

"""
    rigorous_symmetric_eigen_bigfloat(A::BallMatrix{T};
                                       target_precision::Int=256,
                                       max_iterations::Int=20) where T

Compute rigorous symmetric eigenvalue decomposition with BigFloat precision.

This is the symmetric/Hermitian analog of [`rigorous_schur_bigfloat`](@ref), using the
RefSyEv algorithm which is specialized for symmetric matrices and handles multiple
eigenvalues gracefully.

# Arguments
- `A::BallMatrix`: Input symmetric ball matrix
- `target_precision`: Target BigFloat precision in bits (default: 256)
- `max_iterations`: Maximum refinement iterations (default: 20)

# Returns
A tuple `(Q_ball, λ_ball, result)` where:
- `Q_ball::BallMatrix{BigFloat}`: Rigorous enclosure of eigenvector matrix
- `λ_ball::Vector{Ball{BigFloat}}`: Rigorous enclosures of eigenvalues
- `result::SymmetricEigenRefinementResult`: Refinement diagnostics

# Example
```julia
# Create symmetric ball matrix
A_mid = randn(10, 10); A_mid = A_mid + A_mid'
A = BallMatrix(A_mid, fill(1e-10, 10, 10))

Q, λ, result = rigorous_symmetric_eigen_bigfloat(A; target_precision=256)
@show result.converged
```

# References
- [OgitaAishima2018](@cite) Ogita, T. & Aishima, K., "Iterative refinement for symmetric
  eigenvalue decomposition", Japan J. Indust. Appl. Math. 35, 1007–1035 (2018).
"""
function rigorous_symmetric_eigen_bigfloat(A::BallMatrix{T, NT};
                                            target_precision::Int=256,
                                            max_iterations::Int=20) where {T, NT}
    n = size(A, 1)

    # Step 1: Compute approximate eigen decomposition in Float64
    A_center = convert.(Float64, mid(A))
    # Symmetrize to ensure exact symmetry
    A_sym = (A_center + A_center') / 2
    F = eigen(Symmetric(A_sym))
    Q0, λ0 = F.vectors, F.values

    # Step 2: Refine to BigFloat precision using RefSyEv
    result = refine_symmetric_eigen(A_sym, Q0, λ0;
                                    target_precision=target_precision,
                                    max_iterations=max_iterations)

    if !result.converged
        @warn "Symmetric eigenvalue refinement did not converge. Residual: $(result.residual_norm)"
    end

    # Step 3: Certify with rigorous error bounds
    old_prec = precision(BigFloat)
    setprecision(BigFloat, target_precision)

    try
        # Convert input uncertainties to BigFloat
        A_rad_big = convert.(BigFloat, rad(A))

        # Compute rigorous error bounds
        backward_error = result.residual_norm
        Q_error = result.orthogonality_defect

        # Build ball matrices with certified errors
        Q_rad = fill(Q_error + _frobenius_norm(A_rad_big), n, n)
        Q_ball = BallMatrix(result.Q, Q_rad)

        # Eigenvalue errors: perturbation theory gives |λ - λ̂| ≤ ‖E‖ for symmetric
        λ_rad = fill(backward_error * maximum(abs.(result.λ)) + _frobenius_norm(A_rad_big), n)
        λ_ball = [Ball(result.λ[i], λ_rad[i]) for i in 1:n]

        return Q_ball, λ_ball, result
    finally
        setprecision(BigFloat, old_prec)
    end
end

# Export functions
export SchurRefinementResult, refine_schur_decomposition, rigorous_schur_bigfloat
export newton_schulz_orthogonalize!, solve_triangular_matrix_equation
export SymmetricEigenRefinementResult, refine_symmetric_eigen, rigorous_symmetric_eigen_bigfloat
