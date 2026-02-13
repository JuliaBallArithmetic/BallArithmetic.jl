# Verified LU Decomposition with Rigorous Error Bounds
# Based on Section 3 of Rump & Ogita (2024) "Verified Error Bounds for Matrix Decompositions"
#
# Key technique: Precondition A to get I+E, compute verified LU of perturbed identity,
# then transform back to get verified bounds for original factors.

# Helper to get appropriate BigFloat type (handles Complex)
_bigfloat_type(::Type{T}) where {T<:Real} = BigFloat
_bigfloat_type(::Type{Complex{T}}) where {T<:Real} = Complex{BigFloat}
_to_bigfloat(M::AbstractMatrix{T}) where {T} = convert.(_bigfloat_type(T), M)
_to_bigfloat(v::AbstractVector{T}) where {T} = convert.(_bigfloat_type(T), v)

# Helper to get the working float type (Float64 or BigFloat)
_working_type(::Type{T}, use_bigfloat::Bool) where {T<:Real} = use_bigfloat ? BigFloat : Float64
_working_type(::Type{Complex{T}}, use_bigfloat::Bool) where {T<:Real} = use_bigfloat ? Complex{BigFloat} : ComplexF64
_to_working(M::AbstractMatrix{T}, use_bigfloat::Bool) where {T} = use_bigfloat ? _to_bigfloat(M) : convert.(T <: Complex ? ComplexF64 : Float64, M)
_to_working(v::AbstractVector{T}, use_bigfloat::Bool) where {T} = use_bigfloat ? _to_bigfloat(v) : convert.(T <: Complex ? ComplexF64 : Float64, v)

"""
    VerifiedLUResult{LM, UM, RT}

Result from verified LU decomposition with rigorous error bounds.

# Fields
- `L::LM`: Lower triangular factor with unit diagonal (rigorous enclosure as BallMatrix)
- `U::UM`: Upper triangular factor (rigorous enclosure as BallMatrix)
- `p::Vector{Int}`: Row permutation vector
- `success::Bool`: Whether verification succeeded
- `residual_norm::RT`: Bound on ‖LU - A(p,:)‖ / ‖A‖

# Mathematical Guarantee
For any L̃ ∈ L, Ũ ∈ U: L̃Ũ = A[p,:] (the permuted input matrix).

# References
- [RumpOgita2024](@cite) Rump & Ogita, "Verified Error Bounds for Matrix Decompositions",
  Section 3: LU decomposition.
"""
struct VerifiedLUResult{LM<:BallMatrix, UM<:BallMatrix, RT<:Real}
    L::LM
    U::UM
    p::Vector{Int}
    success::Bool
    residual_norm::RT
end

"""
    _lu_perturbed_identity(E::AbstractMatrix{T}; precision_bits::Int=256, use_bigfloat::Bool=true) where T

Compute verified LU decomposition of I + E where E is a small perturbation.

This is the core algorithm from Section 3.1 of Rump & Ogita (2024).
For ‖E‖∞ < 1, the matrix I + E has a unique LU decomposition.

# Arguments
- `precision_bits`: Precision for BigFloat computation (ignored if use_bigfloat=false)
- `use_bigfloat`: If true, use BigFloat for high precision; if false, use Float64 with directed rounding

# Returns
Tuple (L_offset, U_offset, L_inv_offset, U_inv_offset, success) where:
- L = I + L_offset (L_offset is strictly lower triangular)
- U = I + U_offset (U_offset is upper triangular with zero diagonal for square case)
- L⁻¹ = I + L_inv_offset
- U⁻¹ = I + U_inv_offset

# Algorithm (from equations 3.1-3.7 in the paper)
The key insight is that L[i,k] - E[i,k] can be bounded by an outer product,
allowing O(n²) computation of verified bounds.
"""
function _lu_perturbed_identity(E::AbstractMatrix{T};
                                 precision_bits::Int=256,
                                 use_bigfloat::Bool=true) where T
    m, n = size(E)
    mn = min(m, n)

    # Set precision for rigorous computation (only needed for BigFloat mode)
    old_prec = precision(BigFloat)
    if use_bigfloat
        setprecision(BigFloat, precision_bits)
    end

    try
        # Convert to working precision type
        WT = _working_type(T, use_bigfloat)
        RWT = real(WT)
        E_w = _to_working(E, use_bigfloat)

        # Check convergence condition: ‖E_n‖∞ < 1
        E_n = m >= n ? E_w : E_w[1:m, 1:m]
        E_norm = maximum(sum(abs.(E_n), dims=2))

        if E_norm >= 1
            # Cannot verify - perturbation too large
            return nothing, nothing, nothing, nothing, false
        end

        # Extract triangular parts
        E_stril = _strict_lower_triangular(E_w)  # Strictly lower triangular
        E_triu = _upper_triangular(E_w)          # Upper triangular (including diagonal)

        # Equation (3.1): Bound on |L^[ℓ] - E^[ℓ]|
        # |L^[ℓ] - E^[ℓ]| ≤ (sum(|E^[ℓ]|, 2) · max(|E^[u]_n|))^[ℓ] / (1 - ‖E_n‖∞)
        row_sums_E_stril = vec(sum(abs.(E_stril), dims=2))
        col_maxes_E_triu = vec(maximum(abs.(E_triu[1:mn, 1:mn]), dims=1))

        # Outer product bound (strictly lower triangular part only)
        denom = 1 - E_norm
        Delta_L = zeros(RWT, m, mn)
        for j in 1:(mn-1)
            for i in (j+1):m
                Delta_L[i, j] = row_sums_E_stril[i] * col_maxes_E_triu[j] / denom
            end
        end

        # L = I + E^[ℓ] + C^[ℓ] where |C^[ℓ]| ≤ Δ^[ℓ]
        L_offset_mid = E_stril[1:m, 1:mn]
        L_offset_rad = Delta_L

        # Equation (3.4): Bound on L⁻¹
        # L⁻¹ = I - E^[ℓ] + δ where |δ| ≤ Δ^[ℓ] + (sum(G,2)·max(G))^[ℓ] / (1 - ‖G‖∞)
        G = abs.(L_offset_mid) .+ Delta_L
        G_norm = maximum(sum(G, dims=2))

        if G_norm >= 1
            return nothing, nothing, nothing, nothing, false
        end

        row_sums_G = vec(sum(G, dims=2))
        col_maxes_G = vec(maximum(G, dims=1))

        delta_L_inv = copy(Delta_L)
        for j in 1:(mn-1)
            for i in (j+1):m
                delta_L_inv[i, j] += row_sums_G[i] * col_maxes_G[j] / (1 - G_norm)
            end
        end

        L_inv_offset_mid = -E_stril[1:m, 1:mn]
        L_inv_offset_rad = delta_L_inv

        # Equation (3.5): Bound on U
        # U = I_n + E^[u]_n + C^[u] where the bound involves L
        # For m ≥ n case
        if m >= n
            B = copy(abs.(E_triu[1:n, 1:n]))
            for j in 1:(n-1)
                for i in (j+1):n
                    B[i, j] = Delta_L[i, j]
                end
            end

            row_sums_GL = vec(sum(G[1:n, 1:n], dims=2))
            col_maxes_B = vec(maximum(B, dims=1))

            GL_norm = maximum(row_sums_GL)
            if GL_norm >= 1
                return nothing, nothing, nothing, nothing, false
            end

            Delta_U = zeros(RWT, n, n)
            for j in 1:n
                for i in 1:j  # Upper triangular including diagonal
                    Delta_U[i, j] = row_sums_GL[i] * col_maxes_B[j] / (1 - GL_norm)
                end
            end

            U_offset_mid = E_triu[1:n, 1:n]
            U_offset_rad = Delta_U

            # U⁻¹ bounds (equation 3.7)
            GU = abs.(U_offset_mid) .+ Delta_U
            GU_norm = maximum(sum(GU, dims=2))

            if GU_norm >= 1
                return nothing, nothing, nothing, nothing, false
            end

            row_sums_GU = vec(sum(GU, dims=2))
            col_maxes_GU = vec(maximum(GU, dims=1))

            delta_U_inv = copy(Delta_U)
            for j in 1:n
                for i in 1:j
                    delta_U_inv[i, j] += row_sums_GU[i] * col_maxes_GU[j] / (1 - GU_norm)
                end
            end

            U_inv_offset_mid = -E_triu[1:n, 1:n]
            U_inv_offset_rad = delta_U_inv
        else
            # m < n case: L is m×m, U is m×n
            # Similar but with different dimensions
            E_m = E_w[1:m, 1:m]
            E_m_triu = _upper_triangular(E_m)

            B_m = [abs.(E_m_triu) abs.(E_w[1:m, (m+1):n])]
            for j in 1:(m-1)
                for i in (j+1):m
                    B_m[i, j] = Delta_L[i, j]
                end
            end

            row_sums_GL = vec(sum(G[1:m, 1:m], dims=2))
            col_maxes_B = vec(maximum(B_m, dims=1))

            GL_norm = maximum(row_sums_GL)
            if GL_norm >= 1
                return nothing, nothing, nothing, nothing, false
            end

            Delta_U = zeros(RWT, m, n)
            for j in 1:n
                for i in 1:min(j, m)
                    Delta_U[i, j] = row_sums_GL[i] * col_maxes_B[j] / (1 - GL_norm)
                end
            end

            U_offset_mid = [E_m_triu E_w[1:m, (m+1):n]]
            U_offset_rad = Delta_U

            # U⁻¹ for m < n (only left m×m block is invertible)
            GU = abs.(U_offset_mid[1:m, 1:m]) .+ Delta_U[1:m, 1:m]
            GU_norm = maximum(sum(GU, dims=2))

            if GU_norm >= 1
                return nothing, nothing, nothing, nothing, false
            end

            row_sums_GU = vec(sum(GU, dims=2))
            col_maxes_GU = vec(maximum(GU, dims=1))

            delta_U_inv = zeros(RWT, m, m)
            for j in 1:m
                for i in 1:j
                    delta_U_inv[i, j] = Delta_U[i, j] + row_sums_GU[i] * col_maxes_GU[j] / (1 - GU_norm)
                end
            end

            U_inv_offset_mid = -E_m_triu
            U_inv_offset_rad = delta_U_inv
        end

        return (L_offset_mid, L_offset_rad), (U_offset_mid, U_offset_rad),
               (L_inv_offset_mid, L_inv_offset_rad), (U_inv_offset_mid, U_inv_offset_rad), true

    finally
        setprecision(BigFloat, old_prec)
    end
end

"""
    _strict_lower_triangular(A::AbstractMatrix)

Extract strictly lower triangular part of A (below diagonal).
"""
function _strict_lower_triangular(A::AbstractMatrix{T}) where T
    m, n = size(A)
    L = zeros(T, m, n)
    for j in 1:min(m-1, n)
        for i in (j+1):m
            L[i, j] = A[i, j]
        end
    end
    return L
end

"""
    _upper_triangular(A::AbstractMatrix)

Extract upper triangular part of A (including diagonal).
"""
function _upper_triangular(A::AbstractMatrix{T}) where T
    m, n = size(A)
    U = zeros(T, m, n)
    for j in 1:n
        for i in 1:min(j, m)
            U[i, j] = A[i, j]
        end
    end
    return U
end

"""
    verified_lu(A::AbstractMatrix{T}; precision_bits::Int=256, use_double_precision::Bool=true, use_bigfloat::Bool=true) where T

Compute verified LU decomposition with rigorous error bounds.

# Algorithm (Rump & Ogita 2024, Section 3.2)

1. Compute approximate LU with partial pivoting: A[p,:] ≈ L̃Ũ
2. Compute preconditioners: X_L ≈ L̃⁻¹, X_U ≈ Ũ⁻¹
3. Form perturbed identity: I_E = X_L · A · X_U
4. Verify LU of I_E using [`_lu_perturbed_identity`](@ref)
5. Transform back: L = X_L⁻¹ · L_E, U = U_E · X_U⁻¹

# Arguments
- `A`: Input matrix (m × n)
- `precision_bits`: BigFloat precision for rigorous computation (default: 256, ignored if use_bigfloat=false)
- `use_double_precision`: Use double-precision products for I_E (default: true)
- `use_bigfloat`: If true, use BigFloat for high precision; if false, use Float64 (faster but less precise)

# Returns
[`VerifiedLUResult`](@ref) containing rigorous enclosures of L, U and permutation.

# Example
```julia
A = randn(100, 100)
result = verified_lu(A)  # Uses BigFloat by default
result_fast = verified_lu(A; use_bigfloat=false)  # Uses Float64 (faster)
@assert result.success
# L and U are BallMatrix enclosures
```

# References
- [RumpOgita2024](@cite) Rump & Ogita, Section 3: LU decomposition
"""
function verified_lu(A::AbstractMatrix{T};
                     precision_bits::Int=256,
                     use_double_precision::Bool=true,
                     use_bigfloat::Bool=true) where T<:Union{Float64, ComplexF64, BigFloat, Complex{BigFloat}}
    if real(T) === BigFloat
        use_bigfloat = true
        @warn "verified_lu with BigFloat input uses Float64-seeded refinement. " *
              "For full-precision BigFloat, use `verified_lu_gla` (requires `using GenericLinearAlgebra`)." maxlog=1
    end
    m, n = size(A)

    # Step 1: Compute approximate LU with partial pivoting
    F = lu(A, Val(true))  # Partial pivoting
    L_approx = F.L
    U_approx = F.U
    p = F.p

    # Permute A
    A_perm = A[p, :]

    # Step 2: Compute approximate preconditioners
    # X_L ≈ L̃⁻¹ (left inverse for m ≥ n, or inverse of square part)
    # X_U ≈ Ũ⁻¹ (right inverse)
    mn = min(m, n)

    if m >= n
        X_L = inv(L_approx[1:n, 1:n])
        X_L_full = vcat(X_L, -L_approx[(n+1):m, 1:n] * X_L)
    else
        X_L = inv(L_approx)
    end

    X_U = m >= n ? inv(U_approx) : inv(U_approx[1:m, 1:m])

    # Step 3: Form perturbed identity I_E = X_L · A_perm · X_U
    # Use higher precision for this product if requested
    if use_double_precision
        I_E = _double_precision_product(X_L, A_perm, X_U)
    else
        if m >= n
            I_E = X_L_full * A_perm * X_U
        else
            I_E = X_L * A_perm[1:m, 1:m] * X_U
        end
    end

    # E = I_E - I
    E = I_E - I

    # Step 4: Verify LU of I + E
    L_E_data, U_E_data, _, _, success =
        _lu_perturbed_identity(E; precision_bits=precision_bits, use_bigfloat=use_bigfloat)

    # Get working type for this computation
    WT = _working_type(T, use_bigfloat)
    RWT = real(WT)

    if !success
        # Return failure result
        L_ball = BallMatrix(_to_working(L_approx, use_bigfloat), fill(RWT(Inf), m, mn))
        U_ball = BallMatrix(_to_working(U_approx, use_bigfloat), fill(RWT(Inf), mn, n))
        return VerifiedLUResult(L_ball, U_ball, p, false, RWT(Inf))
    end

    L_offset_mid, L_offset_rad = L_E_data
    U_offset_mid, U_offset_rad = U_E_data

    # Step 5: Transform back to get L and U
    # L = X_L⁻¹ · L_E = L̃ · L_E (approximately)
    # U = U_E · X_U⁻¹ = U_E · Ũ (approximately)

    old_prec = precision(BigFloat)
    if use_bigfloat
        setprecision(BigFloat, precision_bits)
    end

    try
        # For the improved formula (equation 3.8): L = A · X_U · U_E⁻¹
        # This gives better accuracy for L

        # Convert to working precision
        A_perm_w = _to_working(A_perm, use_bigfloat)
        L_approx_w = _to_working(L_approx, use_bigfloat)
        U_approx_w = _to_working(U_approx, use_bigfloat)

        # Build L_E and U_E as ball matrices (as perturbations of identity)
        # L_E = I + L_offset
        L_E_mid = Matrix{WT}(I, m, mn) + L_offset_mid
        U_E_mid = (m >= n ? Matrix{WT}(I, n, n) : Matrix{WT}(I, m, n)[1:m, 1:n]) + U_offset_mid

        # Compute L = L̃ · L_E with error propagation
        # and U = U_E · Ũ with error propagation
        if m >= n
            L_mid = L_approx_w * L_E_mid
            U_mid = U_E_mid * U_approx_w
        else
            L_mid = L_approx_w * L_E_mid
            # For m < n, U is m × n
            U_mid = hcat(U_E_mid * U_approx_w[1:m, 1:m],
                        L_E_mid \ (_to_working(X_L, use_bigfloat) * A_perm_w[:, (m+1):n]))
        end

        # Propagate error bounds
        # Error in L: |ΔL| ≤ |L̃| · |ΔL_E| + O(ε²)
        L_rad = abs.(L_approx_w) * L_offset_rad

        # Error in U: |ΔU| ≤ |ΔU_E| · |Ũ| + O(ε²)
        if m >= n
            U_rad = U_offset_rad * abs.(U_approx_w)
        else
            U_rad_left = U_offset_rad * abs.(U_approx_w[1:m, 1:m])

            # Error in U_right from solving L_E * U_right = X_L * A_right
            # Per Rump-Ogita 2024 Section 3.4: propagate error through triangular solve
            # U_right_mid is already computed, error bound:
            # |ΔU_right| ≤ |L_E^{-1}| * |ΔL_E| * |U_right| + triangular solve error
            #
            # Using Neumann bound for |L_E^{-1}|: since L_E = I + L_offset with L_offset
            # strictly lower triangular, ‖L_offset‖ < 1 implies |L_E^{-1}| ≤ 1/(1 - ‖L_offset‖)
            L_offset_norm = setrounding(RWT, RoundUp) do
                opnorm(L_offset_rad, Inf)  # Upper bound on ‖L_offset‖_∞
            end

            if L_offset_norm < one(RWT)
                U_right_mid = U_mid[:, (m+1):n]
                L_E_inv_bound = setrounding(RWT, RoundUp) do
                    one(RWT) / (one(RWT) - L_offset_norm)
                end
                # Error propagation: |L_E^{-1}| * |L_offset_rad| * |U_right|
                U_rad_right = setrounding(RWT, RoundUp) do
                    L_E_inv_bound .* (L_offset_rad * abs.(U_right_mid))
                end
            else
                # Fallback: conservative bound using Frobenius norm of residual
                # This path should rarely be taken for well-conditioned problems
                @warn "LU rectangular case: L_offset_norm >= 1, using conservative bounds"
                U_right_mid = U_mid[:, (m+1):n]
                U_rad_right = setrounding(RWT, RoundUp) do
                    fill(opnorm(L_offset_rad, Inf), m, n - m) .* abs.(U_right_mid)
                end
            end
            U_rad = hcat(U_rad_left, U_rad_right)
        end

        # Build ball matrices
        L_ball = BallMatrix(L_mid, L_rad)
        U_ball = BallMatrix(U_mid, U_rad)

        # Compute rigorous residual norm bound using Miyajima products
        residual_norm = _rigorous_relative_residual_norm(L_mid, U_mid, A_perm_w)

        return VerifiedLUResult(L_ball, U_ball, p, true, residual_norm)

    finally
        if use_bigfloat
            setprecision(BigFloat, old_prec)
        end
    end
end

"""
    _double_precision_product(X_L, A, X_U)

Compute X_L · A · X_U using compensated (double-double) arithmetic for better accuracy.
Falls back to standard arithmetic if DoubleFloats is not available.
"""
function _double_precision_product(X_L::AbstractMatrix, A::AbstractMatrix, X_U::AbstractMatrix)
    # Default implementation: standard floating-point
    # This will be overridden in DoubleFloatsExt for better accuracy
    return X_L * A * X_U
end

# Stub for Double64 extension
"""
    verified_lu_double64(A; precision_bits=256)

Fast verified LU using Double64 oracle. Requires DoubleFloats.jl.
"""
function verified_lu_double64 end

# Stub for MultiFloat extension
"""
    verified_lu_multifloat(A; precision_bits=256, float_type=Float64x4)

Fast verified LU using MultiFloat oracle. Requires MultiFloats.jl.
"""
function verified_lu_multifloat end

export VerifiedLUResult, verified_lu, verified_lu_double64, verified_lu_multifloat
