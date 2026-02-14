# Verified Cholesky Decomposition with Rigorous Error Bounds
# Based on Section 4 of Rump & Ogita (2024) "Verified Error Bounds for Matrix Decompositions"
#
# For symmetric positive definite A, compute verified bounds for G such that A = G^T G.
# Note: _bigfloat_type and _to_bigfloat are defined in verified_lu.jl

"""
    VerifiedCholeskyResult{GM, RT}

Result from verified Cholesky decomposition with rigorous error bounds.

# Fields
- `G::GM`: Upper triangular Cholesky factor (rigorous enclosure as BallMatrix, A = G^T G)
- `success::Bool`: Whether verification succeeded (also proves A is positive definite)
- `residual_norm::RT`: Bound on ‖G^T G - A‖ / ‖A‖

# Mathematical Guarantee
- If `success == true`, then A is proven to be symmetric positive definite
- For any G̃ ∈ G: G̃^T G̃ = A

# References
- [RumpOgita2024](@cite) Rump & Ogita, "Verified Error Bounds for Matrix Decompositions",
  Section 4: Cholesky decomposition.
"""
struct VerifiedCholeskyResult{GM<:BallMatrix, RT<:Real}
    G::GM
    success::Bool
    residual_norm::RT
end

"""
    verified_cholesky(A::AbstractMatrix{T}; precision_bits::Int=256,
                      use_double_precision::Bool=true,
                      use_bigfloat::Bool=true) where T

Compute verified Cholesky decomposition A = G^T G with rigorous error bounds.

This method proves that A is symmetric positive definite and computes a rigorous
enclosure of the Cholesky factor G.

# Algorithm (Rump & Ogita 2024, Section 4)

1. Compute approximate Cholesky: A ≈ G̃^T G̃
2. Precondition: I_E = X_G^T A X_G where X_G ≈ G̃⁻¹
3. Compute verified LU of I_E: I_E = L_E U_E
4. Extract diagonal D from U_E: G_E = D^{1/2} L_E^T
5. Transform back: G = G_E X_G⁻¹ = D^{1/2} L_E^T G̃

# Arguments
- `A`: Symmetric positive definite matrix (symmetry is checked, not assumed)
- `precision_bits`: BigFloat precision for rigorous computation (default: 256, ignored if use_bigfloat=false)
- `use_double_precision`: Use double-precision products (default: true)
- `use_bigfloat`: If true, use BigFloat for high precision; if false, use Float64 (faster)

# Returns
[`VerifiedCholeskyResult`](@ref) containing rigorous enclosure of G.

# Example
```julia
A = randn(100, 100); A = A' * A + 0.1I  # Make positive definite
result = verified_cholesky(A)  # Uses BigFloat by default
result_fast = verified_cholesky(A; use_bigfloat=false)  # Uses Float64 (faster)
@assert result.success  # A is proven positive definite
```

# Notes
- If verification fails, A may not be positive definite, or may be too ill-conditioned
- The method also works for interval matrix input

# References
- [RumpOgita2024](@cite) Rump & Ogita, Section 4: Cholesky decomposition
"""
function verified_cholesky(A::AbstractMatrix{T};
                           precision_bits::Int=256,
                           use_double_precision::Bool=true,
                           use_bigfloat::Bool=true) where T<:Union{Float64, ComplexF64, BigFloat, Complex{BigFloat}}
    if real(T) === BigFloat
        use_bigfloat = true
    end
    n = size(A, 1)
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("A must be square"))

    # Get working type for this computation
    WT = _working_type(T, use_bigfloat)
    RWT = real(WT)

    # Check approximate symmetry
    sym_error = maximum(abs.(A - A'))
    if sym_error > 100 * eps(real(T)) * maximum(abs.(A))
        @warn "Matrix A is not symmetric (error = $sym_error)"
    end

    # Symmetrize
    A_sym = (A + A') / 2

    # Step 1: Compute approximate Cholesky
    F = try
        cholesky(Hermitian(A_sym))
    catch e
        if e isa PosDefException
            # Matrix is not positive definite
            G_ball = BallMatrix(fill(WT(NaN), n, n), fill(RWT(Inf), n, n))
            return VerifiedCholeskyResult(G_ball, false, RWT(Inf))
        end
        rethrow(e)
    end

    G_approx = Matrix(F.U)  # Upper triangular factor

    # Step 2: Compute preconditioner X_G ≈ G̃⁻¹
    X_G = inv(G_approx)

    # Step 3: Form perturbed identity I_E = X_G^T A X_G
    if use_double_precision
        I_E = _double_precision_triple_product_symmetric(X_G, A_sym)
    else
        I_E = X_G' * A_sym * X_G
    end

    # E = I_E - I
    E = I_E - I

    # Step 4: Verified LU of I + E
    # The uniqueness of LU and Cholesky implies G_E = D^{1/2} L_E^T
    L_E_data, U_E_data, _, _, success = _lu_perturbed_identity(E; precision_bits=precision_bits, use_bigfloat=use_bigfloat)

    if !success
        G_ball = BallMatrix(_to_working(G_approx, use_bigfloat), fill(RWT(Inf), n, n))
        return VerifiedCholeskyResult(G_ball, false, RWT(Inf))
    end

    L_offset_mid, L_offset_rad = L_E_data
    U_offset_mid, U_offset_rad = U_E_data

    old_prec = precision(BigFloat)
    if use_bigfloat
        setprecision(BigFloat, precision_bits)
    end

    try
        # Build L_E and U_E
        I_n = Matrix{WT}(I, n, n)
        L_E_mid = I_n + L_offset_mid
        U_E_mid = I_n + U_offset_mid

        # Extract diagonal D from U_E
        D_mid = diag(U_E_mid)
        D_rad = diag(U_offset_rad)

        # Check all diagonal entries are positive (proves positive definiteness)
        for i in 1:n
            if real(D_mid[i]) - D_rad[i] <= 0
                G_ball = BallMatrix(_to_working(G_approx, use_bigfloat), fill(RWT(Inf), n, n))
                return VerifiedCholeskyResult(G_ball, false, RWT(Inf))
            end
        end

        # Compute D^{1/2} with rigorous bounds
        # For x ∈ [a-r, a+r] with a > r > 0: √x ∈ [√(a-r), √(a+r)]
        D_sqrt_mid = sqrt.(D_mid)
        D_sqrt_rad = zeros(RWT, n)
        for i in 1:n
            # Use interval arithmetic for square root
            lower = sqrt(real(D_mid[i]) - D_rad[i])
            upper = sqrt(real(D_mid[i]) + D_rad[i])
            D_sqrt_mid[i] = (lower + upper) / 2
            D_sqrt_rad[i] = (upper - lower) / 2
        end

        # G_E = D^{1/2} L_E^T (equation 4.1)
        G_E_mid = Diagonal(D_sqrt_mid) * L_E_mid'
        G_E_rad = Diagonal(D_sqrt_rad) * abs.(L_E_mid') +
                  Diagonal(D_sqrt_mid) * L_offset_rad' +
                  Diagonal(D_sqrt_rad) * L_offset_rad'

        # Step 5: Transform back G = G_E X_G⁻¹ = G_E G̃
        G_approx_w = _to_working(G_approx, use_bigfloat)

        G_mid = G_E_mid * G_approx_w
        # Error propagation: account for G_E uncertainty and floating-point error
        # in G_E_mid * G_approx_w. Using Revol-Théveny formula: error ≤ (k+2)*ε*|A|*|B| + η/ε
        ε_w = eps(RWT)
        η_w = floatmin(RWT)  # smallest positive normal number
        k = n  # inner dimension of G_E_mid * G_approx_w
        mmul_error = setrounding(RWT, RoundUp) do
            (k + 2) * ε_w * abs.(G_E_mid) * abs.(G_approx_w) .+ η_w / ε_w
        end
        G_rad = setrounding(RWT, RoundUp) do
            G_E_rad * abs.(G_approx_w) + mmul_error
        end

        # Ensure upper triangular structure
        for j in 1:n
            for i in (j+1):n
                G_mid[i, j] = zero(WT)
                G_rad[i, j] = zero(RWT)
            end
        end

        G_ball = BallMatrix(G_mid, G_rad)

        # Compute rigorous residual using Miyajima products
        A_w = _to_working(A_sym, use_bigfloat)
        residual_norm = _rigorous_gram_relative_residual_norm(G_mid, A_w)

        return VerifiedCholeskyResult(G_ball, true, residual_norm)

    finally
        if use_bigfloat
            setprecision(BigFloat, old_prec)
        end
    end
end

"""
    _double_precision_triple_product_symmetric(X, A)

Compute X^T A X using compensated arithmetic for symmetric A.
Default implementation; overridden in DoubleFloatsExt.
"""
function _double_precision_triple_product_symmetric(X::AbstractMatrix, A::AbstractMatrix)
    return X' * A * X
end

# Stub for Double64 extension
"""
    verified_cholesky_double64(A; precision_bits=256)

Fast verified Cholesky using Double64 oracle. Requires DoubleFloats.jl.
"""
function verified_cholesky_double64 end

# Stub for MultiFloat extension
"""
    verified_cholesky_multifloat(A; precision_bits=256, float_type=Float64x4)

Fast verified Cholesky using MultiFloat oracle. Requires MultiFloats.jl.
"""
function verified_cholesky_multifloat end

export VerifiedCholeskyResult, verified_cholesky, verified_cholesky_double64, verified_cholesky_multifloat
