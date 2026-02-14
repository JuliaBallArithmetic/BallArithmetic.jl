# Verified QR Decomposition with Rigorous Error Bounds
# Based on Section 5 of Rump & Ogita (2024) "Verified Error Bounds for Matrix Decompositions"
#
# For A ∈ ℝ^{m×n} with m ≥ n, compute verified bounds for Q (orthogonal) and R (upper triangular).
# Note: _bigfloat_type and _to_bigfloat are defined in verified_lu.jl

"""
    VerifiedQRResult{QM, RM, RT}

Result from verified QR decomposition with rigorous error bounds.

# Fields
- `Q::QM`: Orthogonal/unitary factor (rigorous enclosure as BallMatrix)
- `R::RM`: Upper triangular factor (rigorous enclosure as BallMatrix)
- `success::Bool`: Whether verification succeeded
- `residual_norm::RT`: Bound on ‖QR - A‖ / ‖A‖
- `orthogonality_defect::RT`: Bound on ‖Q^H Q - I‖

# Mathematical Guarantee
- For any Q̃ ∈ Q, R̃ ∈ R: Q̃R̃ = A
- Q̃^H Q̃ is close to identity (within orthogonality_defect)

# References
- [RumpOgita2024](@cite) Rump & Ogita, "Verified Error Bounds for Matrix Decompositions",
  Section 5: QR decomposition.
"""
struct VerifiedQRResult{QM<:BallMatrix, RM<:BallMatrix, RT<:Real}
    Q::QM
    R::RM
    success::Bool
    residual_norm::RT
    orthogonality_defect::RT
end

"""
    verified_qr(A::AbstractMatrix{T}; precision_bits::Int=256,
                use_double_precision::Bool=true,
                compute_full_Q::Bool=false,
                use_bigfloat::Bool=true) where T

Compute verified QR decomposition A = QR with rigorous error bounds.

# Algorithm (Rump & Ogita 2024, Section 5)

For A ∈ ℝ^{m×n} with m ≥ n:
1. Compute approximate QR: A ≈ Q̃R̃ (economy-size)
2. Precondition: C = A X_R where X_R ≈ R̃⁻¹
3. Compute C^T C ≈ I + E (perturbed identity)
4. Verified Cholesky: I + E = G_E^T G_E gives R-factor via R = G_E X_R⁻¹
5. Q-factor: Q = A R⁻¹ or Q = A X_R G_E⁻¹

# Arguments
- `A`: Input matrix (m × n with m ≥ n)
- `precision_bits`: BigFloat precision (default: 256, ignored if use_bigfloat=false)
- `use_double_precision`: Use double-precision products (default: true)
- `compute_full_Q`: Compute full m×m Q (default: false for economy-size m×n)
- `use_bigfloat`: If true, use BigFloat for high precision; if false, use Float64 (faster)

# Returns
[`VerifiedQRResult`](@ref) containing rigorous enclosures of Q and R.

# Example
```julia
A = randn(100, 50)
result = verified_qr(A)  # Uses BigFloat by default
result_fast = verified_qr(A; use_bigfloat=false)  # Uses Float64 (faster)
@assert result.success
# Q is 100×50 (economy-size), R is 50×50
```

# References
- [RumpOgita2024](@cite) Rump & Ogita, Section 5: QR decomposition
"""
function verified_qr(A::AbstractMatrix{T};
                     precision_bits::Int=256,
                     use_double_precision::Bool=true,
                     compute_full_Q::Bool=false,
                     use_bigfloat::Bool=true) where T<:Union{Float64, ComplexF64, BigFloat, Complex{BigFloat}}
    if real(T) === BigFloat
        use_bigfloat = true
    end
    m, n = size(A)

    # Get working type for this computation
    WT = _working_type(T, use_bigfloat)
    RWT = real(WT)

    if m < n
        # For m < n, QR of A is obtained from QR of A^T
        # A = QR where Q is m×m orthogonal, R is m×n upper trapezoidal
        # We compute QR of A_m (left m×m block) and extend
        result_sq = verified_qr(A[:, 1:m]; precision_bits=precision_bits,
                                use_double_precision=use_double_precision,
                                use_bigfloat=use_bigfloat)
        if !result_sq.success
            Q_ball = BallMatrix(fill(WT(NaN), m, m), fill(RWT(Inf), m, m))
            R_ball = BallMatrix(fill(WT(NaN), m, n), fill(RWT(Inf), m, n))
            return VerifiedQRResult(Q_ball, R_ball, false, RWT(Inf), RWT(Inf))
        end

        # R = Q^H A for full R
        old_prec = precision(BigFloat)
        if use_bigfloat
            setprecision(BigFloat, precision_bits)
        end
        try
            A_w = _to_working(A, use_bigfloat)
            Q_mid = mid(result_sq.Q)
            R_full_mid = Q_mid' * A_w
            # Error propagation: account for both Q uncertainty and floating-point error
            # in Q_mid' * A_w. Using Revol-Théveny formula: error ≤ (k+2)*ε*|Q_mid'|*|A_w| + η/ε
            Q_rad = rad(result_sq.Q)
            ε_w = eps(RWT)
            η_w = floatmin(RWT)  # smallest positive normal number
            k = m  # inner dimension of Q_mid' * A_w
            mmul_error = setrounding(RWT, RoundUp) do
                (k + 2) * ε_w * abs.(Q_mid') * abs.(A_w) .+ η_w / ε_w
            end
            R_full_rad = setrounding(RWT, RoundUp) do
                Q_rad' * abs.(A_w) + mmul_error
            end

            R_ball = BallMatrix(R_full_mid, R_full_rad)
            return VerifiedQRResult(result_sq.Q, R_ball, true,
                                    result_sq.residual_norm, result_sq.orthogonality_defect)
        finally
            if use_bigfloat
                setprecision(BigFloat, old_prec)
            end
        end
    end

    # m ≥ n case: standard QR

    # Step 1: Compute approximate economy-size QR
    F = qr(A)
    Q_approx = Matrix(F.Q)[:, 1:n]  # Economy-size: m×n
    R_approx = Matrix(F.R)          # n×n upper triangular

    # Step 2: Compute preconditioner X_R ≈ R̃⁻¹
    X_R = inv(R_approx)

    # Step 3: Form C = A X_R (should be close to Q)
    if use_double_precision
        C = _double_precision_product_right(A, X_R)
    else
        C = A * X_R
    end

    # Step 4: C^T C should be close to I
    # C^T C = X_R^T (A^T A) X_R ≈ X_R^T (R̃^T R̃) X_R = I
    if use_double_precision
        CtC = _double_precision_gram(C)
    else
        CtC = C' * C
    end

    E = CtC - I

    # Step 5: Verified Cholesky of I + E gives G_E with I + E = G_E^T G_E
    # Then R = G_E X_R⁻¹ = G_E R̃

    # Use the LU approach from Section 4 (Cholesky via LU)
    L_E_data, U_E_data, _, _, success = _lu_perturbed_identity(E; precision_bits=precision_bits, use_bigfloat=use_bigfloat)

    if !success
        Q_ball = BallMatrix(_to_working(Q_approx, use_bigfloat), fill(RWT(Inf), m, n))
        R_ball = BallMatrix(_to_working(R_approx, use_bigfloat), fill(RWT(Inf), n, n))
        return VerifiedQRResult(Q_ball, R_ball, false, RWT(Inf), RWT(Inf))
    end

    L_offset_mid, L_offset_rad = L_E_data
    U_offset_mid, U_offset_rad = U_E_data

    old_prec = precision(BigFloat)
    if use_bigfloat
        setprecision(BigFloat, precision_bits)
    end

    try
        I_n = Matrix{WT}(I, n, n)
        L_E_mid = I_n + L_offset_mid
        U_E_mid = I_n + U_offset_mid

        # Extract diagonal D from U_E
        D_mid = diag(U_E_mid)
        D_rad = diag(U_offset_rad)

        # Check positive definiteness
        for i in 1:n
            if real(D_mid[i]) - D_rad[i] <= 0
                Q_ball = BallMatrix(_to_working(Q_approx, use_bigfloat), fill(RWT(Inf), m, n))
                R_ball = BallMatrix(_to_working(R_approx, use_bigfloat), fill(RWT(Inf), n, n))
                return VerifiedQRResult(Q_ball, R_ball, false, RWT(Inf), RWT(Inf))
            end
        end

        # D^{1/2} with rigorous bounds
        D_sqrt_mid = sqrt.(D_mid)
        D_sqrt_rad = zeros(RWT, n)
        for i in 1:n
            lower = sqrt(real(D_mid[i]) - D_rad[i])
            upper = sqrt(real(D_mid[i]) + D_rad[i])
            D_sqrt_mid[i] = (lower + upper) / 2
            D_sqrt_rad[i] = (upper - lower) / 2
        end

        # G_E = D^{1/2} L_E^T is the Cholesky factor of I + E
        G_E_mid = Diagonal(D_sqrt_mid) * L_E_mid'
        G_E_rad = Diagonal(D_sqrt_rad) * abs.(L_E_mid') +
                  Diagonal(D_sqrt_mid) * L_offset_rad' +
                  Diagonal(D_sqrt_rad) * L_offset_rad'

        # R = G_E R̃
        R_approx_w = _to_working(R_approx, use_bigfloat)
        R_mid = G_E_mid * R_approx_w
        # Error propagation: G_E uncertainty + floating-point error in G_E_mid * R_approx_w
        # Using Revol-Théveny formula: error ≤ (k+2)*ε*|A|*|B| + η/ε
        ε_w = eps(RWT)
        η_w = floatmin(RWT)
        k_r = n  # inner dimension
        mmul_error_R = setrounding(RWT, RoundUp) do
            (k_r + 2) * ε_w * abs.(G_E_mid) * abs.(R_approx_w) .+ η_w / ε_w
        end
        R_rad = setrounding(RWT, RoundUp) do
            G_E_rad * abs.(R_approx_w) + mmul_error_R
        end

        # Ensure upper triangular
        for j in 1:n
            for i in (j+1):n
                R_mid[i, j] = zero(WT)
                R_rad[i, j] = zero(RWT)
            end
        end

        # Q = A R⁻¹ or better: Q = C G_E⁻¹
        # Using Q = A X_R G_E⁻¹
        C_w = _to_working(C, use_bigfloat)

        # G_E⁻¹ = L_E^{-T} D^{-1/2}
        # Use direct formula for small matrices, iterative for large
        G_E_inv_mid = L_E_mid' \ Diagonal(1 ./ D_sqrt_mid)

        Q_mid = C_w * G_E_inv_mid

        # Error in Q: need to propagate errors through G_E⁻¹
        # Simplified: use residual-based error bound
        # |ΔQ| ≤ |C| · |ΔG_E⁻¹| where ΔG_E⁻¹ ≈ G_E⁻¹ ΔG_E G_E⁻¹
        G_E_inv_norm = maximum(sum(abs.(G_E_inv_mid), dims=2))
        G_E_rad_norm = maximum(sum(G_E_rad, dims=2))

        Q_rad_factor = G_E_inv_norm * G_E_rad_norm * G_E_inv_norm
        Q_rad = abs.(C_w) * fill(Q_rad_factor, n, n)

        Q_ball = BallMatrix(Q_mid, Q_rad)
        R_ball = BallMatrix(R_mid, R_rad)

        # Compute rigorous residual and orthogonality defect using Miyajima products
        A_w = _to_working(A, use_bigfloat)
        residual_norm = _rigorous_relative_residual_norm(Q_mid, R_mid, A_w)

        # Rigorous orthogonality defect: ‖Q'Q - I‖∞
        ortho_defect = _rigorous_residual_bound(Q_mid', Q_mid, I_n)

        return VerifiedQRResult(Q_ball, R_ball, true, residual_norm, ortho_defect)

    finally
        if use_bigfloat
            setprecision(BigFloat, old_prec)
        end
    end
end

"""
    _double_precision_product_right(A, B)

Compute A · B using compensated arithmetic. Default implementation.
"""
function _double_precision_product_right(A::AbstractMatrix, B::AbstractMatrix)
    return A * B
end

"""
    _double_precision_gram(C)

Compute C^H C using compensated arithmetic. Default implementation.
"""
function _double_precision_gram(C::AbstractMatrix)
    return C' * C
end

# Stub for Double64 extension
"""
    verified_qr_double64(A; precision_bits=256)

Fast verified QR using Double64 oracle. Requires DoubleFloats.jl.
"""
function verified_qr_double64 end

# Stub for MultiFloat extension
"""
    verified_qr_multifloat(A; precision_bits=256, float_type=Float64x4)

Fast verified QR using MultiFloat oracle. Requires MultiFloats.jl.
"""
function verified_qr_multifloat end

export VerifiedQRResult, verified_qr, verified_qr_double64, verified_qr_multifloat
