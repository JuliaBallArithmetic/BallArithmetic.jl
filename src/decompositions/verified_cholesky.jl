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
- `residual_norm::RT`: rigorous upper bound on ‖G^T G - A‖₂ / ‖A‖₂ for the
  *enclosure* `G`. The numerator is bounded with [`upper_bound_L2_opnorm`](@ref)
  over the ball product; the denominator uses the rigorous lower bound
  ‖A‖₂ ≥ maxᵢ Aᵢᵢ, valid for Hermitian positive definite A.

# Mathematical Guarantee
- If `success == true`, then A is proven to be symmetric positive definite
- There exists G̃ ∈ G with G̃^T G̃ = A (`A` here being the symmetrised (A + A*)/2)

# References
- [RumpOgita2024](@cite) Rump & Ogita, "Verified Error Bounds for Matrix Decompositions",
  Section 4: Cholesky decomposition.
"""
struct VerifiedCholeskyResult{GM <: BallMatrix, RT <: Real}
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
2. Precondition with X_G ≈ G̃⁻¹: I_E = X_G* A X_G, enclosed by ball products
3. Compute verified LU of the interval matrix I_E: I_E = L_E U_E
4. Extract diagonal D from U_E: G_E = D^{1/2} L_E^T
5. Transform back: G = G_E X_G⁻¹ = G_E (I + R + T) G̃

Every step is carried out in ball arithmetic, so the perturbation `E = I_E - I`
handed to the verified LU is an interval matrix rather than a floating-point
approximation treated as exact.

Step 5 does *not* evaluate `X_G⁻¹` as `G̃` — that identity holds only for an exact
inverse, and assuming it loses `~eps·κ(G̃)`. Instead `X_G` is treated as *free*
in the sense of Miyajima–Rump: it never has to be the exact inverse, because the
deviation enters only through the certified residual `R = I - G̃X_G`, giving
`X_G⁻¹ = (I - R)⁻¹G̃` with the Neumann tail `T` bounded by
`‖R‖₂²/(1 - ‖R‖₂)`. No inverse is ever formed and everything stays at BLAS
speed; `‖R‖₂ < 1` also proves `G̃` and `X_G` nonsingular.

# Arguments
- `A`: Symmetric positive definite matrix (symmetry is checked, not assumed)
- `precision_bits`: BigFloat precision for rigorous computation (default: 256, ignored if use_bigfloat=false)
- `use_double_precision`: retained for backwards compatibility; it no longer has
  any effect, as the preconditioned product is enclosed with ball arithmetic
  instead of a compensated triple product.
- `use_bigfloat`: If true, use BigFloat for high precision; if false, use Float64
  (faster, but the factor enclosure is many orders of magnitude wider)

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
        precision_bits::Int = 256,
        use_double_precision::Bool = true,
        use_bigfloat::Bool = true) where {T <: Union{
        Float64, ComplexF64, BigFloat, Complex{BigFloat}}}
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

    # Symmetrize.  The halved sum is formed with ball arithmetic as well, so the
    # rounding of aᵢⱼ + aⱼᵢ is enclosed rather than dropped (it is exact when the
    # input is already Hermitian, but we do not rely on that).
    A_sym = (A + A') / 2
    A_sym_ball = let Aw = _to_working(A, use_bigfloat)
        Ab = BallMatrix(Aw)
        (Ab + BallMatrix(collect(Aw'))) * Ball(one(RWT) / 2, zero(RWT))
    end

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

    # Step 2: preconditioner X_G ≈ G̃⁻¹.
    #
    # X_G is *free* in the sense of Miyajima–Rump: it is never required to be the
    # exact inverse of G̃, because the deviation enters only through the certified
    # residual R = I - G̃X_G in step 5.  So it is computed in floating point and,
    # in the BigFloat path, sharpened by one Newton step X ← X(2I - G̃X) in point
    # arithmetic, which squares ‖R‖ for two matrix products.
    G_approx_w = _to_working(G_approx, use_bigfloat)
    X_G_w = _to_working(inv(G_approx), use_bigfloat)
    if use_bigfloat
        X_G_w = X_G_w * (2 * one(X_G_w) - G_approx_w * X_G_w)
    end

    # Step 3: Form the perturbed identity I_E = X_G* A X_G as a rigorous
    # *enclosure*.  The ball products account for every rounding error of the
    # triple product, so the perturbation handed to the verified LU is an
    # interval matrix rather than a floating-point approximation treated as
    # exact.  (Treating it as exact used to lose ~eps‖X_G‖²‖A‖, and the returned
    # factor then failed to enclose the true one.)
    X_G_ball = BallMatrix(X_G_w)
    I_E_ball = (BallMatrix(collect(X_G_w')) * A_sym_ball) * X_G_ball

    # E = I_E - I, as an enclosure
    E_ball = I_E_ball - BallMatrix(Matrix{WT}(I, n, n))

    # Step 4: Verified LU of I + E over the whole interval matrix E
    # The uniqueness of LU and Cholesky implies G_E = D^{1/2} L_E^T
    L_E_data, U_E_data, _, _, success = _lu_perturbed_identity(E_ball.c;
        E_rad = E_ball.r, precision_bits = precision_bits, use_bigfloat = use_bigfloat)

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
        # Build L_E and U_E as enclosures.  Adding the identity is done with ball
        # arithmetic so that 1 + uᵢᵢ is enclosed rather than silently rounded.
        I_ball = BallMatrix(Matrix{WT}(I, n, n))
        L_E_ball = BallMatrix(L_offset_mid, L_offset_rad) + I_ball
        U_E_ball = BallMatrix(U_offset_mid, U_offset_rad) + I_ball

        # Extract diagonal D from U_E
        D = [Ball(U_E_ball.c[i, i], U_E_ball.r[i, i]) for i in 1:n]

        # Check all diagonal entries are positive (proves positive definiteness)
        for i in 1:n
            if real(D[i].c) - D[i].r <= 0
                G_ball = BallMatrix(_to_working(G_approx, use_bigfloat), fill(RWT(Inf), n, n))
                return VerifiedCholeskyResult(G_ball, false, RWT(Inf))
            end
        end

        # Compute D^{1/2} with the rigorous ball square root
        D_sqrt = [sqrt(Ball(real(D[i].c), D[i].r)) for i in 1:n]
        D_sqrt_ball = BallMatrix(diagm([WT(b.c) for b in D_sqrt]),
            diagm([b.r for b in D_sqrt]))

        # G_E = D^{1/2} L_E^T (equation 4.1), via the rigorous ball product
        L_E_adj = BallMatrix(collect(L_E_ball.c'), collect(L_E_ball.r'))
        G_E_ball = D_sqrt_ball * L_E_adj

        # Step 5: Transform back G = G_E X_G⁻¹.
        #
        # X_G⁻¹ is not G̃ — writing it as such is what used to lose ~eps·κ(G̃).
        # Instead use the Miyajima–Rump inversion bound and never form an
        # inverse: with the certified residual R = I - G̃X_G,
        #
        #     G̃X_G = I - R   ⟹   X_G⁻¹ = (I - R)⁻¹ G̃
        #     (I - R)⁻¹ = I + R + R²(I - R)⁻¹,   ‖R²(I-R)⁻¹‖₂ ≤ ‖R‖₂²/(1 - ‖R‖₂)
        #
        # so the Neumann tail is enclosed by a ball matrix of that uniform radius
        # (|Tᵢⱼ| = |eᵢ*Teⱼ| ≤ ‖T‖₂).  ‖R‖₂ < 1 simultaneously proves G̃ and X_G
        # nonsingular.  Everything here is a BLAS-level ball product.
        G_tilde_ball = BallMatrix(G_approx_w)
        R_ball = I_ball - G_tilde_ball * X_G_ball
        nR = upper_bound_L2_opnorm(R_ball)
        if !(nR < 1)
            G_fail = BallMatrix(G_approx_w, fill(RWT(Inf), n, n))
            return VerifiedCholeskyResult(G_fail, false, RWT(Inf))
        end
        tail = let denom = setrounding(RWT, RoundDown) do
                one(RWT) - nR
            end
            setrounding(RWT, RoundUp) do
                (nR * nR) / denom
            end
        end
        neumann = R_ball + BallMatrix(Matrix{WT}(I, n, n), fill(tail, n, n))
        G_ball = (G_E_ball * neumann) * G_tilde_ball

        # Enforce the upper triangular structure.  The exact Cholesky factor is
        # upper triangular by definition, so the strictly lower part is known to
        # be zero and may be set as such.  The Neumann correction above is a full
        # matrix, though, so this is no longer a no-op: check that zero really is
        # inside each of those enclosures, and fail rather than silently tighten
        # if it is not — that would mean the reconstruction is inconsistent.
        G_mid = copy(G_ball.c)
        G_rad = copy(G_ball.r)
        for j in 1:n
            for i in (j + 1):n
                if !(abs(G_mid[i, j]) <= G_rad[i, j])
                    G_fail = BallMatrix(G_approx_w, fill(RWT(Inf), n, n))
                    return VerifiedCholeskyResult(G_fail, false, RWT(Inf))
                end
                G_mid[i, j] = zero(WT)
                G_rad[i, j] = zero(RWT)
            end
        end
        G_ball = BallMatrix(G_mid, G_rad)

        # Rigorous relative residual of the enclosure itself: an upper bound on
        # ‖G*G - A‖₂ over a rigorous *lower* bound on ‖A‖₂.  For Hermitian
        # positive definite A, ‖A‖₂ ≥ maxᵢ Aᵢᵢ = maxᵢ eᵢ*Aeᵢ.
        G_adj = BallMatrix(collect(G_ball.c'), collect(G_ball.r'))
        residual_ball = G_adj * G_ball - A_sym_ball
        residual_abs = upper_bound_L2_opnorm(residual_ball)
        A_norm_lower = setrounding(RWT, RoundDown) do
            maximum(i -> real(A_sym_ball.c[i, i]) - A_sym_ball.r[i, i], 1:n)
        end
        residual_norm = if A_norm_lower > 0
            setrounding(RWT, RoundUp) do
                residual_abs / A_norm_lower
            end
        else
            RWT(Inf)
        end

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

export VerifiedCholeskyResult, verified_cholesky, verified_cholesky_double64,
       verified_cholesky_multifloat
