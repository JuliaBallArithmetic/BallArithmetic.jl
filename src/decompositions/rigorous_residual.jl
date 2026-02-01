# Rigorous residual computation for verified decompositions
# This module provides helper functions for computing rigorous bounds on
# decomposition residuals like ‖LU - A‖ / ‖A‖.
#
# Uses the Miyajima products (MMul4, oishi_MMul) for rigorous matrix multiplication
# with directed rounding.

"""
    _rigorous_MMul_real(F::AbstractMatrix{T}, G::AbstractMatrix{T}) where T<:AbstractFloat

Compute a rigorous BallMatrix enclosure of F*G for real matrices using directed rounding.
Returns BallMatrix{T,T} where the midpoint/radius pair encloses all possible products.

This is essentially MMul4 for point matrices, computing lower/upper bounds via
RoundDown/RoundUp and converting to center-radius form.
"""
function _rigorous_MMul_real(F::AbstractMatrix{T}, G::AbstractMatrix{T}) where T<:AbstractFloat
    # Compute upper bound with RoundUp
    C_up = setrounding(T, RoundUp) do
        F * G
    end

    # Compute lower bound with RoundDown
    C_lo = setrounding(T, RoundDown) do
        F * G
    end

    # Convert to center-radius form
    C_mid, C_rad = setrounding(T, RoundUp) do
        half = T(0.5)
        mid = (C_up .+ C_lo) .* half
        rad = (C_up .- C_lo) .* half
        return mid, rad
    end

    return BallMatrix(C_mid, C_rad)
end

"""
    _rigorous_MMul(F::AbstractMatrix, G::AbstractMatrix)

Compute a rigorous BallMatrix enclosure of F*G for either real or complex matrices.
Dispatches to appropriate method based on element type.
"""
function _rigorous_MMul(F::AbstractMatrix{T}, G::AbstractMatrix{T}) where T<:Real
    return _rigorous_MMul_real(F, G)
end

function _rigorous_MMul(F::AbstractMatrix{Complex{T}}, G::AbstractMatrix{Complex{T}}) where T<:AbstractFloat
    # Use the existing oishi_MMul for complex matrices
    return oishi_MMul(F, G)
end

# Mixed precision / promote to common type
function _rigorous_MMul(F::AbstractMatrix, G::AbstractMatrix)
    T = promote_type(eltype(F), eltype(G))
    return _rigorous_MMul(convert.(T, F), convert.(T, G))
end

"""
    _rigorous_residual_bound(F::AbstractMatrix, G::AbstractMatrix, A::AbstractMatrix)

Compute a rigorous upper bound on ‖FG - A‖∞ (max-norm of residual).
Uses directed rounding for the entire computation chain.

Returns a scalar upper bound (not a Ball) that rigorously contains the true residual norm.
"""
function _rigorous_residual_bound(F::AbstractMatrix{T}, G::AbstractMatrix{T},
                                   A::AbstractMatrix{T}) where T<:Real
    # Step 1: Compute rigorous enclosure of F*G
    FG_ball = _rigorous_MMul_real(F, G)
    FG_mid = mid(FG_ball)
    FG_rad = rad(FG_ball)

    # Step 2: Compute residual = FG - A
    # The residual is in [FG_mid - FG_rad - A, FG_mid + FG_rad - A]
    # = [(FG_mid - A) - FG_rad, (FG_mid - A) + FG_rad]

    # Step 3: Upper bound on |residual| = max(|lower|, |upper|)
    # For each entry: |residual_ij| ≤ |FG_mid_ij - A_ij| + FG_rad_ij
    residual_bound = setrounding(T, RoundUp) do
        diff = FG_mid .- A
        abs.(diff) .+ FG_rad
    end

    # Step 4: Return max norm (‖·‖∞)
    return setrounding(T, RoundUp) do
        maximum(residual_bound)
    end
end

function _rigorous_residual_bound(F::AbstractMatrix{Complex{T}}, G::AbstractMatrix{Complex{T}},
                                   A::AbstractMatrix{Complex{T}}) where T<:AbstractFloat
    # For complex matrices, use oishi_MMul
    FG_ball = oishi_MMul(F, G)
    FG_mid = mid(FG_ball)
    FG_rad = rad(FG_ball)

    # Compute residual bound
    residual_bound = setrounding(T, RoundUp) do
        diff = FG_mid .- A
        abs.(diff) .+ FG_rad
    end

    return setrounding(T, RoundUp) do
        maximum(residual_bound)
    end
end

"""
    _rigorous_relative_residual_norm(F, G, A)

Compute rigorous upper bound on ‖FG - A‖∞ / ‖A‖∞.
This is the relative residual norm used in decomposition verification.

Returns a scalar upper bound that rigorously contains the true relative residual.
"""
function _rigorous_relative_residual_norm(F::AbstractMatrix{T}, G::AbstractMatrix{T},
                                          A::AbstractMatrix{T}) where T<:Real
    residual_bound = _rigorous_residual_bound(F, G, A)

    # Compute ‖A‖∞ = max_i Σ_j |A_ij| with RoundDown for division safety
    A_norm = setrounding(T, RoundDown) do
        maximum(sum(abs.(A), dims=2))
    end

    # Handle near-zero A
    if A_norm <= eps(T) * maximum(abs.(A))
        return T(Inf)
    end

    return setrounding(T, RoundUp) do
        residual_bound / A_norm
    end
end

function _rigorous_relative_residual_norm(F::AbstractMatrix{Complex{T}}, G::AbstractMatrix{Complex{T}},
                                          A::AbstractMatrix{Complex{T}}) where T<:AbstractFloat
    residual_bound = _rigorous_residual_bound(F, G, A)

    A_norm = setrounding(T, RoundDown) do
        maximum(sum(abs.(A), dims=2))
    end

    if A_norm <= eps(T) * maximum(abs.(A))
        return T(Inf)
    end

    return setrounding(T, RoundUp) do
        residual_bound / A_norm
    end
end

"""
    _rigorous_gram_residual_bound(G::AbstractMatrix{T}, A::AbstractMatrix{T}) where T

Compute rigorous upper bound on ‖G'G - A‖∞ for Cholesky verification.
Uses directed rounding throughout.
"""
function _rigorous_gram_residual_bound(G::AbstractMatrix{T}, A::AbstractMatrix{T}) where T<:Real
    return _rigorous_residual_bound(G', G, A)
end

function _rigorous_gram_residual_bound(G::AbstractMatrix{Complex{T}}, A::AbstractMatrix{Complex{T}}) where T<:AbstractFloat
    return _rigorous_residual_bound(G', G, A)
end

"""
    _rigorous_gram_relative_residual_norm(G, A)

Compute rigorous upper bound on ‖G'G - A‖∞ / ‖A‖∞ for Cholesky.
"""
function _rigorous_gram_relative_residual_norm(G::AbstractMatrix{T}, A::AbstractMatrix{T}) where T<:Real
    return _rigorous_relative_residual_norm(G', G, A)
end

function _rigorous_gram_relative_residual_norm(G::AbstractMatrix{Complex{T}}, A::AbstractMatrix{Complex{T}}) where T<:AbstractFloat
    return _rigorous_relative_residual_norm(G', G, A)
end

"""
    _rigorous_max_abs_norm(A::AbstractMatrix)

Compute rigorous upper bound on maximum absolute value of matrix entries.
"""
function _rigorous_max_abs_norm(A::AbstractMatrix{T}) where T<:Real
    return setrounding(T, RoundUp) do
        maximum(abs.(A))
    end
end

function _rigorous_max_abs_norm(A::AbstractMatrix{Complex{T}}) where T<:AbstractFloat
    return setrounding(T, RoundUp) do
        maximum(abs.(A))
    end
end
