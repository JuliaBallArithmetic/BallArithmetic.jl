# Implementing Theorem 2 Miyajima
# Numerical enclosure for each eigenvalue in generalized eigenvalue problem

"""
    RigorousGeneralizedEigenvaluesResult

Container returned by [`rigorous_generalized_eigenvalues`](@ref) bundling
the midpoint eigenvector factorisation, the certified eigenvalue
enclosures, and the norm bounds underpinning their verification.
Besides behaving like the underlying vector of eigenvalue balls, the
struct exposes the interval residual, the projected residual used in the
Miyajima certification, and the coupling defect of the left action
(`left_action * B * right_vectors - I`).
"""
struct RigorousGeneralizedEigenvaluesResult{XT, YT, ΛT, ΛMT, ET, RT, BT}
    """Ball enclosure of the right eigenvectors."""
    right_vectors::XT
    """Ball enclosure of the left action verifying `left_action * B * right_vectors ≈ I`."""
    left_action::YT
    """Certified eigenvalues returned as floating-point balls."""
    eigenvalues::ΛT
    """Diagonal ball matrix containing the eigenvalue enclosure."""
    Λ::ΛMT
    """Residual enclosure `A * right_vectors - B * right_vectors * Λ`."""
    residual::ET
    """Upper bound on `‖residual‖∞`."""
    residual_norm::RT
    """Upper bound on `‖left_action * residual‖∞` (projected residual)."""
    projected_residual_norm::RT
    """Coupling defect enclosure `left_action * B * right_vectors - I`."""
    coupling_defect::BT
    """Upper bound on `‖coupling_defect‖∞`."""
    coupling_defect_norm::RT
end

Base.size(result::RigorousGeneralizedEigenvaluesResult) = size(result.eigenvalues)
Base.length(result::RigorousGeneralizedEigenvaluesResult) = length(result.eigenvalues)
Base.firstindex(result::RigorousGeneralizedEigenvaluesResult) = firstindex(result.eigenvalues)
Base.lastindex(result::RigorousGeneralizedEigenvaluesResult) = lastindex(result.eigenvalues)
Base.lastindex(result::RigorousGeneralizedEigenvaluesResult, i::Int) =
    lastindex(result.eigenvalues, i)
Base.getindex(result::RigorousGeneralizedEigenvaluesResult, inds...) =
    getindex(result.eigenvalues, inds...)
Base.iterate(result::RigorousGeneralizedEigenvaluesResult) = iterate(result.eigenvalues)
Base.iterate(result::RigorousGeneralizedEigenvaluesResult, state) =
    iterate(result.eigenvalues, state)

"""
    RigorousEigenvaluesResult

Container returned by [`rigorous_eigenvalues`](@ref) bundling the
midpoint eigenvector factorisation, the certified eigenvalue enclosures,
and the norm bounds underpinning their verification. Besides behaving
like the underlying vector of eigenvalue balls, the struct exposes the
interval residual, the projected residual used in the Miyajima
certification, and the inverse defect (`inverse * vectors - I`).
"""
struct RigorousEigenvaluesResult{VT, YT, ΛT, ΛMT, ET, RT, BT}
    """Ball enclosure of the eigenvectors."""
    vectors::VT
    """Ball enclosure of the inverse of `vectors`."""
    inverse::YT
    """Certified eigenvalues returned as floating-point balls."""
    eigenvalues::ΛT
    """Diagonal ball matrix containing the eigenvalue enclosure."""
    Λ::ΛMT
    """Residual enclosure `A * vectors - vectors * Λ`."""
    residual::ET
    """Upper bound on `‖residual‖∞`."""
    residual_norm::RT
    """Upper bound on `‖inverse * residual‖∞` (projected residual)."""
    projected_residual_norm::RT
    """Inverse defect enclosure `inverse * vectors - I`."""
    inverse_defect::BT
    """Upper bound on `‖inverse_defect‖∞`."""
    inverse_defect_norm::RT
end

Base.size(result::RigorousEigenvaluesResult) = size(result.eigenvalues)
Base.length(result::RigorousEigenvaluesResult) = length(result.eigenvalues)
Base.firstindex(result::RigorousEigenvaluesResult) = firstindex(result.eigenvalues)
Base.lastindex(result::RigorousEigenvaluesResult) = lastindex(result.eigenvalues)
Base.lastindex(result::RigorousEigenvaluesResult, i::Int) =
    lastindex(result.eigenvalues, i)
Base.getindex(result::RigorousEigenvaluesResult, inds...) =
    getindex(result.eigenvalues, inds...)
Base.iterate(result::RigorousEigenvaluesResult) = iterate(result.eigenvalues)
Base.iterate(result::RigorousEigenvaluesResult, state) =
    iterate(result.eigenvalues, state)

"""
    rigorous_generalized_eigenvalues(A::BallMatrix, B::BallMatrix)

Compute rigorous enclosures for the eigenvalues of the generalised
problem `A * x = λ * B * x`, following Ref. [Miyajima2012](@cite).  The
returned [`RigorousGeneralizedEigenvaluesResult`](@ref) exposes both the
interval enclosures and the norm bounds used during certification.

# References

* [Miyajima2012](@cite) Miyajima, JCAM 246, 9 (2012)
"""
function rigorous_generalized_eigenvalues(A::BallMatrix{T}, B::BallMatrix{T}) where {T}
    gev = eigen(A.c, B.c)
    return _certify_generalized_eigenvalues(A, B, gev)
end

"""
    gevbox(A::BallMatrix{T}, B::BallMatrix{T})

Backward-compatible wrapper returning only the vector of eigenvalue
enclosures produced by [`rigorous_generalized_eigenvalues`](@ref).
"""
function gevbox(A::BallMatrix{T}, B::BallMatrix{T}) where {T}
    result = rigorous_generalized_eigenvalues(A, B)
    return result.eigenvalues
end

function _certify_generalized_eigenvalues(A::BallMatrix{T}, B::BallMatrix{T},
        gev::GeneralizedEigen) where {T}
    X_mid = gev.vectors
    Y_mid = inv(B.c * X_mid)

    bX = BallMatrix(X_mid)
    bY = BallMatrix(Y_mid)

    coupling_defect = bY * B * bX - I
    coupling_defect_norm = upper_bound_L_inf_opnorm(coupling_defect)
    @debug "norm coupling defect" coupling_defect_norm
    @assert coupling_defect_norm < 1 "It is not possible to verify the eigenvalues with this precision"

    D_mid = BallMatrix(Diagonal(gev.values))
    projected_residual_mid = bY * (A * bX - B * bX * D_mid)
    projected_residual_norm_mid = upper_bound_L_inf_opnorm(projected_residual_mid)
    @debug "projected residual norm" projected_residual_norm_mid

    den_up = @down (one(T) - coupling_defect_norm)
    eps = @up projected_residual_norm_mid / den_up

    eigenvalues = [Ball(lam, eps) for lam in gev.values]
    Λ = _diagonal_ball_matrix(eigenvalues)

    residual = A * bX - B * bX * Λ
    residual_norm = upper_bound_L_inf_opnorm(residual)
    projected_residual = bY * residual
    projected_residual_norm = upper_bound_L_inf_opnorm(projected_residual)

    return RigorousGeneralizedEigenvaluesResult(bX, bY, eigenvalues, Λ,
        residual, residual_norm, projected_residual_norm,
        coupling_defect, coupling_defect_norm)
end

"""
    rigorous_eigenvalues(A::BallMatrix)

Compute rigorous enclosures for the eigenvalues of `A`, following
Ref. [Miyajima2012](@cite).  The returned
[`RigorousEigenvaluesResult`](@ref) exposes both the interval enclosures
and the norm bounds used during certification.

TODO: Using Miyajima's algorithm is overkill, may be worth using

# References

* [Miyajima2012](@cite) Miyajima, JCAM 246, 9 (2012)
"""
function rigorous_eigenvalues(A::BallMatrix{T}) where {T}
    gev = eigen(A.c)
    return _certify_eigenvalues(A, gev)
end

"""
    evbox(A::BallMatrix{T})

Backward-compatible wrapper returning only the vector of eigenvalue
enclosures produced by [`rigorous_eigenvalues`](@ref).
"""
function evbox(A::BallMatrix{T}) where {T}
    result = rigorous_eigenvalues(A)
    return result.eigenvalues
end

function _certify_eigenvalues(A::BallMatrix{T}, gev::Eigen) where {T}
    X_mid = gev.vectors
    Y_mid = inv(X_mid)

    bX = BallMatrix(X_mid)
    bY = BallMatrix(Y_mid)

    inverse_defect = bY * bX - I
    inverse_defect_norm = upper_bound_L_inf_opnorm(inverse_defect)
    @debug "norm inverse defect" inverse_defect_norm
    @assert inverse_defect_norm < 1 "It is not possible to verify the eigenvalues with this precision",
        inverse_defect_norm,
        norm(X_mid, 2),
        norm(Y_mid, 2)

    D_mid = BallMatrix(Diagonal(gev.values))

    # probably something better can be done here
    # since this is not GEV, but only EV
    # need to look better at Miyajima
    # https://www.sciencedirect.com/science/article/pii/S037704270900795X

    projected_residual_mid = bY * (A * bX - bX * D_mid)

    projected_residual_norm_mid = upper_bound_L_inf_opnorm(projected_residual_mid)
    @debug "projected residual norm" projected_residual_norm_mid

    den_up = @down (one(T) - inverse_defect_norm)
    eps = @up projected_residual_norm_mid / den_up

    eigenvalues = [Ball(lam, eps) for lam in gev.values]
    Λ = _diagonal_ball_matrix(eigenvalues)

    residual = A * bX - bX * Λ
    residual_norm = upper_bound_L_inf_opnorm(residual)
    projected_residual = bY * residual
    projected_residual_norm = upper_bound_L_inf_opnorm(projected_residual)

    return RigorousEigenvaluesResult(bX, bY, eigenvalues, Λ, residual,
        residual_norm, projected_residual_norm, inverse_defect, inverse_defect_norm)
end
