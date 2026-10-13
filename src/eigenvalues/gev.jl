# Implementing Theorem 2 Miyajima
# Numerical enclosure for each eigenvalue in generalized eigenvalue problem

"""
    RigorousGeneralizedEigenvaluesResult

Container returned by [`rigorous_generalized_eigenvalues`](@ref) bundling
the midpoint eigenvector factorisation, the balls whose union contains the
eigenvalues, and the norm bounds underpinning their verification.
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
    """Balls whose union contains the eigenvalues; a single ball need not contain one."""
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
midpoint eigenvector factorisation, the balls whose union contains the eigenvalues,
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
    """Balls whose union contains the eigenvalues; a single ball need not contain one."""
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

Compute a set of balls whose union contains the eigenvalues of the generalised
problem `A * x = λ * B * x`, by Theorem 1 of Miyajima (2010), below.  The
returned [`RigorousGeneralizedEigenvaluesResult`](@ref) exposes both the
balls and the norm bounds used during certification.

**What is certified.** The returned balls have the computed eigenvalues as centres and one common
radius, and every eigenvalue of every pair of matrices in the input balls lies in their union.
A single ball need not contain an eigenvalue, and the number of eigenvalues in a ball or in a
group of overlapping balls is not certified. For sets with a certified number of eigenvalues
use [`verifyeigall`](@ref).

The balls are discs of the complex plane, with complex centres, also when `A` and `B` are real
and the computed eigenvalues are real: pairs of matrices in the input balls may have complex
eigenvalues, which lie in the discs.

# References

S. Miyajima, *Fast enclosure for all eigenvalues in generalized eigenvalue problems*,
J. Comput. Appl. Math. **233**(11) (2010) 2994-3004, doi 10.1016/j.cam.2009.11.048,
Theorem 1 ([Miyajima2010](@cite)): for an arbitrary `Y`, with `R₁ = Y(AX̃ − BX̃D̃)` and
`R₂ = YBX̃ − I`, if `‖R₂‖_∞ < 1` then every eigenvalue is within `ε = ‖R₁‖_∞/(1 − ‖R₂‖_∞)` of
one of the diagonal entries of `D̃`.
"""
function rigorous_generalized_eigenvalues(A::BallMatrix{T}, B::BallMatrix{T}) where {T}
    gev = try
        eigen(A.c, B.c)
    catch e
        # no generalised eigensolver for this arithmetic (BigFloat): the approximation of
        # `verifyeigall(A, B)`, which falls back on the eigendecomposition of B⁻¹A. Nothing is
        # assumed about the approximation; the certification below is the same.
        e isa MethodError || rethrow(e)
        V, λ = _miyajima2014a_alg1_step1(Matrix{Complex{T}}(A.c), Matrix{Complex{T}}(B.c))
        V === nothing && throw(ArgumentError("rigorous_generalized_eigenvalues: no approximate " *
                                             "eigendecomposition, the midpoint of B is singular"))
        (vectors = V, values = λ)
    end
    # the eigenvalues of a pencil are complex numbers, also when the computed ones are real: the
    # balls are discs of the complex plane
    CT = Complex{T}
    return _certify_generalized_eigenvalues(A, B,
        (vectors = Matrix{CT}(gev.vectors), values = Vector{CT}(gev.values)))
end

"""
    gevbox(A::BallMatrix{T}, B::BallMatrix{T})

Backward-compatible wrapper returning only the vector of balls
produced by [`rigorous_generalized_eigenvalues`](@ref): their union contains the eigenvalues, a single ball need
not contain one.
"""
function gevbox(A::BallMatrix{T}, B::BallMatrix{T}) where {T}
    result = rigorous_generalized_eigenvalues(A, B)
    return result.eigenvalues
end

# `gev` has the approximate eigenvectors in `vectors` and the approximate eigenvalues in `values`
function _certify_generalized_eigenvalues(A::BallMatrix{T}, B::BallMatrix{T}, gev) where {T}
    X_mid = gev.vectors
    Y_mid = inv(B.c * X_mid)

    bX = BallMatrix(X_mid)
    bY = BallMatrix(Y_mid)

    coupling_defect = bY * B * bX - I
    coupling_defect_norm = upper_bound_L_inf_opnorm(coupling_defect)
    @debug "norm coupling defect" coupling_defect_norm
    # the hypothesis of the theorem; an explicit error, since assertions may be compiled out
    coupling_defect_norm < 1 ||
        error("It is not possible to verify the eigenvalues with this precision: the bound " *
              "of ‖Y B X − I‖_∞ is $coupling_defect_norm, not below 1")

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
    rigorous_eigenvalues(H::Hermitian)

Compute a set of balls whose union contains the eigenvalues of `A`, following
Theorem 1 of Miyajima (2010), below, with `B = I`.  The returned [`RigorousEigenvaluesResult`](@ref) exposes both the
balls and the norm bounds used during certification.

**What is certified.** The returned balls have the computed eigenvalues as centres and one common
radius, and every eigenvalue of every matrix in the input ball lies in their union.
A single ball need not contain an eigenvalue, and the number of eigenvalues in a ball or in a
group of overlapping balls is not certified. For sets with a certified number of eigenvalues
use [`verifyeigall`](@ref).

The type of the argument decides the kind of ball:

- for a `BallMatrix` the balls are discs of the complex plane, with complex centres, also when
  the matrix is real and the computed eigenvalues are real: matrices in the input ball may have
  complex eigenvalues, which lie in the discs;
- for a [`HermitianBallMatrix`](@ref), the Hermitian matrices within the tolerances, and for a
  point matrix that is Hermitian by its type (`Hermitian(M)`, or `Symmetric(M)` with `M` real),
  the eigenvalues are real and the balls are real intervals: every eigenvalue of every
  Hermitian matrix within the tolerances lies in their union.

# References

S. Miyajima, *Fast enclosure for all eigenvalues in generalized eigenvalue problems*,
J. Comput. Appl. Math. **233**(11) (2010) 2994-3004, doi 10.1016/j.cam.2009.11.048,
Theorem 1 ([Miyajima2010](@cite)): for an arbitrary `Y`, with `R₁ = Y(AX̃ − BX̃D̃)` and
`R₂ = YBX̃ − I`, if `‖R₂‖_∞ < 1` then every eigenvalue is within `ε = ‖R₁‖_∞/(1 − ‖R₂‖_∞)` of
one of the diagonal entries of `D̃`.
"""
function rigorous_eigenvalues(A::BallMatrix{T}) where {T}
    F = eigen(A.c)
    # the eigenvalues of a general matrix are complex numbers, also when the computed ones are
    # real: the balls are discs of the complex plane
    CT = Complex{T}
    return _certify_eigenvalues(A, Eigen(Vector{CT}(F.values), Matrix{CT}(F.vectors)))
end

# A Hermitian ball matrix: the statement is for the Hermitian matrices within the tolerances, whose
# eigenvalues are real, so the discs of the theorem are cut down to real intervals. The
# certification runs on the dense ball matrix that contains them all, with the eigendecomposition
# of the Hermitian midpoint, whose eigenvalues are real numbers.
function rigorous_eigenvalues(A::HermitianBallMatrix)
    D = _dense_hull(A)
    return _certify_eigenvalues(D, eigen(Hermitian(mid(D))))
end
# a point matrix that is Hermitian by its type: the same with radius zero
rigorous_eigenvalues(H::_HermitianPointMatrix) = rigorous_eigenvalues(HermitianBallMatrix(H))

"""
    evbox(A::BallMatrix)
    evbox(H::Hermitian)

Backward-compatible wrapper returning only the vector of balls
produced by [`rigorous_eigenvalues`](@ref): their union contains the eigenvalues, a single ball need
not contain one.
"""
evbox(A::Union{BallMatrix, _HermitianPointMatrix}) = rigorous_eigenvalues(A).eigenvalues

function _certify_eigenvalues(A::BallMatrix{T}, gev::Eigen) where {T}
    X_mid = gev.vectors
    Y_mid = inv(X_mid)

    bX = BallMatrix(X_mid)
    bY = BallMatrix(Y_mid)

    inverse_defect = bY * bX - I
    inverse_defect_norm = upper_bound_L_inf_opnorm(inverse_defect)
    @debug "norm inverse defect" inverse_defect_norm
    # the hypothesis of the theorem; an explicit error, since assertions may be compiled out
    inverse_defect_norm < 1 ||
        error("It is not possible to verify the eigenvalues with this precision: the bound " *
              "of ‖Y X − I‖_∞ is $inverse_defect_norm, not below 1")

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
