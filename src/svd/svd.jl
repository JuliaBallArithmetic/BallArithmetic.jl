"""
    RigorousSVDResult

Container returned by [`rigorous_svd`](@ref) bundling the midpoint
factorisation, the certified singular-value enclosures, and the
block-diagonal refinement obtained from [`miyajima_vbd`](@ref).  Besides
the singular values themselves the struct exposes the residual and
orthogonality defect bounds that underpin the certification.
"""
struct RigorousSVDResult{UT, ST, ΣT, VT, ET, RT, VBDT}
    """Ball enclosure of the left singular vectors."""
    U::UT
    """Certified singular values returned as floating-point balls."""
    singular_values::ST
    """Diagonal ball matrix containing the singular-value enclosure."""
    Σ::ΣT
    """Ball enclosure of the right singular vectors."""
    V::VT
    """Residual enclosure `U * Σ * V' - A`."""
    residual::ET
    """Upper bound on `‖residual‖₂`."""
    residual_norm::RT
    """Upper bound on `‖V'V - I‖₂` (right orthogonality defect)."""
    right_orthogonality_defect::RT
    """Upper bound on `‖U'U - I‖₂` (left orthogonality defect)."""
    left_orthogonality_defect::RT
    """Verified block diagonalisation of `Σ'Σ` via Miyajima's procedure, or `nothing` when skipped."""
    block_diagonalisation::VBDT
end

"""
    rigorous_svd(A::BallMatrix; apply_vbd = true)

Compute a rigorous singular value decomposition of the ball matrix `A`.
The midpoint SVD is certified following Theorem 3.1 of
Ref. [Rump2011](@cite); optionally, the resulting singular-value
enclosure can be refined by applying [`miyajima_vbd`](@ref) to `Σ'Σ`,
yielding a block-diagonal structure with a rigorously bounded remainder.

The returned [`RigorousSVDResult`](@ref) exposes both the enclosures and
the intermediate norm bounds that justify them. When `apply_vbd` is set
to `false`, the `block_diagonalisation` field is `nothing`.

# References

* [Rump2011](@cite) Rump S., BIT 51, 2 (2011)
"""
function rigorous_svd(A::BallMatrix{T}; apply_vbd::Bool = true) where {T}
    svdA = svd(A.c)
    return _certify_svd(A, svdA; apply_vbd)
end

"""
    svdbox(A::BallMatrix; apply_vbd = true)

Backward-compatible wrapper returning only the vector of singular-value
enclosures produced by [`rigorous_svd`](@ref).  New code should prefer
[`rigorous_svd`](@ref) directly to access the additional certification
data.  The optional `apply_vbd` flag mirrors the one in
[`rigorous_svd`](@ref).
"""
function svdbox(A::BallMatrix{T}; apply_vbd::Bool = true) where {T}
    result = rigorous_svd(A; apply_vbd)
    return result.singular_values
end

function _certify_svd(A::BallMatrix{T}, svdA::SVD; apply_vbd::Bool = true) where {T}
    U = BallMatrix(svdA.U)
    V = BallMatrix(svdA.V)
    Vt = BallMatrix(svdA.Vt)
    Σ_mid = BallMatrix(Diagonal(svdA.S))

    E = U * Σ_mid * Vt - A
    normE = collatz_upper_bound_L2_opnorm(E)
    @debug "norm E" normE

    F = Vt * V - I
    normF = collatz_upper_bound_L2_opnorm(F)
    @debug "norm F" normF

    G = U' * U - I
    normG = collatz_upper_bound_L2_opnorm(G)
    @debug "norm G" normG

    @assert normF < 1 "It is not possible to verify the singular values with this precision"
    @assert normG < 1 "It is not possible to verify the singular values with this precision"

    den_down = @up (1.0 + normF) * (1.0 + normG)
    den_up = @down (1.0 - normF) * (1.0 - normG)

    svdbounds_down = setrounding(T, RoundDown) do
        [(σ - normE) / den_down for σ in svdA.S]
    end

    svdbounds_up = setrounding(T, RoundUp) do
        [(σ + normE) / den_up for σ in svdA.S]
    end

    midpoints = (svdbounds_down + svdbounds_up) / 2
    radii = setrounding(T, RoundUp) do
        [max(svdbounds_up[i] - midpoints[i], midpoints[i] - svdbounds_down[i])
         for i in 1:length(midpoints)]
    end

    singular_values = [Ball(midpoints[i], radii[i]) for i in 1:length(midpoints)]
    Σ = _diagonal_ball_matrix(singular_values)

    ΔΣ = Σ - Σ_mid
    # Reuse the midpoint residual `E` and only account for the interval
    # widening introduced when replacing `Σ_mid` with the ball diagonal `Σ`.
    residual = E + U * ΔΣ * Vt
    residual_norm = collatz_upper_bound_L2_opnorm(residual)

    vbd = nothing
    if apply_vbd
        H = adjoint(Σ) * Σ
        vbd = miyajima_vbd(H; hermitian = true)
    end

    return RigorousSVDResult(U, singular_values, Σ, V, residual,
        residual_norm, normF, normG, vbd)
end

function _diagonal_ball_matrix(values::Vector{Ball{T, NT}}) where {T, NT}
    mids = map(mid, values)
    rads = map(rad, values)
    return BallMatrix(Diagonal(mids), Diagonal(rads))
end
