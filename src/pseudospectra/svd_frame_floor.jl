# A lower bound of σ_min(A − zI) at O(n²) operations for each z, from one singular value
# decomposition of A.

export SVDFrameFloor, svd_frame_floor

"""
    SVDFrameFloor{T}

The quantities of [`svd_frame_floor`](@ref) that do not depend on `z`: ball matrices `H ∋ V*A*AV`,
`K ∋ V*AV` and `G ∋ V*V` for a floating-point matrix `V`, and `gram_norm ≥ ‖V‖₂²` (`Inf` when `V`
is not proved invertible). Evaluate with [`sigma_min_floor`](@ref) or [`resolvent_bound`](@ref).
"""
struct SVDFrameFloor{T <: AbstractFloat}
    H::BallMatrix
    K::BallMatrix
    G::BallMatrix
    gram_norm::T
end

"""
    svd_frame_floor(A::BallMatrix) -> SVDFrameFloor

Prepare, from one singular value decomposition of the midpoint of `A`, a lower bound of
`σ_min(A − zI)` that costs `O(n²)` operations at each `z`. For a ball matrix `A` the bound holds
for every matrix of the ball.

# The bound

Let `V` be any invertible matrix and `N(z) := V*(A − zI)*(A − zI)V`, Hermitian. Expanding,

    N(z) = H − z̄ K − z K* + |z|² G,      H = V*A*AV,   K = V*AV,   G = V*V,

so `N(z)` is known entrywise from three matrices that do not depend on `z`. For `y = Vx`,
`‖(A − zI)y‖² = x*N(z)x ≥ λ_min(N(z))‖x‖²` and `‖y‖² ≤ ‖V‖₂²‖x‖²`, hence, when `λ_min(N(z)) ≥ 0`,

    σ_min(A − zI)² ≥ λ_min(N(z)) / ‖V‖₂².

`λ_min(N(z))` is bounded below by Gershgorin's theorem for a Hermitian matrix,

    λ_min(N(z)) ≥ γ(z) := min_i ( N_ii(z) − Σ_{j≠i} |N_ij(z)| ).

`V` is the matrix of right singular vectors of the midpoint of `A`, computed in floating point:
then `H` is close to the diagonal matrix of the squared singular values, `G` to the identity, and
the off-diagonal part of `N(z)` is `−z̄K − zK*` up to rounding, which vanishes for a normal
matrix with distinct singular values and grows with the departure from normality. Nothing is
assumed of `V` beyond what is checked: `H`, `K`, `G` are computed in ball arithmetic from `A` and
`V`, `‖V‖₂² = ‖G‖₂ ≤ 1 + ‖G − I‖₂`, and `V` is invertible when `‖G − I‖₂ < 1`. The left singular
vectors are not used.

The statement is the theorem "Certified σ_min surface from one verified SVD" of I. Nisoli,
*Enclosure of eigenvalues via approximate diagonalization* (unpublished note, 2026), with the
diagonal part of the term linear in `z` kept in the Gershgorin centre and with the deviation
of the computed decomposition from an exact one carried by the radii of `H`, `K`, `G` in place of
a separate term; the argument is the one above.
"""
function svd_frame_floor(A::BallMatrix{T}) where {T}
    size(A, 1) == size(A, 2) || throw(ArgumentError("svd_frame_floor: A must be square"))
    V = BallMatrix(Matrix(svd(Matrix(mid(A))).V))
    AV = A * V
    G = V' * V
    ρ = upper_bound_L2_opnorm(G - I)
    return SVDFrameFloor{T}(AV' * AV, V' * AV, G, ρ < 1 ? add_up(one(T), T(ρ)) : T(Inf))
end

"""
    sigma_min_floor(f::SVDFrameFloor, z)

A lower bound of `σ_min(A − zI)`: `√(γ(z)/‖V‖₂²)` with the quantities of
[`svd_frame_floor`](@ref), every operation rounded toward a smaller result; zero when `γ(z)` is
not positive or `V` was not proved invertible.
"""
function sigma_min_floor(f::SVDFrameFloor{T}, z::Number) where {T}
    isfinite(f.gram_norm) || return zero(T)
    n = size(f.H, 1)
    zb = Ball(Complex{T}(z))
    zc = conj(zb)
    az2 = zb * zc
    γ = T(Inf)
    for i in 1:n
        off = zero(T)
        d = zero(T)
        for j in 1:n
            b = f.H[i, j] - zc * f.K[i, j] - zb * conj(f.K[j, i]) + az2 * f.G[i, j]
            if i == j
                d = sub_down(real(mid(b)), rad(b))
            else
                off = add_up(off, add_up(abs_up(mid(b)), rad(b)))
            end
        end
        γ = min(γ, sub_down(d, off))
    end
    γ > 0 || return zero(T)
    return sqrt_down(div_down(γ, f.gram_norm))
end

"""
    resolvent_bound(f::SVDFrameFloor, z)

An upper bound of `‖(A − zI)⁻¹‖₂`: the reciprocal, rounded up, of [`sigma_min_floor`](@ref); `Inf`
where that is zero.
"""
function resolvent_bound(f::SVDFrameFloor{T}, z::Number) where {T}
    s = sigma_min_floor(f, z)
    return s > 0 ? div_up(one(T), s) : T(Inf)
end
