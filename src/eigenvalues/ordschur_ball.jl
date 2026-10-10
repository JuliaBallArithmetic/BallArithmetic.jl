# Reordering of an approximate Schur pair, with its defects measured afterwards, and the bound
# on the spectral projector that uses them.

"""
    ordschur_bigfloat(T::AbstractMatrix, Q::AbstractMatrix,
                      select::AbstractVector{Bool})

Reorder a Schur decomposition `A = Q T Q^H` so that the eigenvalues
corresponding to `select[i] == true` appear in the top-left block.

Delegates to `GenericSchur.ordschur` after wrapping the inputs in a
`Schur` factorization object.  Works for `BigFloat` and `Complex{BigFloat}`.

# Arguments
- `T`: Upper triangular Schur form (n × n)
- `Q`: Unitary Schur basis (n × n)
- `select`: Boolean mask of length n — `true` for eigenvalues to move to top-left

# Returns
`(Q_ord, T_ord, values)` — reordered Schur basis, Schur form, and eigenvalues.
"""
function ordschur_bigfloat(T::AbstractMatrix, Q::AbstractMatrix,
                           select::AbstractVector{Bool})
    vals = diag(T)
    F = Schur(Matrix(T), Matrix(Q), vals)
    F_ord = ordschur(F, BitVector(select))
    return F_ord.Z, F_ord.T, F_ord.values
end

"""
    ordschur_ball(Q_ball::BallMatrix, T_ball::BallMatrix, select; A = nothing)

Reorder an approximate Schur pair so that the diagonal entries with `select[i] == true` come
first, and bound the defects of the reordered pair.

The reordering itself is the floating-point one: `ordschur` on the midpoint of `T_ball` returns
the reordered matrix and the accumulated rotation `G`. No error is propagated through the
rotations. What the reordered pair satisfies is measured afterwards, in ball arithmetic.

# What is returned

A `NamedTuple` with

- `G`: the accumulated rotation, a matrix with exact entries (it is only approximately unitary);
- `T`: `T̃`, the upper triangle of the reordered midpoint, as a `BallMatrix` of radius zero. It is
  exactly upper triangular, and whatever the floating-point reordering left below the diagonal is
  in the residuals below;
- `Q`: the ball matrix `Q_ball·G`;
- `values`: the diagonal of `T̃`;
- `rotation_orth_defect`: upper bound of `‖G*G − I‖₂`;
- `rotation_residual`: upper bound of `‖TG − GT̃‖₂` for every `T` in `T_ball`;
- `orth_defect`, `fact_defect`: when the matrix `A` is given (a `BallMatrix`), upper bounds of
  `‖Q*Q − I‖₂` and of `‖AQ − QT̃‖₂` for every `Q` in the returned ball and every `A` in the
  ball `A`; `nothing` otherwise.

These are the two numbers [`spectral_projector_error_bound`](@ref) takes. Without `A`, they
follow from the defects `δ₀ ≥ ‖Q₀*Q₀ − I‖₂` and `ε₀ ≥ ‖AQ₀ − Q₀T‖₂` of the pair that was given,
since `AQ₀G − Q₀GT̃ = (AQ₀ − Q₀T)G + Q₀(TG − GT̃)`:

    ‖AQ − QT̃‖₂ ≤ ε₀ √(1 + δ_G) + √(1 + δ₀) ρ,

with `δ_G` and `ρ` the two rotation bounds.

`T_ball` is expected to be upper triangular (a complex Schur form). The cost is that of
`ordschur` plus three products of n×n ball matrices, and three more when `A` is given.
"""
function ordschur_ball(Q_ball::BallMatrix, T_ball::BallMatrix,
        select::AbstractVector{Bool}; A::Union{Nothing, BallMatrix} = nothing)
    n = size(T_ball, 1)
    n == size(T_ball, 2) || throw(DimensionMismatch("T_ball must be square"))
    n == size(Q_ball, 1) == size(Q_ball, 2) || throw(DimensionMismatch("Q_ball must be n×n"))
    length(select) == n || throw(DimensionMismatch("select must have length n"))
    A === nothing || size(A) == (n, n) || throw(DimensionMismatch("A must be n×n"))

    ET = eltype(mid(T_ball))
    # the rotation: the reordering applied to the identity
    G, T_ord, _ = ordschur_bigfloat(mid(T_ball), Matrix{ET}(I, n, n), select)
    T̃ = Matrix(UpperTriangular(Matrix(T_ord)))
    bG = BallMatrix(Matrix(G))
    bT̃ = BallMatrix(T̃)

    rotation_orth_defect = upper_bound_L2_opnorm(bG' * bG - I)
    rotation_residual = upper_bound_L2_opnorm(T_ball * bG - bG * bT̃)
    Q = Q_ball * bG

    orth_defect = fact_defect = nothing
    if A !== nothing
        orth_defect = upper_bound_L2_opnorm(Q' * Q - I)
        fact_defect = upper_bound_L2_opnorm(A * Q - Q * bT̃)
    end

    return (; Q, T = bT̃, values = diag(T̃), G = Matrix(G), rotation_orth_defect,
        rotation_residual, orth_defect, fact_defect)
end

"""
    spectral_projector_error_bound(; resolvent_bound_A, contour_radius, orth_defect, fact_defect)

An upper bound of `‖P_A − P_c‖₂`, where `P_A` is the spectral projector of `A` for the part of
its spectrum inside a contour `Γ`, and `P_c = Q P_T̃ Q*` with `P_T̃` the spectral projector of a
matrix `T̃` for the part of its spectrum inside the same contour. `Q` and `T̃` are any pair with
`δ = orth_defect ≥ ‖I − Q*Q‖₂ < 1` and `ε = fact_defect ≥ ‖AQ − QT̃‖₂`, for instance the one
returned by [`ordschur_ball`](@ref) with `A` given.

# The bound

Both projectors are contour integrals, `P_A = (1/2πi)∮_Γ (zI − A)⁻¹ dz` and
`P_c = (1/2πi)∮_Γ Q(zI − T̃)⁻¹Q* dz`. With `R = AQ − QT̃`, so that `(zI − A)Q = Q(zI − T̃) − R`,

    (zI − A)⁻¹ − Q(zI − T̃)⁻¹Q* = (zI − A)⁻¹ [ (I − QQ*) + R (zI − T̃)⁻¹ Q* ].

With `M_A ≥ ‖(zI − A)⁻¹‖₂` on `Γ` (`resolvent_bound_A`), `M_T̃ ≥ ‖(zI − T̃)⁻¹‖₂` on `Γ`,
`‖I − QQ*‖₂ = ‖I − Q*Q‖₂ ≤ δ` for a square `Q`, `‖Q‖₂ ≤ √(1 + δ)`, and a contour of length at most
`2πr` (`contour_radius`; a circle of radius `r` or a polygon inscribed in it),

    ‖P_A − P_c‖₂ ≤ r · M_A · ( δ + ε · M_T̃ · √(1 + δ) ).

`M_T̃` is obtained from `M_A`: with `E = I − Q*Q` and `F = Q*(zI − A)Q`,
`(I − E)(zI − T̃) = F + Q*R`, `‖F⁻¹‖₂ ≤ M_A/(1 − δ)`, so with
`γ = M_A √(1 + δ) ε/(1 − δ) < 1`,

    M_T̃ ≤ M_A (1 + δ) / ((1 − δ)(1 − γ)).

The hypothesis that `T̃` has no spectrum on `Γ` follows from `γ < 1`. Every operation is rounded
up. Returns `Inf` when `δ ≥ 1` or `γ ≥ 1`.
"""
function spectral_projector_error_bound(; resolvent_bound_A::Real, contour_radius::Real,
        orth_defect::Real, fact_defect::Real)
    M_A, r, δ, ε = promote(float(resolvent_bound_A), float(contour_radius), float(orth_defect),
        float(fact_defect))
    RT = typeof(M_A)
    (isfinite(M_A) && isfinite(r) && isfinite(δ) && isfinite(ε)) || return RT(Inf)
    (M_A >= 0 && r >= 0 && δ >= 0 && ε >= 0) ||
        throw(ArgumentError("spectral_projector_error_bound: the four bounds must be nonnegative"))
    δ < 1 || return RT(Inf)
    one_ = one(RT)
    σ_Q = sqrt_up(add_up(one_, δ))                              # ‖Q‖₂
    F_inv = div_up(M_A, sub_down(one_, δ))                      # ‖F⁻¹‖₂
    γ = mul_up(mul_up(F_inv, σ_Q), ε)
    γ < 1 || return RT(Inf)
    M_T = div_up(mul_up(F_inv, add_up(one_, δ)), sub_down(one_, γ))
    return mul_up(mul_up(r, M_A), add_up(δ, mul_up(mul_up(ε, M_T), σ_Q)))
end

export ordschur_bigfloat, ordschur_ball, spectral_projector_error_bound
