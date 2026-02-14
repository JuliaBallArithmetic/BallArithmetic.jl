# Verified ordered Schur decomposition for BigFloat / BallMatrix
# Wraps GenericSchur's ordschur and adds rigorous error propagation

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
    ordschur_ball(Q_ball::BallMatrix, T_ball::BallMatrix,
                  select::AbstractVector{Bool})

Reorder a rigorous Schur decomposition enclosed in `BallMatrix` form.

Given `(Q_ball, T_ball)` enclosing the true Schur factors (e.g. from
[`rigorous_schur_bigfloat`](@ref)), reorder so that eigenvalues with
`select[i] == true` move to the top-left block, and return rigorous
`BallMatrix` enclosures of the reordered factors.

# Algorithm
1. Reorder midpoints via [`ordschur_bigfloat`](@ref).
2. Compute the accumulated unitary transformation `G_c = mid(Q_ball)^H Q_{ord,c}`.
3. Compute rigorous verification residuals:
   - **Orthogonality**: `‖I - Q_{ord,c}^H Q_{ord,c}‖₂`
   - **Factorization**: `‖A Q_{ord,c} - Q_{ord,c} T_{ord,c}‖₂` where
     `A` is enclosed by `Q_ball T_ball Q_ball^H`
4. Build `G_ball` with the orthogonality defect as componentwise radius,
   then propagate in ball arithmetic:
   - `T_{ord} = G_ball^H T_ball G_ball`
   - `Q_{ord} = Q_ball G_ball`

# Returns
A `NamedTuple` with fields:
- `Q::BallMatrix` — rigorous enclosure of the reordered Schur basis
- `T::BallMatrix` — rigorous enclosure of the reordered Schur form
- `values::Vector` — reordered eigenvalues (midpoint only)
- `orth_defect` — `‖I - Q_{ord,c}^H Q_{ord,c}‖₂`
- `fact_defect` — `‖A Q_{ord,c} - Q_{ord,c} T_{ord,c}‖₂`

# Example
```julia
A = BallMatrix(randn(ComplexF64, 5, 5))
Q_ball, T_ball, _ = rigorous_schur_bigfloat(A)
select = [true, true, false, false, false]
result = ordschur_ball(Q_ball, T_ball, select)
result.T  # reordered Schur form with rigorous radii
```
"""
function ordschur_ball(Q_ball::BallMatrix, T_ball::BallMatrix,
                       select::AbstractVector{Bool})
    n = size(T_ball, 1)
    n == size(T_ball, 2) || throw(DimensionMismatch("T_ball must be square"))
    n == size(Q_ball, 1) == size(Q_ball, 2) || throw(DimensionMismatch("Q_ball must be n×n"))
    length(select) == n || throw(DimensionMismatch("select must have length n"))

    ET = eltype(mid(T_ball))   # e.g. Complex{BigFloat}
    RT = real(ET)              # e.g. BigFloat

    # Step 1: Compute ordschur on midpoints
    Q_ord_c, T_ord_c, vals = ordschur_bigfloat(mid(T_ball), mid(Q_ball), select)

    # Step 2: Accumulated unitary transformation G_c = Q_c^H * Q_ord_c
    G_c = mid(Q_ball)' * Q_ord_c

    # Step 3: Rigorous verification residuals
    # 3a. Orthogonality defect of Q_ord_c
    GtG_minus_I = G_c' * G_c - Matrix{ET}(I, n, n)
    orth_defect = upper_bound_L2_opnorm(BallMatrix(GtG_minus_I))

    if orth_defect > sqrt(eps(RT))
        @warn "ordschur_ball: large orthogonality defect $(Float64(orth_defect)) in G"
    end

    # 3b. Factorization residual: A * Q_ord_c - Q_ord_c * T_ord_c
    #     where A ∈ Q_ball * T_ball * Q_ball^H
    #     Compute as Q_ball * (T_ball * (Q_ball^H * Q_ord_c)) - Q_ord_c * T_ord_c
    Q_ord_exact = BallMatrix(Q_ord_c)   # zero radii
    T_ord_exact = BallMatrix(T_ord_c)   # zero radii
    QtQ_ord = Q_ball' * Q_ord_exact     # ≈ G in ball arithmetic
    AQ_ord = Q_ball * (T_ball * QtQ_ord)
    fact_res = AQ_ord - Q_ord_exact * T_ord_exact
    fact_defect = upper_bound_L2_opnorm(fact_res)

    # Step 4: Build G_ball with orthogonality defect as componentwise radius
    # If ‖G_c^H G_c - I‖₂ ≤ δ, then there exists a unitary U with ‖U - G_c‖₂ ≤ δ
    # and |U_ij - (G_c)_ij| ≤ ‖U - G_c‖₂ ≤ δ
    G_ball = BallMatrix(G_c, fill(orth_defect, n, n))

    # Step 5: Propagate in ball arithmetic — rigorous enclosures
    T_ord_ball = G_ball' * T_ball * G_ball
    Q_ord_ball = Q_ball * G_ball

    return (Q=Q_ord_ball, T=T_ord_ball, values=vals,
            orth_defect=orth_defect, fact_defect=fact_defect)
end

"""
    spectral_projector_error_bound(; resolvent_bound_A, contour_radius,
                                     orth_defect, fact_defect)

Rigorous upper bound on `‖P_A - P_computed‖₂`, the distance between the true
spectral projector of `A` and the projector computed from the Miyajima
Sylvester solve on the point Schur form `T̃`.

# Mathematical Background

The true spectral projector is the contour integral of the resolvent:

    P_A = (1/2πi) ∮_Γ (zI - A)⁻¹ dz

The computed projector uses the reordered Schur form:

    P_computed = Q_ord · [I  Y; 0  0] · Q_ord^H
              = (1/2πi) ∮_Γ Q_ord (zI - T̃)⁻¹ Q_ord^H dz

The resolvent identity gives:

    (zI-A)⁻¹ - Q_ord(zI-T̃)⁻¹Q_ord^H
        = (zI-A)⁻¹(I - Q_ord Q_ord^H) + (zI-A)⁻¹ R (zI-T̃)⁻¹ Q_ord^H

where `R = A Q_ord - Q_ord T̃` is the factorization residual. Integrating over
the circle Γ of radius `r`:

    ‖P_A - P_computed‖ ≤ r · M_A · (δ + ε · M_T̃ · √(1+δ))

where:
- `M_A = max_{z∈Γ} ‖(zI-A)⁻¹‖` — input resolvent bound
- `δ` — orthogonality defect `‖I - Q_ord^H Q_ord‖`
- `ε` — factorization defect `‖A Q_ord - Q_ord T̃‖`
- `M_T̃` — resolvent of `T̃`, bounded from `M_A` via:

    (zI - T̃) = (I-E)⁻¹ [Q_ord^H(zI-A)Q_ord + Q_ord^H R]

    M_T̃ ≤ M_A · (1+δ) / [(1-δ)(1-γ)]

  with `γ = M_A · √(1+δ) · ε / (1-δ)`.

# Arguments (keyword)
- `resolvent_bound_A::Real`: `max_{z∈Γ} ‖(zI-A)⁻¹‖₂` on the enclosing circle
- `contour_radius::Real`: radius `r` of the circular contour Γ
- `orth_defect::Real`: `‖I - Q_ord^H Q_ord‖₂` (from `ordschur_ball`)
- `fact_defect::Real`: `‖A Q_ord - Q_ord T̃‖₂` (from `ordschur_ball`)

# Returns
Scalar upper bound on `‖P_A - P_computed‖₂`. Returns `Inf` if the Neumann
series conditions are not met (γ ≥ 1 or δ ≥ 1).

# Example
```julia
# After ordschur_ball and CertifScripts resolvent bound on a circle:
bound = spectral_projector_error_bound(
    resolvent_bound_A = 42.0,    # from CertifScripts
    contour_radius = 0.5,        # circle separating eigenvalue clusters
    orth_defect = 1e-77,         # from ordschur_ball
    fact_defect = 1e-75          # from ordschur_ball
)
```
"""
function spectral_projector_error_bound(; resolvent_bound_A::Real,
                                          contour_radius::Real,
                                          orth_defect::Real,
                                          fact_defect::Real)
    M_A = resolvent_bound_A
    r   = contour_radius
    δ   = orth_defect
    ε   = fact_defect
    RT  = promote_type(typeof(M_A), typeof(r), typeof(δ), typeof(ε))

    if δ >= one(RT)
        return RT(Inf)
    end

    # ‖Q_ord‖ ≤ √(1+δ) from Q^H Q = I - E with ‖E‖ ≤ δ
    σ_max_Q = sqrt(one(RT) + δ)

    # Step 1: Bound ‖(zI - T̃)⁻¹‖ from M_A and residuals
    #
    # From (I-E)(zI-T̃) = Q^H(zI-A)Q + Q^H R:
    #   F = Q^H(zI-A)Q  ⟹  ‖F⁻¹‖ ≤ M_A / (1-δ)
    #   γ  = ‖F⁻¹ Q^H R‖ ≤ M_A · √(1+δ) · ε / (1-δ)
    #   M_T̃ = ‖(zI-T̃)⁻¹‖ ≤ M_A · (1+δ) / [(1-δ)(1-γ)]

    F_inv_bound = M_A / (one(RT) - δ)
    γ = F_inv_bound * σ_max_Q * ε

    if γ >= one(RT)
        return RT(Inf)
    end

    M_T = F_inv_bound * (one(RT) + δ) / (one(RT) - γ)

    # Step 2: Projector error bound via contour integral
    #
    # ‖P_A - P_computed‖ ≤ r · M_A · (δ + ε · M_T̃ · √(1+δ))
    #
    # Two terms from the resolvent identity:
    #   (zI-A)⁻¹(I - Q Q^H)           → M_A · δ
    #   (zI-A)⁻¹ R (zI-T̃)⁻¹ Q^H     → M_A · ε · M_T̃ · ‖Q‖

    return r * M_A * (δ + ε * M_T * σ_max_Q)
end

export ordschur_bigfloat, ordschur_ball, spectral_projector_error_bound
