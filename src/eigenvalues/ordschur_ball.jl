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

# ── Ball-arithmetic Givens primitives ──────────────────────────────────

"""
    _ball_abs_s(s, RT)

Compute a rigorous upper bound on |s| for a Givens rotation parameter `s`.
Uses directed rounding for BigFloat; plain `abs` otherwise.
"""
function _ball_abs_s(s::Complex{T}, ::Type{T}) where {T<:BigFloat}
    setrounding(T, RoundUp) do
        sqrt(real(s) * real(s) + imag(s) * imag(s))
    end
end
_ball_abs_s(s::Complex{T}, ::Type{T}) where {T<:AbstractFloat} = abs(s)
_ball_abs_s(s::T, ::Type{T}) where {T<:Real} = abs(s)

"""
    _ball_givens_lmul!(c, s, abs_s, k, M_mid, M_rad, eps_RT)

Apply a Givens rotation `G(k, k+1)` from the left: `M ← G * M`.
Operates in-place on rows `k` and `k+1` of `(M_mid, M_rad)`.

`c` is real ≥ 0, `s` is the Givens sine, `abs_s = |s|` (precomputed with RoundUp),
and `eps_RT` is the machine epsilon for the real type.

Radius formula: for exact `c, s` applied to ball `[m ± r]`:
  new_r = c * r_k + |s| * r_{k+1}  +  eps * (c * |old_mid_k| + |s| * |old_mid_{k+1}|)
The last term bounds floating-point error in the midpoint computation.
"""
function _ball_givens_lmul!(c::RT, s, abs_s::RT, k::Int,
                            M_mid::AbstractMatrix, M_rad::AbstractMatrix{RT},
                            eps_RT::RT) where {RT<:AbstractFloat}
    n = size(M_mid, 2)
    sc = conj(s)
    @inbounds for j in 1:n
        m_k  = M_mid[k, j]
        m_k1 = M_mid[k+1, j]
        r_k  = M_rad[k, j]
        r_k1 = M_rad[k+1, j]

        # Midpoint update (default rounding)
        M_mid[k, j]   =  c * m_k + s * m_k1
        M_mid[k+1, j] = -sc * m_k + c * m_k1

        # Radius update (outward rounding)
        abs_mk  = abs(m_k)
        abs_mk1 = abs(m_k1)
        setrounding(RT, RoundUp) do
            inp_k  = c * abs_mk + abs_s * abs_mk1   # bound on |exact result|
            M_rad[k, j]   = (c * r_k + abs_s * r_k1) + eps_RT * inp_k
            M_rad[k+1, j] = (abs_s * r_k + c * r_k1) + eps_RT * inp_k
        end
    end
    return nothing
end

"""
    _ball_givens_rmul_adj!(c, s, abs_s, k, M_mid, M_rad, eps_RT)

Apply `M ← M * G(k, k+1)^H` in-place on columns `k` and `k+1`.
Same radius logic as `_ball_givens_lmul!`, transposed.
"""
function _ball_givens_rmul_adj!(c::RT, s, abs_s::RT, k::Int,
                                M_mid::AbstractMatrix, M_rad::AbstractMatrix{RT},
                                eps_RT::RT) where {RT<:AbstractFloat}
    m = size(M_mid, 1)
    sc = conj(s)
    @inbounds for i in 1:m
        m_k  = M_mid[i, k]
        m_k1 = M_mid[i, k+1]
        r_k  = M_rad[i, k]
        r_k1 = M_rad[i, k+1]

        # M * G^H: column k  ←  c * col_k + conj(s) * col_{k+1}
        #          column k+1 ← -s * col_k + c * col_{k+1}
        M_mid[i, k]   =  c * m_k + sc * m_k1
        M_mid[i, k+1] = -s * m_k + c * m_k1

        abs_mk  = abs(m_k)
        abs_mk1 = abs(m_k1)
        setrounding(RT, RoundUp) do
            inp_k = c * abs_mk + abs_s * abs_mk1
            M_rad[i, k]   = (c * r_k + abs_s * r_k1) + eps_RT * inp_k
            M_rad[i, k+1] = (abs_s * r_k + c * r_k1) + eps_RT * inp_k
        end
    end
    return nothing
end

"""
    _ball_trexchange!(T_mid, T_rad, Q_mid, Q_rad, iold, inew, eps_RT)

Move the eigenvalue at position `iold` to position `inew` via a sequence
of Givens rotations applied directly to the ball-arithmetic arrays.
Mirrors GenericSchur's `_trexchange!`.
"""
function _ball_trexchange!(T_mid, T_rad, Q_mid, Q_rad, iold::Int, inew::Int,
                           eps_RT)
    RT = typeof(eps_RT)
    krange = iold > inew ? (iold-1:-1:inew) : (iold:inew-1)
    for k in krange
        # Givens to annihilate T[k+1,k] after swapping diagonal entries
        G, _ = givens(T_mid[k, k+1], T_mid[k+1, k+1] - T_mid[k, k], k, k+1)
        c = real(G.c)
        s = G.s
        abs_s = _ball_abs_s(s, RT)

        # T ← G * T * G^H
        _ball_givens_lmul!(c, s, abs_s, k, T_mid, T_rad, eps_RT)
        _ball_givens_rmul_adj!(c, s, abs_s, k, T_mid, T_rad, eps_RT)

        # Q ← Q * G^H
        _ball_givens_rmul_adj!(c, s, abs_s, k, Q_mid, Q_rad, eps_RT)
    end
    return nothing
end

# ── End Givens primitives ─────────────────────────────────────────────

"""
    ordschur_ball(Q_ball::BallMatrix, T_ball::BallMatrix,
                  select::AbstractVector{Bool})

Reorder a rigorous Schur decomposition enclosed in `BallMatrix` form.

Given `(Q_ball, T_ball)` enclosing the true Schur factors (e.g. from
[`rigorous_schur_bigfloat`](@ref)), reorder so that eigenvalues with
`select[i] == true` move to the top-left block, and return rigorous
`BallMatrix` enclosures of the reordered factors.

# Algorithm (incremental Givens)
Instead of computing `ordschur` on midpoints and then propagating through
full O(n³) ball-matrix multiplies, this applies each Givens rotation of the
reordering directly to the `(mid, rad)` arrays. This is O(kn) rotations at
O(n) ball operations each — dramatically faster for large matrices.

The radius propagation through each Givens rotation is rigorous: for exact
rotation parameters `(c, s)` applied to a ball `[m ± r]`, the output radius
bounds both the input-radius contribution and the floating-point rounding
error of the midpoint computation.

# Returns
A `NamedTuple` with fields:
- `Q::BallMatrix` — rigorous enclosure of the reordered Schur basis
- `T::BallMatrix` — rigorous enclosure of the reordered Schur form
- `values::Vector` — reordered eigenvalues (midpoint only)
- `orth_defect` — zero (tracked in radii)
- `fact_defect` — zero (tracked in radii)

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
    eps_RT = machine_epsilon(RT)

    # Work on copies of mid/rad arrays
    T_mid = copy(mid(T_ball))
    T_rad = copy(rad(T_ball))
    Q_mid = copy(mid(Q_ball))
    Q_rad = copy(rad(Q_ball))

    # Incremental ordschur: bubble selected eigenvalues to top-left
    ks = 0
    for k in 1:n
        if select[k]
            ks += 1
            if k != ks
                _ball_trexchange!(T_mid, T_rad, Q_mid, Q_rad, k, ks, eps_RT)
            end
        end
    end

    # Enforce upper triangularity: absorb subdiagonal midpoints into radii
    for i in 2:n, j in 1:i-1
        T_rad[i, j] += abs(T_mid[i, j])
        T_mid[i, j] = zero(ET)
    end

    return (Q=BallMatrix(Q_mid, Q_rad), T=BallMatrix(T_mid, T_rad),
            values=diag(T_mid), orth_defect=zero(RT), fact_defect=zero(RT))
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
