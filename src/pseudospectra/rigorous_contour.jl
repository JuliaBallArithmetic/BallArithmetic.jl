# Chains of discs along a curve, with a certified lower bound of σ_min(T − zI) on each disc.
#
# The curve is followed in floating point (a circle, or a level curve of σ_min reached and
# followed by Newton and Euler steps). What is certified is, for each disc, the smallest singular
# value of the ball matrix T − B(c, r)·I, and, for the chain, that consecutive discs overlap.

export Enclosure

"""
    Enclosure

A chain of discs with a certified bound on each.

- `λ`: the point the chain was built around
- `points[i]`, `radiuses[i]`: centre and radius of disc `i`
- `bounds[i]`: a ball containing `σ_min(T − zI)` for every `z` of disc `i`
- `loop_closure`: `true` when every disc is proved to overlap the next and the last the first.
  The closed polygon through the centres is then contained in the union of the discs, since a
  segment between two centres at distance less than the sum of the radii is covered by the two
  discs. That the polygon winds around `λ` is not checked.

The bounds are for the matrix `T` the chain was built from.
"""
struct Enclosure
    λ::Any
    points::Vector{ComplexF64}
    bounds::Vector{Ball{Float64, Float64}}
    radiuses::Vector{Float64}
    loop_closure::Bool
end

# the ball for σ_min(T − zI) over the disc of centre c and radius r, from the approximate SVD K
function _disc_bound(T, c, r, K, svd_method, apply_vbd)
    return _certify_svd(BallMatrix(T) - Ball(c, r) * I, K, svd_method; apply_vbd)[end]
end

# consecutive discs overlap, and the last overlaps the first: distances rounded up, sums down
function _chain_closed(points, radiuses)
    N = length(points)
    N >= 2 || return false
    for i in 1:N
        j = i == N ? 1 : i + 1
        dist_up(points[i], points[j]) < add_down(radiuses[i], radiuses[j]) || return false
    end
    return true
end

_enclosure(λ, points, bounds, radiuses) = Enclosure(λ, ComplexF64.(points),
    Ball{Float64, Float64}.(bounds), Float64.(radiuses), _chain_closed(points, radiuses))

"""
    check_enclosure(E::Enclosure)

Whether every disc of `E` overlaps the next and the last overlaps the first, recomputed from the
centres and radii with directed rounding.
"""
check_enclosure(E::Enclosure) = _chain_closed(E.points, E.radiuses)

"""
    bound_resolvent(E::Enclosure)

An upper bound of `‖(zI − T)⁻¹‖₂` for every `z` in the union of the discs of `E`: the reciprocal
of the smallest lower end of the bounds, rounded up; `Inf` when one of them is not positive.
"""
function bound_resolvent(E::Enclosure)
    lo = minimum(sub_down(mid(b), rad(b)) for b in E.bounds)
    return lo > 0 ? div_up(1.0, lo) : Inf
end

# N discs centred at equispaced points of the circle of centre λ and radius r, each of radius 5/8
# of the arc between two consecutive centres
function _certify_circle(T, λ, r, N; svd_method::SVDMethod = MiyajimaM1(),
        apply_vbd::Bool = false)
    N = Int(N)
    points = ComplexF64[]
    bounds = Ball{Float64, Float64}[]
    radiuses = Float64[]
    pearl_radius = 5 * (r * 2 * π / N) / 8
    for j in 0:(N - 1)
        z = ComplexF64(λ + r * cispi(2 * j / N))
        push!(points, z)
        push!(bounds, _disc_bound(T, z, pearl_radius, svd(T - z * I), svd_method, apply_vbd))
        push!(radiuses, pearl_radius)
    end
    return _enclosure(λ, points, bounds, radiuses)
end

# A chain built step by step: `advance(z, K)` gives the next point from the current one and the SVD
# of T − zI. The disc of a step is centred at the current point, with radius 5/8 of the distance
# to the next. The walk stops when the current disc reaches the first one (after ten steps at
# least) or after `max_steps`; whether the chain closes is decided by `_chain_closed`.
function _walk_chain(advance, T, λ, z0, max_steps, svd_method, apply_vbd)
    points = ComplexF64[]
    bounds = Ball{Float64, Float64}[]
    radiuses = Float64[]
    z = ComplexF64(z0)
    for t_step in 1:max_steps
        K = svd(T - z * I)
        z_next = ComplexF64(advance(z, K))
        r = mul_up(dist_up(z, z_next), 0.625)
        push!(points, z)
        push!(bounds, _disc_bound(T, z, r, K, svd_method, apply_vbd))
        push!(radiuses, r)
        t_step > 10 && abs(z - points[1]) < r + radiuses[1] && break
        z = z_next
    end
    return _enclosure(λ, points, bounds, radiuses)
end

# along the circle of centre λ and radius r, with steps proportional to the distance from the
# nearest diagonal entry of T
function _compute_exclusion_set(T, r; max_steps, rel_steps, λ = 0 + im * 0,
        svd_method::SVDMethod = MiyajimaM1(), apply_vbd::Bool = false)
    eigvals = diag(T)
    advance = function (z, K)
        τ = minimum(abs.(eigvals .- z)) / rel_steps
        w = z + τ * im * (z - λ) / abs(z - λ)
        return w - (abs(w - λ)^2 - r^2) / conj(w - λ)      # back to the circle
    end
    return _walk_chain(advance, T, λ, λ + r, max_steps, svd_method, apply_vbd)
end

_compute_exclusion_circle(T, λ, r; max_steps, rel_steps,
    svd_method::SVDMethod = MiyajimaM1(), apply_vbd::Bool = false) =
    _compute_exclusion_set(T, r; max_steps, rel_steps, λ, svd_method, apply_vbd)

# one Euler step along the level curve of σ_min through z
function _follow_level_set(z::ComplexF64, τ::Float64, K::SVD)
    u = K.U[:, end]
    v = K.V[:, end]
    σ = K.S[end]
    ort = im * (v' * u)
    return z + τ * ort / abs(ort), σ
end

# one Newton step towards the level curve σ_min = ϵ
function _newton_step(z, K::SVD, ϵ)
    u = K.U[:, end]
    v = K.V[:, end]
    σ = K.S[end]
    return z + (σ - ϵ) / (u' * v), σ
end

# a point near the level curve σ_min(T − zI) = ϵ, by Newton steps from z
function _reach_level_set(T, z, ϵ, max_initial_newton)
    for _ in 1:max_initial_newton
        z, σ = _newton_step(z, svd(T - z * I), ϵ)
        (σ - ϵ) / ϵ < 1 / 256 && break
    end
    return z
end

# the circle of centre λ through a point of the ϵ level curve, walked with adaptive steps
function _compute_exclusion_circle_level_set_ode(T, λ, ϵ; max_steps, rel_steps,
        max_initial_newton, svd_method::SVDMethod = MiyajimaM1(), apply_vbd::Bool = false)
    r = abs(λ - _reach_level_set(T, λ + ϵ, ϵ, max_initial_newton))
    return _compute_exclusion_circle(T, λ, r; max_steps, rel_steps, svd_method, apply_vbd)
end

# The same circle with equispaced discs of radius `rel_pearl_size` times the radius of the circle:
# with 1/32 there are 160 discs, with 1/64 there are 320.
function _compute_exclusion_circle_level_set_priori(T, λ, ϵ; rel_pearl_size,
        max_initial_newton, svd_method::SVDMethod = MiyajimaM1(), apply_vbd::Bool = false)
    r = abs(λ - _reach_level_set(T, λ + ϵ, ϵ, max_initial_newton))
    dist_points = (r * rel_pearl_size * 8) / 5
    N = ceil(8 * r / dist_points)                  # above 2πr/dist_points
    return _certify_circle(T, λ, r, N; svd_method, apply_vbd)
end

# the ϵ level curve of σ_min around the diagonal entry λ of T: an Euler step along the curve and a
# Newton step back to it
function _compute_enclosure_eigval(T, λ, ϵ; max_initial_newton, max_steps, rel_steps,
        svd_method::SVDMethod = MiyajimaM1(), apply_vbd::Bool = false)
    eigvals = diag(T)
    z0 = _reach_level_set(T, λ + 4 * sign(real(λ)) * ϵ, ϵ, max_initial_newton)
    advance = function (z, K)
        τ = minimum(abs.(eigvals .- z)) / rel_steps
        w, _ = _follow_level_set(z, τ, K)
        return first(_newton_step(w, K, ϵ))
    end
    return _walk_chain(advance, T, λ, z0, max_steps, svd_method, apply_vbd)
end

"""
    compute_enclosure(A::BallMatrix, r1, r2, ϵ; max_initial_newton = 30,
        max_steps = Int64(ceil(256 * π)), rel_steps = 16,
        svd_method = MiyajimaM1(), apply_vbd = false)

Chains of discs around the spectrum of the Schur factor `T` of the midpoint of `A` (computed in
`Float64`), each returned as an [`Enclosure`](@ref):

- one around each diagonal entry `λ` of `T` with `r1 < |λ| < r2`, following the level curve
  `σ_min(T − zI) = ϵ`;
- one along the circle `|z| = r1` when `T` has diagonal entries of modulus below `r1`;
- one along the circle `|z| = r2` when it has diagonal entries of modulus above `r2`.

`max_initial_newton` bounds the Newton steps used to reach a level curve, `max_steps` the length of
a chain, and `rel_steps` sets the step relative to the distance from the nearest diagonal entry.

The bounds are for `T`, not for `A`: the radius of `A` and the defects of the Schur factorisation
are not used. `CertifScripts.run_certification` certifies a circle for `A` itself.
[`bound_resolvent`](@ref) gives the bound on the union of the discs of a chain, and its
`loop_closure` says whether the chain is proved closed.
"""
function compute_enclosure(A::BallMatrix{T}, r1, r2, ϵ; max_initial_newton = 30,
        max_steps = Int64(ceil(256 * π)), rel_steps = 16,
        svd_method::SVDMethod = MiyajimaM1(), apply_vbd::Bool = false) where {T}
    F = schur(Complex{Float64}.(A.c))
    d = diag(F.T)
    output = Enclosure[]
    for λ in d[[r1 < abs(x) < r2 for x in d]]
        push!(output, _compute_enclosure_eigval(F.T, λ, ϵ; max_initial_newton,
            max_steps, rel_steps, svd_method, apply_vbd))
    end
    if any(x -> abs(x) < r1, d)
        push!(output, _compute_exclusion_set(F.T, r1; max_steps, rel_steps, svd_method, apply_vbd))
    end
    if any(x -> abs(x) > r2, d)
        push!(output, _compute_exclusion_set(F.T, r2; max_steps, rel_steps, svd_method, apply_vbd))
    end
    return output
end
