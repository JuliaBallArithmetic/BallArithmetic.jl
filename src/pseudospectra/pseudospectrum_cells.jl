# A box of the complex plane divided into cells proved outside the ε-pseudospectrum, proved inside
# it, and undecided, from certified bounds of σ_min(A − zI) at the centres of the cells.

export PseudospectrumCell, PseudospectrumCells, pseudospectrum_cells, sigma_min_upper

"""
    PseudospectrumCell{T}

The closed rectangle `{z : |Re(z − centre)| ≤ hx, |Im(z − centre)| ≤ hy}`.
"""
struct PseudospectrumCell{T <: AbstractFloat}
    centre::Complex{T}
    hx::T
    hy::T
end

"""
    PseudospectrumCells{T}

Returned by [`pseudospectrum_cells`](@ref): `excluded`, cells on which `σ_min(A − zI) > ε` at
every point; `included`, cells on which `σ_min(A − zI) < ε` at every point; `undecided`, the
others; `evaluations`, the number of cell centres at which the bounds were computed.
"""
struct PseudospectrumCells{T <: AbstractFloat}
    excluded::Vector{PseudospectrumCell{T}}
    included::Vector{PseudospectrumCell{T}}
    undecided::Vector{PseudospectrumCell{T}}
    epsilon::T
    evaluations::Int
end

# An upper bound of the distance from the centre of a cell to any of its points, with a margin of
# two units in the last place of the centre: the centres of the four parts of a cell are rounded,
# so the parts need not tile it exactly, and the margin makes the discs of the parts cover it.
function _cell_radius(c::PseudospectrumCell{T}) where {T}
    r = sqrt_up(add_up(mul_up(c.hx, c.hx), mul_up(c.hy, c.hy)))
    return add_up(r, mul_up(T(2), eps(max(abs(real(c.centre)), abs(imag(c.centre)), floatmin(T)))))
end

function _quadrisect(c::PseudospectrumCell{T}) where {T}
    hx, hy = c.hx / 2, c.hy / 2
    return (PseudospectrumCell(c.centre + complex(-hx, -hy), hx, hy),
        PseudospectrumCell(c.centre + complex(hx, -hy), hx, hy),
        PseudospectrumCell(c.centre + complex(-hx, hy), hx, hy),
        PseudospectrumCell(c.centre + complex(hx, hy), hx, hy))
end

"""
    pseudospectrum_cells(lower, upper, lo, hi; ε, min_halfdiag, max_evaluations = 100_000)
        -> PseudospectrumCells

Divide the rectangle with opposite corners `lo` and `hi` into cells that lie outside the set
`{z : σ_min(A − zI) ≤ ε}`, cells that lie inside `{z : σ_min(A − zI) < ε}`, and cells left
undecided.

`lower(z)` and `upper(z)` are functions returning certified bounds
`lower(z) ≤ σ_min(A − zI) ≤ upper(z)`; `upper` may return `Inf` and `lower` zero. For instance
`z -> sigma_min_floor(f, z)` with `f` from [`block_resolvent_floor`](@ref) or
[`svd_frame_floor`](@ref), or the larger of several such, and `z -> sigma_min_upper(A, z)`.

# What is proved

`z ↦ σ_min(A − zI)` is 1-Lipschitz, since `|σ_min(X + E) − σ_min(X)| ≤ ‖E‖₂` and
`‖(z − w)I‖₂ = |z − w|`. For a cell with centre `c` whose points are within `r` of `c`:

- if `lower(c) − r > ε` then `σ_min(A − zI) > ε` on the cell, which is `excluded`;
- if `upper(c) + r < ε` then `σ_min(A − zI) < ε` on the cell, which is `included`.

A cell that is neither is cut into four, down to cells whose `r` is below `min_halfdiag`, which
are returned as `undecided`; so are the cells not yet examined when `max_evaluations` centres
have been used. `r` is rounded up and carries a margin for the rounding of the centres of the
parts, so that the discs of radius `r` about the centres of the returned cells cover the
rectangle. Cells accumulate along the level curve `σ_min = ε` and not over the area.

No use is made of the maximum principle for the resolvent norm on cells free of eigenvalues,
which would let the bound on the boundary of such a cell serve for its interior.
"""
function pseudospectrum_cells(lower, upper, lo::Number, hi::Number; ε::Real,
        min_halfdiag::Real, max_evaluations::Integer = 100_000)
    T = float(promote_type(typeof(real(lo)), typeof(real(hi)), typeof(ε)))
    (ε > 0 && min_halfdiag > 0) ||
        throw(ArgumentError("pseudospectrum_cells: ε and min_halfdiag must be positive"))
    l, h = Complex{T}(lo), Complex{T}(hi)
    hx, hy = abs(real(h) - real(l)) / 2, abs(imag(h) - imag(l)) / 2
    (hx > 0 && hy > 0) || throw(ArgumentError("pseudospectrum_cells: the rectangle is degenerate"))
    # the half sizes are enlarged by one unit so that the root cell contains the rectangle
    root = PseudospectrumCell((l + h) / 2, nextfloat(hx), nextfloat(hy))
    excluded = PseudospectrumCell{T}[]
    included = PseudospectrumCell{T}[]
    undecided = PseudospectrumCell{T}[]
    stack = [root]
    evaluations = 0
    εT = T(ε)
    while !isempty(stack)
        cell = pop!(stack)
        if evaluations >= max_evaluations
            push!(undecided, cell)
            continue
        end
        r = _cell_radius(cell)
        evaluations += 1
        if sub_down(T(lower(cell.centre)), r) > εT
            push!(excluded, cell)
        elseif add_up(T(upper(cell.centre)), r) < εT
            push!(included, cell)
        elseif r < min_halfdiag
            push!(undecided, cell)
        else
            append!(stack, _quadrisect(cell))
        end
    end
    return PseudospectrumCells{T}(excluded, included, undecided, εT, evaluations)
end

"""
    sigma_min_upper(A::BallMatrix, z)

An upper bound of `σ_min(M − zI)` for every matrix `M` of the ball `A`: with `v` a floating-point
right singular vector of the midpoint of `A − zI` for its smallest singular value,
`σ_min(M − zI) ≤ ‖(M − zI)v‖₂/‖v‖₂`, the numerator bounded above and the denominator below in
ball arithmetic. Any nonzero `v` gives a valid bound; the singular vector is the one that makes
it close. One singular value decomposition in floating point for each `z`.
"""
function sigma_min_upper(A::BallMatrix{T}, z::Number) where {T}
    n = size(A, 1)
    n == size(A, 2) || throw(ArgumentError("sigma_min_upper: A must be square"))
    zc = Complex{T}(z)
    v = Vector{Complex{T}}(svd(Matrix{Complex{T}}(mid(A)) - zc * I).V[:, n])
    w = _shifted_ball(zc, A) * BallMatrix(reshape(v, n, 1))
    num = zero(T)
    for (m, ρ) in zip(mid(w), rad(w))
        a = add_up(abs_up(m), ρ)
        num = add_up(num, mul_up(a, a))
    end
    den = zero(T)
    for x in v
        a = abs_down(x)
        den = add_down(den, mul_down(a, a))
    end
    den > 0 || return T(Inf)
    return div_up(sqrt_up(num), sqrt_down(den))
end
