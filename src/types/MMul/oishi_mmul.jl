"""
    _oishi_MMul_up_lo(F::AbstractMatrix{<:Complex}, G::AbstractMatrix{<:Complex})

Compute outward lower/upper enclosures for the real and imaginary parts of the
complex matrix product `F*G` using directed-rounding (rounding-mode controlled)
evaluation in the sense of Oishi–Rump.

Returns `(Hrl, Hru, Hil, Hiu, T)`, where `Hrl ≤ Re(F*G) ≤ Hru` and
`Hil ≤ Im(F*G) ≤ Hiu` hold entrywise, and `T` is the promoted underlying real
working type.

Promotion follows `T = promote_type(float(real(eltype(F))), float(real(eltype(G))))`.
For mathematical correctness, `T` must honor `setrounding` (e.g., `BigFloat`).

Reference:
- “Fast enclosure of matrix eigenvalues and singular values via rounding mode controlled
  computation,” *Linear Algebra and its Applications* **324** (2001), 133–146.
"""
function _oishi_MMul_up_lo(F::AbstractMatrix{<:Complex}, G::AbstractMatrix{<:Complex})
    size(F, 2) == size(G, 1) || throw(DimensionMismatch("size(F,2) must equal size(G,1)"))

    TF = float(typeof(real(zero(eltype(F)))))  # underlying real of F
    TG = float(typeof(real(zero(eltype(G)))))  # underlying real of G
    T = promote_type(TF, TG)                  # e.g. Float64 or BigFloat
    CT = Complex{T}

    F = CT.(F)
    G = CT.(G)

    Fr, Fi = real.(F), imag.(F)
    Gr, Gi = real.(G), imag.(G)

    # lower bounds (RoundDown)
    Hrl, Hil = setrounding(T, RoundDown) do
        Hrl = Fr * Gr + (-Fi) * Gi            # Re = Fr*Gr - Fi*Gi
        Hil = Fr * Gi + Fi * Gr             # Im = Fr*Gi + Fi*Gr
        return Hrl, Hil
    end

    # upper bounds (RoundUp)
    Hru, Hiu = setrounding(T, RoundUp) do
        Hru = Fr * Gr + (-Fi) * Gi
        Hiu = Fr * Gi + Fi * Gr
        return Hru, Hiu
    end

    return Hrl, Hru, Hil, Hiu, T
end

"""
    _ccr(Hrl, Hru, Hil, Hiu, ::Type{T}) where {T<:AbstractFloat}

Collapse rectangular enclosures for `Re(F*G)` and `Im(F*G)` to a complex
ball enclosure `BallMatrix(Hc, Hr)`.

Centers are midpoints `Rc = (Hru+Hrl)/2`, `Ic = (Hiu+Hil)/2` computed with
`RoundNearest`. Radii are the outward 2-norm of the half-widths:
`Hr = sqrt(((Hru-Hrl)/2).^2 + ((Hiu-Hil)/2).^2)` computed with `RoundUp`.

This implements the ball conversion step used with the Oishi–Rump
rounding-mode controlled product.

Reference:
- “Fast enclosure of matrix eigenvalues and singular values via rounding mode controlled
  computation,” *Linear Algebra and its Applications* **324** (2001), 133–146.
"""
function _ccr(Hrl::AbstractMatrix{<:Real}, Hru::AbstractMatrix{<:Real},
        Hil::AbstractMatrix{<:Real}, Hiu::AbstractMatrix{<:Real}, ::Type{T}) where {T <:
                                                                                    AbstractFloat}
    half = T(0.5)

    # centers at midpoints (nearest rounding is fine/tight)
    Rc, Ic = setrounding(T, RoundNearest) do
        (Hru .+ Hrl) .* half, (Hiu .+ Hil) .* half
    end

    # radii are half-widths combined in 2-norm, rounded upward
    Hr = setrounding(T, RoundUp) do
        Rr = (Hru .- Hrl) .* half
        Ir = (Hiu .- Hil) .* half
        sqrt.(Rr .^ 2 .+ Ir .^ 2)
    end

    Hc = complex.(Rc, Ic)
    return BallMatrix(Hc, Hr)  # or return (Hc, Hr)
end

"""
    oishi_MMul(F, G)

High-level wrapper: compute a complex ball enclosure of the product `F*G`
via the Oishi–Rump rounding-mode controlled method. Internally calls
`_oishi_MMul_up_lo` to get componentwise rectangular bounds, then `_ccr`
to convert them into a `BallMatrix`.

Reference:
- “Fast enclosure of matrix eigenvalues and singular values via rounding mode controlled
  computation,” *Linear Algebra and its Applications* **324** (2001), 133–146.
"""
function oishi_MMul(F, G)
    Hrl, Hru, Hil, Hiu, T = _oishi_MMul_up_lo(F, G)
    return _ccr(Hrl, Hru, Hil, Hiu, T)
end
