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

# Internal helper used by the Miyajima kernels below.  The name mirrors the
# terminology in Ref. [Miyajima2010](@cite), where the routine is referred to as
# the “complex product” computed with directed rounding.
_cprod(F::AbstractMatrix{<:Complex}, G::AbstractMatrix{<:Complex}) =
    _oishi_MMul_up_lo(F, G)

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
    _ccrprod(J::AbstractMatrix{<:Complex}, Hc::AbstractMatrix{<:Complex}, Hr::AbstractMatrix{<:Real})

Implement Algorithm 4 of Ref. [Miyajima2010](@cite).  Given a complex matrix
`J` and a ball enclosure `(Hc, Hr)` for another complex matrix, return
rectangular bounds `(Krl, Kru, Kil, Kiu)` and the working type `T` such that
`Krl ≤ Re(J * (Hc ± Hr)) ≤ Kru` and `Kil ≤ Im(J * (Hc ± Hr)) ≤ Kiu` hold
entrywise.  All computations are performed with outward rounding using the
promoted working type `T`.
"""
function _ccrprod(J::AbstractMatrix{<:Complex},
        Hc::AbstractMatrix{<:Complex}, Hr::AbstractMatrix{<:Real})
    TJ = float(typeof(real(zero(eltype(J)))))  # underlying real of J
    THc = float(typeof(real(zero(eltype(Hc)))))  # underlying real of Hc
    THr = float(typeof(zero(eltype(Hr))))
    T = promote_type(TJ, THc, THr)  # e.g. Float64 or BigFloat
    CT = Complex{T}
    J = CT.(J)
    Hc = CT.(Hc)
    Hr = T.(Hr)

    Mrl, Mru, Mil, Miu, Tprod = _cprod(J, Hc)
    if T !== Tprod
        T = promote_type(T, Tprod)
        CT = Complex{T}
        J = CT.(J)
        Hc = CT.(Hc)
        Hr = T.(Hr)
    end

    R, Kru, Kiu = setrounding(T, RoundUp) do
        R = (abs.(real(J)) + abs.(imag(J))) * Hr
        Kru = Mru + R
        Kiu = Miu + R
        return R, Kru, Kiu
    end

    Krl, Kil = setrounding(T, RoundDown) do
        Krl = Mrl - R
        Kil = Mil - R
        return Krl, Kil
    end

    return Krl, Kru, Kil, Kiu, T
end

"""
    _cr(Fl::AbstractMatrix{<:Real}, Fu::AbstractMatrix{<:Real}, ::Type{T}) where {T}

Algorithm 5 of Ref. [Miyajima2010](@cite).  Convert entrywise real lower/upper
bounds `Fl ≤ F ≤ Fu` into midpoint and radius matrices `(Fc, Fr)` evaluated in
type `T` using directed rounding.
"""
function _cr(Fl::AbstractMatrix{<:Real}, Fu::AbstractMatrix{<:Real},
        ::Type{T}) where {T <: AbstractFloat}
    Fl = T.(Fl)
    Fu = T.(Fu)
    half = T(0.5)

    Fc = setrounding(T, RoundNearest) do
        (Fu .+ Fl) .* half
    end

    Fr = setrounding(T, RoundUp) do
        (Fu .- Fl) .* half
    end

    return Fc, Fr
end

"""
    _cr(Fl::AbstractMatrix{<:Real}, Fu::AbstractMatrix{<:Real})

Convenience overload returning `(Fc, Fr, T)` where `T` is the promoted working
type.
"""
function _cr(Fl::AbstractMatrix{<:Real}, Fu::AbstractMatrix{<:Real})
    TFl = float(typeof(real(zero(eltype(Fl)))))
    TFu = float(typeof(real(zero(eltype(Fu)))))
    T = promote_type(TFl, TFu)
    Fc, Fr = _cr(Fl, Fu, T)
    return Fc, Fr, T
end

"""
    _iprod(F::AbstractMatrix{<:Real}, Gc::AbstractMatrix{<:Real}, Gr::AbstractMatrix{<:Real})

Algorithm 6 of Ref. [Miyajima2010](@cite).  Multiply a real matrix `F` by a
ball enclosure `(Gc, Gr)` and return rectangular bounds `(Hl, Hu)` and the
working type `T` enclosing `F * (Gc ± Gr)`.
"""
function _iprod(
        F::AbstractMatrix{<:Real}, Gc::AbstractMatrix{<:Real}, Gr::AbstractMatrix{<:Real})
    TF = float(typeof(real(zero(eltype(F)))))
    TGc = float(typeof(real(zero(eltype(Gc)))))
    TGr = float(typeof(real(zero(eltype(Gr)))))
    T = promote_type(TF, TGc, TGr)

    F = T.(F)
    Gc = T.(Gc)
    Gr = T.(Gr)

    R, Hu = setrounding(T, RoundUp) do
        R = abs.(F) * Gr
        Hu = F * Gc + R
        return R, Hu
    end

    Hl = setrounding(T, RoundDown) do
        Hl = F * Gc - R
        return Hl
    end

    return Hl, Hu, T
end

"""
    _ciprod(J::AbstractMatrix{<:Complex}, Hrl, Hru, Hil, Hiu)

Algorithm 7 of Ref. [Miyajima2010](@cite).  Multiply a complex matrix `J` by
rectangular bounds on another complex matrix, provided as lower/upper bounds
for the real and imaginary parts.  The result is returned as rectangular bounds
`(Krl, Kru, Kil, Kiu)` together with the working type `T`.
"""
function _ciprod(J::AbstractMatrix{<:Complex},
        Hrl::AbstractMatrix{<:Real}, Hru::AbstractMatrix{<:Real},
        Hil::AbstractMatrix{<:Real}, Hiu::AbstractMatrix{<:Real})
    TJ = float(typeof(real(zero(eltype(J)))))
    THr = float(typeof(real(zero(eltype(Hrl)))))
    THu = float(typeof(real(zero(eltype(Hru)))))
    THil = float(typeof(real(zero(eltype(Hil)))))
    THiu = float(typeof(real(zero(eltype(Hiu)))))
    T = promote_type(TJ, THr, THu, THil, THiu)

    Jr = T.(real.(J))
    Ji = T.(imag.(J))

    Hrc, Hrr = _cr(Hrl, Hru, T)
    Hic, Hir = _cr(Hil, Hiu, T)

    Mrrl, Mrru, _ = _iprod(Jr, Hrc, Hrr)
    Mirl, Miru, _ = _iprod(Ji, Hrc, Hrr)
    Mril, Mriu, _ = _iprod(Jr, Hic, Hir)
    Miil, Miiu, _ = _iprod(Ji, Hic, Hir)

    Krl, Kil = setrounding(T, RoundDown) do
        Krl = Mrrl - Miiu
        Kil = Mirl + Mril
        return Krl, Kil
    end

    Kru, Kiu = setrounding(T, RoundUp) do
        Kru = Mrru - Miil
        Kiu = Miru + Mriu
        return Kru, Kiu
    end

    return Krl, Kru, Kil, Kiu, T
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
