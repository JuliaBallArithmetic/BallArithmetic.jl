# Verified error bounds for matrix decompositions, after
#
#   S. M. Rump and T. Ogita, Verified Error Bounds for Matrix Decompositions,
#   SIAM J. Matrix Anal. Appl. 45(4) (2024), 2155-2183, doi 10.1137/24M165096X.
#
# Each routine is named for the algorithm or the section of the paper it implements.

"""
    _rumpogita2024_alg3_2(E::BallMatrix) -> (; LE, UE, LinvE, UinvE) or nothing

Algorithm 3.2 (`LU_E`) of Rump and Ogita (2024): inclusions of the factors of the LU
decomposition, without pivoting, of a perturbed identity matrix, and of their inverses.

For `E` of size `m × n`, `p = min(m, n)` and every `Ẽ` of the ball `E`, the matrix `I_{m,n} + Ẽ`
has a unique decomposition `LU` with `L` of size `m × p` unit lower triangular (trapezoidal) and
`U` of size `p × n` upper triangular (trapezoidal), and

    L ∈ I_{m,p} + LE,      U ∈ I_{p,n} + UE,

`LE` strictly lower and `UE` upper triangular. `LinvE` is returned when `L` is square
(`m ≤ n`), with `L⁻¹ ∈ I + LinvE`, and `UinvE` when `U` is square (`m ≥ n`), with
`U⁻¹ ∈ I + UinvE`; otherwise the field is `nothing`. The cost is `O(mn)` operations.

The bounds are (3.1)-(3.7) of the paper. With `Δ_L := sum(|E^{[ℓ]}|, 2) max(|E|) / (1 − ‖E‖_∞)`,
strictly lower part, `|L − I − E^{[ℓ]}| ≤ Δ_L` (3.1); with `G_L = |E^{[ℓ]}| + Δ_L`,
`|L⁻¹ − I + E^{[ℓ]}| ≤ Δ_L + sum(G_L, 2) max(G_L)/(1 − ‖G_L‖_∞)` (3.4); with `B` equal to `|E|`
with its strictly lower part replaced by `Δ_L`, `|U − I − E^{[u]}| ≤ Δ_U := sum(G_L, 2) max(B) /
(1 − ‖G_L‖_∞)`, upper part (3.5); and with `G_U = |E^{[u]}| + Δ_U`,
`|U⁻¹ − I + E^{[u]}| ≤ Δ_U + sum(G_U, 2) max(G_U)/(1 − ‖G_U‖_∞)` (3.7). Here `sum(·, 2)` is the
column of row sums, `max(·)` the row of column maxima, and the products are outer products.

Returns `nothing` when one of the three norms `‖E‖_∞`, `‖G_L‖_∞`, `‖G_U‖_∞` is not below one,
which the paper's listing assumes. Every bound is rounded up.

# Reference

S. M. Rump and T. Ogita, *Verified Error Bounds for Matrix Decompositions*, SIAM J. Matrix Anal.
Appl. **45**(4) (2024), 2155-2183, doi 10.1137/24M165096X, Section 3.1 and Algorithm 3.2.
"""
function _rumpogita2024_alg3_2(E::BallMatrix{T}) where {T}
    m, n = size(E)
    p = min(m, n)
    Em, magE = mid(E), upper_abs(E)
    CT = eltype(Em)
    rowsum(M) = T[sum_up(view(M, i, :)) for i in axes(M, 1)]
    colmax(M) = T[maximum(view(M, :, j); init = zero(T)) for j in axes(M, 2)]
    # sum(M, 2) * max(N) / (1 − ν), keeping the entries (i, j) for which keep(i, j)
    function outer(σ, μ, ν, keep)
        d = sub_down(one(T), ν)
        return T[keep(i, j) ? div_up(mul_up(σ[i], μ[j]), d) : zero(T) for i in 1:m, j in 1:n]
    end
    lower(i, j) = i > j
    upper(i, j) = i <= j
    part(M, keep) = [keep(i, j) ? M[i, j] : zero(eltype(M)) for i in 1:m, j in 1:n]

    νE = maximum(rowsum(magE); init = zero(T))
    νE < 1 || return nothing
    # lines 4-6: L
    DeltaL = outer(rowsum(part(magE, lower)), colmax(magE), νE, lower)
    LEm = part(Em, lower)
    # line 7: GL = mag(LE)
    GL = add_up.(part(magE, lower), DeltaL)
    νL = maximum(rowsum(GL); init = zero(T))
    νL < 1 || return nothing
    # lines 8-10: L⁻¹
    deltaL = add_up.(DeltaL, outer(rowsum(GL), colmax(GL), νL, lower))
    # lines 11-14: U
    B = add_up.(part(magE, upper), DeltaL)
    DeltaU = outer(rowsum(GL), colmax(B), νL, upper)
    UEm = part(Em, upper)
    # line 15: GU = mag(UE)
    GU = add_up.(part(magE, upper), DeltaU)
    νU = maximum(rowsum(GU); init = zero(T))
    νU < 1 || return nothing
    # lines 16-18: U⁻¹
    deltaU = add_up.(DeltaU, outer(rowsum(GU), colmax(GU), νU, upper))

    # the radii of E itself belong to the inclusions: LE = tril(E, −1) + midrad(0, DeltaL)
    Er = rad(E)
    ball(M, R, rows, cols) = BallMatrix(Matrix{CT}(M[rows, cols]), R[rows, cols])
    LE = ball(LEm, add_up.(part(Er, lower), DeltaL), 1:m, 1:p)
    UE = ball(UEm, add_up.(part(Er, upper), DeltaU), 1:p, 1:n)
    LinvE = m <= n ? ball(-LEm, add_up.(part(Er, lower), deltaL), 1:m, 1:m) : nothing
    UinvE = m >= n ? ball(-UEm, add_up.(part(Er, upper), deltaU), 1:n, 1:n) : nothing
    return (; LE, UE, LinvE, UinvE)
end
