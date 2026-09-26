using BallArithmetic
using Test
using LinearAlgebra
using Random

# Regression test for the rigor of the per-block coupling radius in the verified block
# diagonalisations.
#
# `_vbd_block_data` charges each cluster its own off-block row, and the enclosure it feeds is
#     spec(A) ⊆ ⋃ᵢ {z : σ_min(Pᵢ − zI) ≤ rᵢ}.
# Historically `rᵢ` was the bare spectral norm of the off-block row strip, `‖Ñ[Cᵢ,:]‖₂ + β₂`.
# That bounds neither of the two quantities the Feingold–Varga argument produces: writing an
# eigenvector in blocks and taking `i` with `‖xᵢ‖` largest,
#     σ_min(Pᵢ − λI) ‖xᵢ‖ ≤ ‖Ñ[Cᵢ,:] x̂‖,   x̂ = x with block `Cᵢ` zeroed,
# and the right-hand side is at most `Σ_{j≠i} ‖Ñ[Cᵢ,Cⱼ]‖₂ ‖xᵢ‖` (row sum) or
# `√(q−1) ‖Ñ[Cᵢ,:]‖₂ ‖xᵢ‖` (strip, since `x̂` has at most `q−1` blocks of norm ≤ ‖xᵢ‖).
# The strip norm without the `√(q−1)` is smaller than both and lets an eigenvalue escape.
#
# The matrix below is one that does escape: 10×10, five 2×2 diagonal blocks, off-blocks damped
# by 0.35, MersenneTwister(4348).  Its real eigenvalue λ ≈ −2.50791 has
# σ_min(Pᵢ − λI) ≈ 1.100039 for the tightest block against a bare strip norm of ≈ 1.091111,
# so λ lies outside every disc.  The row sum (≈ 2.303) and √4 · strip (≈ 2.182) both contain it.

const VBD_Q, VBD_B = 5, 2
vbd_blk(i) = ((i - 1) * VBD_B + 1):(i * VBD_B)

function vbd_escaping_matrix()
    rng = MersenneTwister(4348)
    n = VBD_Q * VBD_B
    M = randn(rng, n, n)
    for i in 1:VBD_Q, j in 1:VBD_Q
        i == j && continue
        M[vbd_blk(i), vbd_blk(j)] .*= 0.35
    end
    return M
end

@testset "VBD per-block coupling: the bare strip norm is not rigorous" begin
    M = vbd_escaping_matrix()
    n = size(M, 1)
    λs = eigvals(M)
    λ = λs[argmin(abs.(λs .- (-2.5079108472)))]
    @test isreal(λ) || abs(imag(λ)) < 1e-10
    @test abs(real(λ) - (-2.5079108472)) < 1e-8

    # the three candidate radii for each block, on the exact matrix (no ball inflation, so
    # β₂ = 0): the quantity under test is the row bound alone
    σ = [minimum(svdvals(M[vbd_blk(i), vbd_blk(i)] - λ * I)) for i in 1:VBD_Q]
    bare = [opnorm(M[vbd_blk(i), setdiff(1:n, vbd_blk(i))]) for i in 1:VBD_Q]
    rowsum = [sum(opnorm(M[vbd_blk(i), vbd_blk(j)]) for j in 1:VBD_Q if j != i)
              for i in 1:VBD_Q]
    strip = sqrt(VBD_Q - 1) .* bare

    # the bare strip norm excludes this eigenvalue from every disc: the enclosure fails
    @test all(σ .> bare)
    # both correct forms contain it in at least one disc
    @test any(σ .<= rowsum)
    @test any(σ .<= strip)
    # and what the code now uses, the minimum of the two, still contains it
    @test any(σ .<= min.(rowsum, strip))
end

@testset "VBD per-block coupling: every eigenvalue is enclosed, random blocks" begin
    rng = MersenneTwister(20260927)
    for trial in 1:300
        n = VBD_Q * VBD_B
        M = randn(rng, n, n)
        for i in 1:VBD_Q, j in 1:VBD_Q
            i == j && continue
            M[vbd_blk(i), vbd_blk(j)] .*= 0.35
        end
        rowsum = [sum(opnorm(M[vbd_blk(i), vbd_blk(j)]) for j in 1:VBD_Q if j != i)
                  for i in 1:VBD_Q]
        strip = sqrt(VBD_Q - 1) .*
                [opnorm(M[vbd_blk(i), setdiff(1:n, vbd_blk(i))]) for i in 1:VBD_Q]
        r = min.(rowsum, strip)
        for λ in eigvals(M)
            σ = [minimum(svdvals(M[vbd_blk(i), vbd_blk(i)] - λ * I)) for i in 1:VBD_Q]
            @test any(σ .<= r .+ 1e-10)
        end
    end
end
