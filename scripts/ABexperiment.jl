import Pkg;
Pkg.activate("../");

using BallArithmetic

import Pkg;
Pkg.add("RigorousInvariantMeasures");
using RigorousInvariantMeasures

@time begin
    B = Fourier1D(128)

    T(x) = 3.3 * x * (1 - x)

    NK = RigorousInvariantMeasures.GaussianNoise(B, 0.5)

    P = assemble(B, T)

    import IntervalArithmetic
    midI = IntervalArithmetic.mid
    Pfloat = midI.(real.(P)) + im * midI.(imag.(P))

    Q = NK.NK * Pfloat

    enc = BallArithmetic.compute_enclosure(BallMatrix(Q), 0.5, 0.9, 10^-10)
end

Pkg.add("JLD")
using JLD

save("enc33.jld", "Q", Q, "enc", enc)
