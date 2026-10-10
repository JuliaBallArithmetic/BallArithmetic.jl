using BallArithmetic
using LinearAlgebra
using Random
using Test

# The projector and the coefficient are compared with the spectral projector of A computed at
# 512 bits: from a Schur form at that precision, the projector of a triangular matrix for a set of
# diagonal positions being obtained by reordering them first and solving T₁₁Y − YT₂₂ = T₁₂.
@testset "Spectral projector and coefficient: enclosure of the projector of A" begin
    Random.seed!(20261011)
    CS = BallArithmetic.CertifScripts

    function exact_projector(A, inside)            # `inside(λ)` selects the eigenvalues
        setprecision(BigFloat, 512) do
            F = schur(Complex{BigFloat}.(A))
            select = [inside(ComplexF64(λ)) for λ in diag(F.T)]
            k = count(select)
            Z, Tm, _ = ordschur_bigfloat(F.T, F.Z, select)
            n = size(Tm, 1)
            T11, T12, T22 = Tm[1:k, 1:k], Tm[1:k, (k + 1):n], Tm[(k + 1):n, (k + 1):n]
            K = kron(Matrix{Complex{BigFloat}}(I, n - k, n - k), T11) -
                kron(transpose(T22), Matrix{Complex{BigFloat}}(I, k, k))
            Y = reshape(K \ vec(T12), k, n - k)
            PT = [Matrix{Complex{BigFloat}}(I, k, k) Y; zeros(Complex{BigFloat}, n - k, n)]
            Z * PT * Z'
        end
    end
    encloses(B, X) = setprecision(BigFloat, 512) do
        all(abs.(X .- Complex{BigFloat}.(mid(B))) .<= BigFloat.(rad(B)))
    end

    n = 6
    for trial in 1:3
        Ac = Matrix(Diagonal(ComplexF64.([3.0, 3.3, -0.5, 1im, -1im, 0.2 + 0.3im]))) +
             0.15 * randn(ComplexF64, n, n)
        A = BallMatrix(Ac)
        F = schur(Ac)
        centre, radius = 3.15 + 0im, 1.2
        inside(λ) = abs(λ - centre) < radius
        idx = findall(inside, diag(F.T))
        @test length(idx) == 2
        cert = CS.run_certification(Ac, CS.CertificationCircle(centre, radius; samples = 32);
            log_io = devnull)
        P_A = exact_projector(Ac, inside)

        r = compute_spectral_projector_schur(A, idx; resolvent_bound = cert.resolvent_original,
            contour_radius = radius)
        @test isfinite(r.projector_error_bound) && r.projector_error_bound < 1e-9
        @test encloses(r.projector, P_A)
        @test r.orth_defect < 1e-12 && r.fact_defect < 1e-11
        @test istriu(r.schur_form) && r.cluster_indices == 1:2
        @test verify_spectral_projector_properties(r, A; tol = 1e-8, check_commutation = true)

        # without the resolvent bound the result says so
        r0 = compute_spectral_projector_schur(A, idx)
        @test r0.projector_error_bound == Inf
        @test all(rad(r0.projector) .<= rad(r.projector))
        @test_throws ArgumentError compute_spectral_projector_schur(A, idx; resolvent_bound = 1.0)

        # the coefficient: Q₁* P_A v
        v = randn(ComplexF64, n)
        c = compute_spectral_coefficient(A, v, idx; resolvent_bound = cert.resolvent_original,
            contour_radius = radius)
        exact_c = setprecision(BigFloat, 512) do
            Complex{BigFloat}.(c.schur_basis[:, 1:2])' * (P_A * Complex{BigFloat}.(v))
        end
        @test isfinite(c.coefficient_error_bound)
        @test encloses(c.coefficients, exact_c)
        @test compute_spectral_coefficient(A, v, idx).coefficient_error_bound == Inf
        # the projector applied to v agrees with Q₁ c
        Pv = project_vector_spectral(v, r)
        @test encloses(Pv, P_A * Complex{BigFloat}.(v))
    end

    @testset "a ball of matrices" begin
        Ac = Matrix(Diagonal(ComplexF64.([3.0, -0.5, 1im, 0.4]))) + 0.1 * randn(ComplexF64, 4, 4)
        ρ = 1e-7
        A = BallMatrix(Ac, fill(ρ, 4, 4))
        centre, radius = 3.0 + 0im, 1.2
        inside(λ) = abs(λ - centre) < radius
        cert = CS.run_certification(A, CS.CertificationCircle(centre, radius; samples = 32);
            log_io = devnull)
        idx = findall(inside, diag(schur(Ac).T))
        r = compute_spectral_projector_schur(A, idx; resolvent_bound = cert.resolvent_original,
            contour_radius = radius)
        @test isfinite(r.projector_error_bound)
        for _ in 1:20
            Am = Ac + ρ .* rand((-1.0, 1.0), 4, 4)
            @test encloses(r.projector, exact_projector(Am, inside))
        end
    end

    @testset "Hermitian" begin
        H = [4.0 1.0 0.2; 1.0 3.0 0.1; 0.2 0.1 -1.0]
        A = BallMatrix(H)
        λ = eigvals(Hermitian(H))                      # increasing: the cluster is the last two
        centre, radius = (λ[2] + λ[3]) / 2 + 0im, (λ[3] - λ[2]) / 2 + 1.0
        cert = CS.run_certification(complex(H), CS.CertificationCircle(centre, radius; samples = 32);
            log_io = devnull)
        r = compute_spectral_projector_hermitian(A, 2:3; resolvent_bound = cert.resolvent_original,
            contour_radius = radius)
        @test encloses(r.projector, exact_projector(H, z -> abs(z - centre) < radius))
        @test compute_spectral_projector_hermitian(A, 2:3).projector_error_bound == Inf
    end

    @testset "a real matrix with complex eigenvalues is refused in real form" begin
        A = BallMatrix([0.0 -1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 3.0])
        @test_throws ArgumentError compute_spectral_projector_schur(A, 1:1)
    end
end
