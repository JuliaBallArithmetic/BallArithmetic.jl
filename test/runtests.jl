using BallArithmetic
using Test

@testset "BallArithmetic.jl" begin
    # Core types
    include("test_ball/test_ball.jl")
    include("test_types/test_constructors.jl")
    include("test_types/test_algebra.jl")
    include("test_types/test_MMul.jl")
    include("test_types/test_convert_promote.jl")
    include("test_types/test_promotion.jl")
    include("test_types/test_mmul5.jl")
    include("test_types/test_vector.jl")
    include("test_types/test_matrix.jl")
    include("test_types/test_array.jl")
    include("test_types/test_vector_operations.jl")

    # Rounding and BigFloat
    include("test_rounding/test_bigfloat_rounding.jl")
    include("test_rounding/test_ball_bigfloat.jl")

    # Matrix classifiers
    include("test_matrix_classifier/test_matrix_classifier.jl")

    # Eigenvalues
    include("test_eigen/test_eigen.jl")
    include("test_eigen/test_miyajima_new.jl")
    include("test_eigenvalues/test_verified_gev.jl")
    include("test_eigenvalues/test_gev_coherence.jl")
    include("test_eigenvalues/test_riesz_projections.jl")
    include("test_eigenvalues/test_iterative_schur_refinement.jl")

    # Rump methods
    include("test_rump_methods/test_rump_2022a.jl")
    include("test_rump_methods/test_rump_lange_2023.jl")
    include("test_rump_methods/test_rump_oishi_2024.jl")

    # SVD
    include("test_svd/test_svd.jl")
    include("test_svd/test_miyajima_svd_bounds.jl")
    include("test_svd/test_adaptive_ogita_svd.jl")
    include("test_svd/test_subepsilon_certification.jl")
    include("test_svd/test_precision_cascade_svd.jl")
    include("test_svd/test_precision_cascade_core.jl")
    include("test_svd/test_gla_svd.jl")

    # Norm bounds
    include("test_norm_bounds/test_norm_bounds.jl")
    include("test_norm_bounds/test_oishi.jl")
    include("test_norm_bounds/test_oishi_triangular.jl")
    include("test_norm_bounds/test_oishi_2023_schur.jl")

    # Pseudospectra
    include("test_pseudospectra/test_pseudospectra.jl")
    include("test_pseudospectra/test_sylvester_resolvent.jl")

    # Linear solvers
    include("test_linear_solvers/test_solvers.jl")
    include("test_linear_solvers/test_inflation.jl")
    include("test_linear_solvers/test_backward_substitution.jl")
    include("test_linear_system/test_verified_hmatrix.jl")
    include("test_linear_system/test_krawczyk.jl")
    include("test_linear_system/test_shaving.jl")
    include("test_horacek_methods.jl")

    # Decompositions
    include("test_decompositions/test_iterative_refinement.jl")
    include("test_decompositions/test_iterative_refinement_ext.jl")
    include("test_decompositions/test_verified_decompositions.jl")
    include("test_decompositions/test_verified_takagi.jl")
    include("test_decompositions/test_rigorous_residual.jl")

    # Certification
    include("test_certification/test_certifscripts.jl")
    include("test_numerical_test/test_numerical_test.jl")

    # Extensions
    include("test_interval_arithmetic_ext/test_interval_arithmetic_ext.jl")
    include("test_arbnumerics_ext/test_arbnumerics_ext.jl")
    include("test_fft_ext/test_fft.jl")
    include("test_doublefloats_ext.jl")
end
