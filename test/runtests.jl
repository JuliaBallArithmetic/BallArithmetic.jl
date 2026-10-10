using BallArithmetic
using Test

@testset "BallArithmetic.jl" begin
    # Core types
    include("test_types/test_ball.jl")
    include("test_types/test_constructors.jl")
    include("test_types/test_algebra.jl")
    include("test_types/test_MMul.jl")
    include("test_types/test_convert_promote.jl")
    include("test_types/test_promotion.jl")
    include("test_types/test_mmul5.jl")
    include("test_types/test_ogita_rump_oishi.jl")
    include("test_types/test_rigour_foundations.jl")
    include("test_types/test_vector.jl")
    include("test_types/test_matrix.jl")
    include("test_types/test_mixed_products.jl")
    include("test_types/test_array.jl")
    include("test_types/test_structured_radius.jl")
    include("test_types/test_indexing_and_validation.jl")
    include("test_types/test_vector_operations.jl")

    # Rounding and BigFloat
    include("test_rounding/test_bigfloat_rounding.jl")
    include("test_rounding/test_ball_bigfloat.jl")
    include("test_rounding/test_scalar_bounds.jl")

    # Matrix classifiers
    include("test_matrix_classifiers/test_matrix_classifier.jl")

    # Eigenvalues
    include("test_eigenvalues/test_eigen.jl")
    include("test_error_free_transformations.jl")
    include("test_eigenvalues/test_rump_verifyeigall.jl")
    include("test_eigenvalues/test_rigour_verifyeigall.jl")
    include("test_eigenvalues/test_verifyeigall_count.jl")
    include("test_eigenvalues/test_miyajima_2014a.jl")
    include("test_eigenvalues/test_miyajima_new.jl")
    include("test_eigenvalues/test_verified_gev.jl")
    include("test_eigenvalues/test_miyajima_gev_enclosure.jl")
    include("test_eigenvalues/test_gev_coherence.jl")
    include("test_eigenvalues/test_riesz_projections.jl")
    include("test_eigenvalues/test_iterative_schur_refinement.jl")
    include("test_eigenvalues/test_rump_lange_2023.jl")
    include("test_eigenvalues/test_newton_kantorovich_eigenpair.jl")
    include("test_eigenvalues/test_ordschur_ball.jl")
    include("test_eigenvalues/test_spectral_projector_enclosure.jl")

    # SVD
    include("test_decompositions/test_svd/test_svd.jl")
    include("test_decompositions/test_svd/test_miyajima_svd_bounds.jl")
    include("test_decompositions/test_svd/test_svd_theorems.jl")
    include("test_decompositions/test_svd/test_adaptive_ogita_svd.jl")
    include("test_decompositions/test_svd/test_subepsilon_certification.jl")
    include("test_decompositions/test_svd/test_precision_cascade_svd.jl")
    include("test_decompositions/test_svd/test_precision_cascade_core.jl")
    include("test_decompositions/test_svd/test_gla_svd.jl")
    include("test_decompositions/test_svd/test_miyajima_2014a_schurnewton.jl")
    include("test_decompositions/test_svd/test_vbd_remainder_norm.jl")
    include("test_decompositions/test_svd/test_vbd_block_coupling.jl")
    include("test_decompositions/test_svd/test_vbd_block_merge.jl")
    include("test_decompositions/test_svd/test_rigorous_svd_gpu.jl")

    # Norm bounds
    include("test_norm_bounds/test_norm_bounds.jl")
    include("test_norm_bounds/test_abs_norm_bounds.jl")
    include("test_norm_bounds/test_oishi.jl")
    include("test_norm_bounds/test_oishi_triangular.jl")
    include("test_norm_bounds/test_oishi_2023_schur.jl")
    include("test_norm_bounds/test_rump_oishi_2024.jl")

    # Pseudospectra
    include("test_pseudospectra/test_pseudospectra.jl")
    include("test_pseudospectra/test_block_resolvent_floor.jl")
    include("test_pseudospectra/test_sylvester_resolvent.jl")
    include("test_pseudospectra/test_gram_transform.jl")

    # Polynomials
    include("test_polynomials/test_poly_range.jl")

    # Linear system
    include("test_linear_system/test_inflation.jl")
    include("test_linear_system/test_backward_substitution.jl")
    include("test_linear_system/test_verified_hmatrix.jl")
    include("test_linear_system/test_krawczyk.jl")
    include("test_linear_system/test_shaving.jl")
    include("test_linear_system/test_horacek_methods.jl")
    include("test_linear_system/test_sylvester_schur.jl")

    # Decompositions
    include("test_decompositions/test_iterative_refinement.jl")
    include("test_decompositions/test_iterative_refinement_ext.jl")
    include("test_decompositions/test_verified_decompositions.jl")
    include("test_decompositions/test_verified_takagi.jl")
    include("test_decompositions/test_rigorous_residual.jl")

    # Certification
    include("test_certification/test_certifscripts.jl")
    include("test_certification/test_rigour_certifscripts.jl")
    include("test_numerical_test/test_numerical_test.jl")

    # Extensions
    include("test_interval_arithmetic_ext/test_interval_arithmetic_ext.jl")
    include("test_arbnumerics_ext/test_arbnumerics_ext.jl")
    include("test_fft_ext/test_fft.jl")
    include("test_doublefloats_ext/test_doublefloats_ext.jl")
end
