using BallArithmetic
using BallArithmetic.CertifScripts
using BallArithmetic.NumericalTest
using Documenter
using DocumenterCitations

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style = :numeric
)

DocMeta.setdocmeta!(BallArithmetic, :DocTestSetup, :(using BallArithmetic);
    recursive = true)

makedocs(;
    plugins = [bib],
    modules = [BallArithmetic, CertifScripts, NumericalTest],
    warnonly = [:missing_docs],
    authors = "Luca Ferranti, Isaia Nisoli",
    repo = Documenter.Remotes.GitHub("JuliaBallArithmetic", "BallArithmetic.jl"),
    sitename = "BallArithmetic.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://juliaballarithmetic.github.io/BallArithmetic.jl/",
        edit_link = "main",
        assets = String["assets/citations.css"]
    ),
    pages = [
        "Home" => "index.md",
        "Matrix Decompositions" => "decompositions.md",
        "SVD" => "svd.md",
        "Eigenvalues" => "eigenvalues.md",
        "Pseudospectra" => "pseudospectra.md",
        "Linear Systems" => "linearsystems.md",
        "API" => [
            "Overview" => "API.md",
            "Core Types" => "api/core.md",
            "Linear Systems" => "api/linearsystems.md",
            "Eigenvalues & SVD" => "api/eigenvalues.md",
            "CertifScripts" => "api/certifscripts.md",
            "NumericalTest" => "api/numericaltest.md"
        ],
        "References" => "references.md"
    ])

deploydocs(;
    repo = "github.com/JuliaBallArithmetic/BallArithmetic.jl.git",
    devbranch = "main")
