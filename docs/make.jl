using BallArithmetic
using BallArithmetic.CertifScripts
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
    modules = [BallArithmetic, CertifScripts],
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
        "API" => "API.md",
        "Eigenvalues" => "eigenvalues.md",
        "References" => "references.md"
    ])

deploydocs(;
    repo = "github.com/JuliaBallArithmetic/BallArithmetic.jl.git",
    devbranch = "main")
