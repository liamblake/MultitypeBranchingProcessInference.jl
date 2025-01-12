import Pkg
Pkg.develop([
    (; path=joinpath(@__DIR__, "..", "MultitypeBranchingProcesses.jl")),
    (; path=joinpath(@__DIR__, "..", "ParticleFilter.jl")),
    (; path=joinpath(@__DIR__, "..", "KalmanFilters.jl")),
    (; path=joinpath(@__DIR__, "..", "MetropolisHastings.jl")),
])