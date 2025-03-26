module MultitypeBranchingProcessInference

using Reexport

using Random
using Distributions
using LinearAlgebra
using StaticArrays
Reexport.@reexport using MultitypeBranchingProcesses
Reexport.@reexport using ParticleFilter
Reexport.@reexport using KalmanFilters
Reexport.@reexport using MetropolisHastings

import MultitypeBranchingProcesses: init!
import ParticleFilter: gettime, iterate!
import Distributions: mean, mean!, logpdf!

export MTBPParams,
    gettime,
    MTBPParamsSequence,
    setparams!

export MTBPKalmanFilterApproximation,
    ParticleFilterApproximation,
    HybridFilterApproximation,
    MTBPLikelihoodApproximationMethods,
    init!,
    iterate!,
    noswitch,
    RoundedMvNormal,
    rand!,
    logpdf!

include("mtbpparams.jl")
include("particlefilterapproximation.jl")
include("kalmanfilterapproximation.jl")
include("hybridfilter.jl")
include("likelihoodapproximation.jl")

end
