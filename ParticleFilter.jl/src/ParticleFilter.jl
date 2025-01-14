module ParticleFilter
using Reexport

using Random
using Distributions
using LinearAlgebra
Reexport.@reexport using MultitypeBranchingProcesses

include("./StateSpaceModels/StateSpaceModels.jl")
Reexport.@reexport using .StateSpaceModels

import Random: rand, rand!
import Distributions: logpdf, mean, mean!
import MultitypeBranchingProcesses: paramtype, getntypes, init!
import .StateSpaceModels.simulatestate!

export Weights,
    ParticleStore,
    resample!,
    Observation,
    Observations,
    particlefilter!,
    init!,
    initstate!,
    iterate!,
    summarise!,
    mean!,
    mean,
    vcov!,
    vcov,
    gettime,
    getvalue,
    elementtype,
    AbstractThreadInfo, 
    SingleThreadded,
    MultiThreadded,
    ess

include("store.jl")
include("observations.jl")
include("filter.jl")

end
