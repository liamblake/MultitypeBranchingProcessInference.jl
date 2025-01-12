module StateSpaceModels

using Reexport
using Random

using LinearAlgebra
Reexport.@reexport using MultitypeBranchingProcesses

import Distributions: logpdf, rand!
import MultitypeBranchingProcesses: getntypes,
    paramtype,
    variabletype,
    statetype,
    getstate,
    setstate!,
    simulate!,
    init!

export StateSpaceModel,
    randinitstate!,
    simulatestate!,
    AbstractObservationModel,
    LinearGaussianObservationModel,
    logpdf,
    IdentityObservationModel

include("observationmodel.jl")
include("models.jl")

end