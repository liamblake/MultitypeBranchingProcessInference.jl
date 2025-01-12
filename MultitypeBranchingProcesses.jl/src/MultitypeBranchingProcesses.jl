module MultitypeBranchingProcesses

using Random
using LinearAlgebra
using ExponentialUtilities

import Distributions: mean, mean!
import Random: rand, rand!

export MTBPDiscreteDistribution,
    dummy_mtbp_discrete_distribution,
    variabletype,
    paramtype,
    statetype,
    getntypes,
    rand,
    rand!,
    ProgenyDistribution,
    InitialStateDistribution,
    setcdf!

export MTBPMomentsOperator,
    mean!,
    variance_covariance!,
    mean,
    variance_covariance,
    moments!

export MultitypeBranchingProcess,
    getstate,
    setstate!,
    setrates!,
    setprogenycdf!,
    characteristicmatrix!,
    getcharacteristicmatrix,
    getmeanoperator,
    init!,
    simulate!

export SEIR, 
    PoissonProcess,
    obs_state_idx,
    immigration_state_idx,
    getconststatecount,
    getrandomstatecount,
    getrandomstateidx

include("typedistributions.jl")
include("moments.jl")
include("multitypebranchingprocess.jl")
include("simulate.jl")
include("commonmtbpmodels.jl")

end
