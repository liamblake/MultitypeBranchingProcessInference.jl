using YAML
using Distributions
using Random 
using MCMCChains
using LinearAlgebra
using DelimitedFiles

using MultitypeBranchingProcessInference

include("./utils/config.jl")
include("./utils/io.jl")

function main(argv)
    argc = length(argv)
    if argc != 1
        error("inference.jl program expects 1 argument \
               \n    - config file name.")
    end

    config = YAML.load_file(joinpath(pwd(), argv[1]))

    setenvironment!(config)
    
    model, params_seq = makemodel(config)
    
    # get data
    path, t = readparticles(joinpath(pwd(), config["simulation"]["outfilename"]))
    # only need daily cases
    cases = pathtodailycases(path, obs_state_idx(model.stateprocess))
    observations = Observations(t, cases)
    
    loglikelihood = makeloglikelihood(model, params_seq, observations, config)

    # prior logpdf function 
    prior_logpdf = makeprior(config)

    proposal_distribuion = makeproposal(config)

    mh_rng, mh_config = makemhconfig(config)
    
    # run mcmc
    @time nsamples = MetropolisHastings.metropolis_hastings(
        mh_rng, loglikelihood, prior_logpdf, proposal_distribuion, mh_config
    )
    println()
    return 
end

main(ARGS)
