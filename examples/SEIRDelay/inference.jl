using YAML
using Random
using LinearAlgebra
using MultitypeBranchingProcessInference
using Distributions

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

	model, _ = makemodel(config)

	if config["inference"]["data"] == "simulated"
		# If specified, use simulated data as observations
		path, t = readparticles(joinpath(pwd(), config["simulation"]["outfilename"]))
		cases = pathtodailycases(path, obs_state_idx(model.stateprocess))
		observations = Observations(t, cases)
	else
		# Otherwise, assume "filename" and "first_observation_time" keys
		raw_observations =
			read_observations(joinpath(pwd(), config["inference"]["data"]["filename"]))
		t =
			config["inference"]["data"]["first_observation_time"] .+
			(0:(length(raw_observations)-1))
		observations = Observations(t, raw_observations)
	end

	loglikelihood = makeloglikelihood(observations, config)

	prior_logpdf = makeprior(config)

	proposal_distribuion = makeproposal(config)

	mh_rng, mh_config = makemhconfig(config)

	@time nsamples = MetropolisHastings.metropolis_hastings(
		mh_rng, loglikelihood, prior_logpdf, proposal_distribuion, mh_config,
	)
	println()
	return
end

main(ARGS)
