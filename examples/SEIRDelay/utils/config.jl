include("../../utils/gaussianprocesses.jl")

using Distributions

function setenvironment!(config)
	if "env" in keys(config) && "blas_num_threads" in keys(config["env"])
		LinearAlgebra.BLAS.set_num_threads(config["env"]["blas_num_threads"])
	end
	return
end

function makerng(seed)
	rng = TaskLocalRNG()
	Random.seed!(rng, seed)
	return rng
end

"""
Convert SEIR with delay parameters from mean times to rates.
"""
function convertseirdparamstorates(R_0, T_E, T_I, T_D, E_state_count, I_state_count)
	# rate of symptom onset
	delta = E_state_count / T_E
	# rate of recovery
	lambda = I_state_count / T_I
	# rate of infection
	beta = R_0 / T_I
	# notification rate
	notification_rate = 1.0 / T_D
	return delta, lambda, beta, notification_rate
end

"""
Create a MultitypeBranchingProcess instance for a SEIR model with delayed notifications
"""
function SEIR_delay(
	N, M,
	infection_rate::T,
	exposed_stage_chage_rate::T,
	infectious_stage_chage_rate::T,
	observation_probablity::T,
	notification_rate::T,
	immigration_rates::AbstractArray{T},
	initial_dist::MTBPDiscreteDistribution,
) where {T}
	@assert length(immigration_rates) == N + M "length of immigration rates must match the number exposed and infectious stages N+M"
	# State space has the following interpretation
	# [ E1; ... EN; I1; ... IM;  O; N; IM]
	# Ei - Exposed stage i
	# Ii - Infectious stage i
	# O  - Infectious count
	# N - Observed notification count
	# IM - Immigration
	# IM state remains constant (i.e., poisson immigration events)
	ntypes = N + M + 3
	S = variabletype(initial_dist)

	# define all progeny events
	# Define the changes to the state space
	# Infection
	infection_event = zeros(S, ntypes)
	infection_event[1] = one(S)

	# Progression through latent stages
	stage_progression_events = Array{S, 1}[]
	for state_idx in 1:(N+M-1)
		event = zeros(S, ntypes)
		event[state_idx] = -one(S)
		event[state_idx+1] = one(S)
		push!(stage_progression_events, event)
	end

	recovery_event = zeros(S, ntypes)
	recovery_event[N+M] = -one(S)

	# Event observation occurs simulataneously with EN -> I1, with specified observation probability
	observation_event = copy(stage_progression_events[N])
	observation_event[N+M+1] = one(S)

	# Notification after delay
	notification_event = zeros(S, ntypes)
	notification_event[N+M+1] = -one(S)
	notification_event[N+M+2] = one(S)

	immigration_events = Array{S, 1}[]
	for state_idx in 1:(N+M)
		event = zeros(S, ntypes)
		event[state_idx] = one(S)
		push!(immigration_events, event)
	end

	# define progeny distributions themselves
	progeny_dist_type = MTBPDiscreteDistribution{
		Vector{T}, # CDF parameters
		Vector{Vector{S}}, # Events
		Vector{T}, # first moments
		Matrix{T}, # second moments
	}
	progeny_dists = progeny_dist_type[]

	# unobserved exposed states
	unobserved_exposed_state_cdf = T[1]
	for state_idx in 1:N-1
		# only stage transitions occur while exposed
		unobserved_exposed_state_events = [
			stage_progression_events[state_idx],
		]
		dist =
			MTBPDiscreteDistribution(unobserved_exposed_state_cdf, unobserved_exposed_state_events)
		push!(progeny_dists, dist)
	end

	# observed exposed state
	observed_exposed_state_cdf = T[(1-observation_probablity), 1]
	observed_exposed_state_events = [
		stage_progression_events[N],
		observation_event,
	]
	dist = MTBPDiscreteDistribution(observed_exposed_state_cdf, observed_exposed_state_events)
	push!(progeny_dists, dist)

	# infectious state transitions without recovery
	infectious_state_cdf = T[infection_rate/(infection_rate+infectious_stage_chage_rate), 1]
	for state_idx in N+1:N+M-1
		# stage transitions or infections can occur
		infectious_state_events = [
			infection_event,
			stage_progression_events[state_idx],
		]
		dist = MTBPDiscreteDistribution(infectious_state_cdf, infectious_state_events)
		push!(progeny_dists, dist)
	end

	# infectious state transitions with recovery
	infectious_state_cdf = T[infection_rate/(infection_rate+infectious_stage_chage_rate), 1]
	infectious_state_events = [infection_event, recovery_event]
	dist = MTBPDiscreteDistribution(infectious_state_cdf, infectious_state_events)
	push!(progeny_dists, dist)

	# Observation state
	# observation_state_cdf = T[1]
	observation_state_events = [notification_event]
	dist = MTBPDiscreteDistribution(T[1], observation_state_events)
	push!(progeny_dists, dist)

	# Notification state - persist throughout simulation
	dist = dummy_mtbp_discrete_distribution(ntypes, typeof(infection_event), T)
	push!(progeny_dists, dist)

	# Immigration
	total_immigration_rate = sum(immigration_rates)
	if total_immigration_rate == zero(total_immigration_rate)
		# ensure cdf is proper
		immigration_cdf = collect(
			Iterators.drop(
				range(
					zero(total_immigration_rate),
					one(total_immigration_rate),
					length(immigration_rates) + 1,
				),
				1,
			),
		)
	else
		immigration_pmf = immigration_rates ./ total_immigration_rate
		immigration_cdf = cumsum(immigration_pmf)
		# ensure cdf is proper
		immigration_cdf ./= immigration_cdf[end]
	end
	immigration_progeny_dist = MTBPDiscreteDistribution(immigration_cdf, immigration_events)
	push!(progeny_dists, immigration_progeny_dist)

	rates = zeros(T, ntypes)
	rates[1:N] .= exposed_stage_chage_rate
	rates[N+1:N+M] .= infection_rate + infectious_stage_chage_rate
	rates[N+M+1] = notification_rate
	rates[N+M+3] = total_immigration_rate

	return MultitypeBranchingProcess(ntypes, initial_dist, progeny_dists, rates)
end

"""
SEIR_delay taking in the initial state as a vector
"""
function SEIR_delay(
	N, M,
	infection_rate::T,
	exposed_stage_chage_rate::T,
	infectious_stage_chage_rate::T,
	observation_probablity::T,
	notification_rate::T,
	immigration_rates::AbstractArray{T},
	initial_state = [1; zeros(Int, N + M); 1],
) where {T}
	initial_cdf = T[1]
	initial_dist = MTBPDiscreteDistribution(initial_cdf, [initial_state])
	return SEIR_delay(
		N, M,
		infection_rate,
		exposed_stage_chage_rate,
		infectious_stage_chage_rate,
		observation_probablity,
		notification_rate,
		immigration_rates,
		initial_dist,
	)
end


"""
In-place update of the parameters in a MTBPParams instance
"""
function param_map!(
	mtbpparams,
	E_state_count,
	I_state_count,
	seir_params,
	immigration,
)
	R_0, T_E, T_I, T_D = seir_params
	delta, lambda, beta, notification_rate =
		convertseirdparamstorates(R_0, T_E, T_I, T_D, E_state_count, I_state_count)

	# exposed individuals progress to infectious at rate delta
	exposed_states = 1:E_state_count
	for i in exposed_states
		mtbpparams.rates[i] = delta
	end
	# Note: infection events are either observed or unobserved
	# with a fixed probability. Hence the cdfs of exposed progeny
	# events is fixed and does not need to be updated

	# infectious individuals create infections at rate beta and recover at rate lambda
	infectious_states = (E_state_count+1):(E_state_count+I_state_count)
	for i in infectious_states
		mtbpparams.rates[i] = beta + lambda
	end
	# Count state
	mtbpparams.rates[end-2] = notification_rate # CDF does not change
	# Notification state - death rate always 0
	mtbpparams.rates[end-1] = zero(eltype(mtbpparams.rates))

	# Immigration state
	mtbpparams.rates[end] = sum(immigration)

	p = beta / (beta + lambda)
	one_ = one(eltype(mtbpparams.rates))
	for i in infectious_states
		mtbpparams.cdfs[i][1] = p
		mtbpparams.cdfs[i][2] = one_
	end

	# Immigration rates - if relevant
	if iszero(mtbpparams.rates[end])
		mtbpparams.cdfs[end] .= range(
			zero(eltype(mtbpparams.cdfs[end])),
			one(eltype(mtbpparams.cdfs[end])),
			length(immigration),
		)
	else
		mtbpparams.cdfs[end] .= cumsum(immigration)
		mtbpparams.cdfs[end] ./= mtbpparams.cdfs[end][end]
	end

	return mtbpparams
end

"""
Create a SEIR model with delay and parameter sequence from a given configuration.
"""
function makemodel(config)
	seirconfig = config["model"]["stateprocess"]["params"]
	delta, lambda, beta, notification_rate = convertseirdparamstorates(
		first(seirconfig["R_0"]), first(seirconfig["T_E"]), first(seirconfig["T_I"]),
		first(seirconfig["T_D"]),
		seirconfig["E_state_count"], seirconfig["I_state_count"],
	)

	N = seirconfig["E_state_count"]
	M = seirconfig["I_state_count"]

	seir = SEIR_delay(
		N,
		M,
		beta,
		delta,
		lambda,
		seirconfig["observation_probability"],
		notification_rate,
		seirconfig["immigration_rate"],
		config["model"]["stateprocess"]["initial_state"],
	)

	# Set up observation process
	obs_config = config["model"]["observation"]

	obs_operator = zeros(1, getntypes(seir))
	obs_operator[obs_state_idx(seir)] = 1.0
	obs_model = LinearGaussianObservationModel(
		obs_operator,
		obs_config["mean"],
		reshape(obs_config["cov"], 1, 1),
	)

	model = StateSpaceModel(seir, obs_model)

	# Parameter sequence for time-evolution of parameters
	param_vec = MTBPParams{paramtype(model), Vector{paramtype(model)}}[]
	param_seq = MTBPParamsSequence(param_vec)
	for i in eachindex(seirconfig["R_0"])
		mtbpparams = MTBPParams(seir)
		paramtimestamp = seirconfig["timestamps"][i]
		R_0 = seirconfig["R_0"][i]
		T_E = seirconfig["T_E"][i]
		T_I = seirconfig["T_I"][i]
		T_D = seirconfig["T_D"][i]
		immigration = seirconfig["immigration_rate"]
		param_map!(
			mtbpparams,
			seirconfig["E_state_count"],
			seirconfig["I_state_count"],
			(R_0, T_E, T_I, T_D),
			immigration,
		)
		mtbpparams.time = paramtimestamp
		push!(param_seq.seq, mtbpparams)
	end
	return model, param_seq
end

function pathtodailycases(path, cases_idx)
	cumulative_cases = [[state[cases_idx]] for state in path]
	daily_cases = diff(cumulative_cases)
	cases = [[cumulative_cases[1]]; daily_cases]
	return cases
end

function reset_obs_state_iter_setup!(
	f::HybridFilterApproximation,
	model, dt, observation, iteration, use_prev_iter_params,
)
	return
end

function reset_obs_state_iter_setup!(
	f::MTBPKalmanFilterApproximation,
	model, dt, observation, iteration, use_prev_iter_params,
)
	reset_idx = obs_state_idx(model.stateprocess)
	kf = f.kalmanfilter
	kf.state_estimate[reset_idx] = zero(eltype(kf.state_estimate))
	kf.state_estimate_covariance[:, reset_idx] .= zero(eltype(kf.state_estimate_covariance))
	kf.state_estimate_covariance[reset_idx, :] .= zero(eltype(kf.state_estimate_covariance))
	return
end

function reset_obs_state_iter_setup!(
	f::ParticleFilterApproximation,
	model, dt, observation, iteration, use_prev_iter_params,
)
	reset_idx = obs_state_idx(model.stateprocess)
	return for particle in f.store.store
		particle[reset_idx] = zero(eltype(particle))
	end
end

function makeloglikelihood(observations, config)
	model, param_seq = makemodel(config)
	if config["inference"]["likelihood_approx"]["method"] == "hybrid"
		pf_rng = makerng(config["inference"]["likelihood_approx"]["particle_filter"]["seed"])
		nparticles = config["inference"]["likelihood_approx"]["particle_filter"]["nparticles"]

		switch_rng = makerng(config["inference"]["likelihood_approx"]["switch"]["seed"])
		switch_threshold = config["inference"]["likelihood_approx"]["switch"]["threshold"]

		randomstatesidx = getrandomstateidx(model.stateprocess)

		approx = HybridFilterApproximation(
			model, pf_rng, switch_rng, nparticles, switch_threshold, randomstatesidx,
		)
	elseif config["inference"]["likelihood_approx"]["method"] == "particle_filter"
		pf_rng = makerng(config["inference"]["likelihood_approx"]["particle_filter"]["seed"])
		nparticles = config["inference"]["likelihood_approx"]["particle_filter"]["nparticles"]
		ismultithreadded =
			config["inference"]["likelihood_approx"]["particle_filter"]["multithreaded"]

		approx = ParticleFilterApproximation(model, pf_rng, nparticles, ismultithreadded)

		if "switch" in keys(config["inference"]["likelihood_approx"])
			@warn "Unused \"switch\" params in config with approximation method \"particle_filter\"."
		end
	elseif config["inference"]["likelihood_approx"]["method"] == "kalman_filter"
		approx = MTBPKalmanFilterApproximation(model)
		if "switch" in keys(config["inference"]["likelihood_approx"])
			@warn "Unused \"switch\" params in config with approximation method \"kalman_filter\"."
		end
		if "particle_filter" in keys(config["inference"]["likelihood_approx"])
			@warn "Unused \"particle_filter\" params in config with approximation method \"kalman_filter\"."
		end
	else
		error("Unknown likelihood_approx method specified in config.")
	end

	seirconfig = config["model"]["stateprocess"]["params"]
	function llparam_map!(mtbpparams, param)
		return param_map!(
			mtbpparams,
			seirconfig["E_state_count"],
			seirconfig["I_state_count"],
			param,
			seirconfig["immigration_rate"],
		)
	end

	# Determine which parameters are estimated, and hence in the call to pars
	# Parameters are [R_0, T_E, T_I, T_D]
	param_names = ["R_0", "T_E", "T_I", "T_D"]
	@assert all(x -> x in param_names, config["inference"]["parameters"]) "Unexpected parameter specified in inference config. \
	Expected one or more of [R_0, T_E, T_I, T_D]"

	# true/false indicates whether the parameter is estimated
	# Order is important - used to index curr_params
	is_estimated = map(x -> x in config["inference"]["parameters"], param_names)
	not_is_estimated = .!is_estimated

	# In order [R_0, T_E, T_I, T_D]
	curr_params = zeros(paramtype(model), length(param_names))

	# Fixed params - take the values specified in the config
	# indexed with [time][param_idx]
	timestamps = seirconfig["timestamps"]
	not_estimated_vals = Vector{Vector{Float64}}(undef, length(timestamps))
	for i in eachindex(timestamps)
		not_estimated_vals[i] = map(x -> seirconfig[x][i], param_names[not_is_estimated])
	end


	loglikelihood =
		(pars) -> begin
			# println("Proposed pars: ", pars)
			for i in eachindex(param_seq.seq)
				curr_params[is_estimated] .= pars[i]
				curr_params[not_is_estimated] .= not_estimated_vals[i]
				llparam_map!(param_seq[i], curr_params)
			end

			return logpdf!(model, param_seq, observations, approx, reset_obs_state_iter_setup!)
		end

	return loglikelihood
end

struct RandomWalkGammaInitialDistR0Prior{F}
	initial_dist::Gamma{F}
	randomwalkstddev::F
end

function Distributions.logpdf(
	rw::RandomWalkGammaInitialDistR0Prior{F},
	R0s::AbstractVector,
) where {F}
	ll = logpdf(rw.initial_dist, R0s[1])
	for i in Iterators.drop(eachindex(R0s), 1)
		ll += logpdf(Normal(R0s[i-1], rw.randomwalkstddev), R0s[i])
	end
	return ll
end

function makeconstpriordists(config)
	const_prior_dists = Any[
		Gamma(config["inference"]["prior_parameters"]["T_E"]["shape"],
			config["inference"]["prior_parameters"]["T_E"]["scale"]),
		Gamma(config["inference"]["prior_parameters"]["T_I"]["shape"],
			config["inference"]["prior_parameters"]["T_I"]["scale"]),
	]
	return tuple(const_prior_dists...)
end

function makeprior(config)

	# TODO: Accept a list of parameters and constant distributions!
	# const_prior_dists = makeconstpriordists(config)
	const_prior_dists = tuple([]...)

	# CHYECK THIS!!!! INDEXING MIGHT BE OFF
	if "R_0" in config["inference"]["parameters"]
		if config["inference"]["prior_parameters"]["R_0"]["type"] ==
		   "random_walk_gamma_initial_dist"
			init_dist = Gamma(config["inference"]["prior_parameters"]["R_0"]["shape"],
				config["inference"]["prior_parameters"]["R_0"]["scale"])
			sigma = config["inference"]["prior_parameters"]["R_0"]["sigma"]
			R0prior = RandomWalkGammaInitialDistR0Prior(init_dist, sigma)
			prior_logpdf =
				(params) -> begin
					val = zero(eltype(params))
					for i in eachindex(const_prior_dists)
						val += logpdf(const_prior_dists[i], params[i])
					end
					val += logpdf(R0prior, params[(length(const_prior_dists)+1):end])
					return val
				end
		elseif config["inference"]["prior_parameters"]["R_0"]["type"] == "gaussian_processes"
			if config["inference"]["prior_parameters"]["R_0"]["covariance_function"] ==
			   "exponential"
				cov_fun = GP.ExponentialCovarianceFunction(
					config["inference"]["prior_parameters"]["R_0"]["sigma"]^2,
					config["inference"]["prior_parameters"]["R_0"]["ell"],
				)
				timestamps = Matrix(
					reshape(
						Float64.(config["model"]["stateprocess"]["params"]["timestamps"]),
						1,
						:,
					),
				)
				mu = config["inference"]["prior_parameters"]["R_0"]["mu"]
				R0prior = GP.GaussianProcess(timestamps, mu, cov_fun)
			elseif config["inference"]["prior_parameters"]["R_0"]["covariance_function"] ==
				   "squared_exponential"
				cov_fun = GP.SquaredExponentialCovarianceFunction(
					config["inference"]["prior_parameters"]["R_0"]["sigma"]^2,
					config["inference"]["prior_parameters"]["R_0"]["ell"],
				)
				timestamps = Matrix(
					reshape(
						Float64.(config["model"]["stateprocess"]["params"]["timestamps"]),
						1,
						:,
					),
				)
				mu = config["inference"]["prior_parameters"]["R_0"]["mu"]
				R0prior = GP.GaussianProcess(timestamps, mu, cov_fun)
			else
				error("Unknown covariance function in config")
			end
			if config["inference"]["prior_parameters"]["R_0"]["transform"] == "log"
				cache = zeros(Float64, length(timestamps))
				gpmemcache = GP.gp_logpdf_memcache(R0prior, cache)
				prior_logpdf =
					(params) -> begin
						if any(p -> p <= zero(p), params)
							return -Inf
						end
						val = zero(eltype(params))
						for i in eachindex(const_prior_dists)
							val += logpdf(const_prior_dists[i], params[i])
						end
						for i in Iterators.drop(eachindex(params), length(const_prior_dists))
							cache[i-length(const_prior_dists)] = log(params[i])
						end
						val += GP.logpdf(R0prior, cache, gpmemcache)
						val -= sum(cache)
						return val
					end
			elseif config["inference"]["prior_parameters"]["R_0"]["transform"] == "none"
				cache = zeros(Float64, length(timestamps))
				gpmemcache = GP.gp_logpdf_memcache(R0prior, cache)
				prior_logpdf =
					(params) -> begin
						val = zero(eltype(params))
						for i in eachindex(const_prior_dists)
							val += logpdf(const_prior_dists[i], params[i])
						end
						for i in Iterators.drop(eachindex(params), length(const_prior_dists))
							cache[i-length(const_prior_dists)] = params[i]
						end
						val += GP.logpdf(R0prior, cache, gpmemcache)
						return val
					end
			else
				error(
					"Unkown R_0 prior specification, expected \"log\" or \"none\", got $(config["inference"]["prior_parameters"]["R_0"]["transform"])",
				)
			end
		else
			error("Unknown prior R0 specififcation")
		end
	end
	return prior_logpdf
end

function makeproposal(config)
	proposalconfig = config["inference"]["proposal_parameters"]
	mu = proposalconfig["mean"]
	sigma = reshape(proposalconfig["cov"], length(mu), length(mu))
	return MutableMvNormal(mu, sigma)
end

function makemhconfig(config)
	mh_rng = makerng(config["inference"]["mh_config"]["seed"])
	mh_config = MHConfig(
		config["inference"]["mh_config"]["buffer_size"],
		config["inference"]["mh_config"]["outfilename"],
		config["inference"]["mh_config"]["max_iters"],
		config["inference"]["mh_config"]["nparams"],
		config["inference"]["mh_config"]["max_time_sec"],
		config["inference"]["mh_config"]["init_sample"],
		config["inference"]["mh_config"]["verbose"],
		config["inference"]["mh_config"]["infofilename"],
		config["inference"]["mh_config"]["adaptive"],
		config["inference"]["mh_config"]["nadapt"],
		config["inference"]["mh_config"]["adapt_cov_scale"],
		config["inference"]["mh_config"]["continue"],
	)
	return mh_rng, mh_config
end
