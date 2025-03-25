include("../../utils/gaussianprocesses.jl")

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

function convertseirparamstorates(R_0, T_E, T_I, E_state_count, I_state_count)
    # rate of symptom onset
    delta = E_state_count/T_E
    # rate of recovery
    lambda = I_state_count/T_I
    # rate of infection
    beta = R_0/T_I
    return delta, lambda, beta
end

function param_map!(
    mtbpparams, 
    E_state_count, 
    I_state_count, 
    seir_params, # R_0, T_E, T_I 
    immigration, 
)
    R_0, T_E, T_I = seir_params
    delta, lambda, beta = convertseirparamstorates(R_0, T_E, T_I, E_state_count, I_state_count)

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
        mtbpparams.rates[i] = beta+lambda
    end
    mtbpparams.rates[end-1] = zero(eltype(mtbpparams.rates))
    mtbpparams.rates[end] = sum(immigration)

    p = beta/(beta+lambda)
    one_ = one(eltype(mtbpparams.rates))
    for i in infectious_states
        mtbpparams.cdfs[i][1] = p
        mtbpparams.cdfs[i][2] = one_
    end

    if iszero(mtbpparams.rates[end])
        mtbpparams.cdfs[end] .= range(zero(eltype(mtbpparams.cdfs[end])), one(eltype(mtbpparams.cdfs[end])), length(immigration))
    else
        mtbpparams.cdfs[end] .= cumsum(immigration)
        mtbpparams.cdfs[end] ./= mtbpparams.cdfs[end][end]
    end
    return mtbpparams
end

function makemodel(config)
    seirconfig = config["model"]["stateprocess"]["params"]
    delta, lambda, beta = convertseirparamstorates(
        first(seirconfig["R_0"]), first(seirconfig["T_E"]), first(seirconfig["T_I"]),
        seirconfig["E_state_count"], seirconfig["I_state_count"],
    )
    seir = SEIR(
        seirconfig["E_state_count"], 
        seirconfig["I_state_count"],
        beta, 
        delta, 
        lambda,
        seirconfig["observation_probability"], 
        seirconfig["immigration_rate"],
        config["model"]["stateprocess"]["initial_state"],
    )

    obs_config = config["model"]["observation"]
    obs_operator = zeros(1, getntypes(seir))
    obs_operator[obs_state_idx(seir)] = 1.0
    obs_model = LinearGaussianObservationModel(
        obs_operator, obs_config["mean"], 
        reshape(obs_config["cov"], 1, 1)
    )

    model = StateSpaceModel(seir, obs_model)

    param_vec = MTBPParams{paramtype(model), Vector{paramtype(model)}}[]
    param_seq = MTBPParamsSequence(param_vec)
    if seirconfig["is_time_homogeneous"]
        mtbpparams = MTBPParams(seir)
        R_0 = seirconfig["R_0"]
        T_E = seirconfig["T_E"]
        T_I = seirconfig["T_I"]
        immigration = seirconfig["immigration_rate"]
        param_map!(
            mtbpparams, 
            seirconfig["E_state_count"], 
            seirconfig["I_state_count"], 
            (R_0, T_E, T_I), 
            immigration, 
        )
        push!(param_seq.seq, mtbpparams)
    else
        for i in eachindex(seirconfig["R_0"])
            mtbpparams = MTBPParams(seir)
            paramtimestamp = seirconfig["timestamps"][i]
            R_0 = seirconfig["R_0"][i]
            T_E = seirconfig["T_E"][i]
            T_I = seirconfig["T_I"][i]
            immigration = seirconfig["immigration_rate"]
            param_map!(
                mtbpparams, 
                seirconfig["E_state_count"], 
                seirconfig["I_state_count"], 
                (R_0, T_E, T_I),
                immigration, 
            )
            mtbpparams.time = paramtimestamp
            push!(param_seq.seq, mtbpparams)
        end
    end
    return model, param_seq
end

function read_observations(filename)
    observations = open(filename, "r") do io
        nlines = countlines(io)
        seekstart(io)

        lineno = 1
        # skip header line
        readline(io)
        
        observations = Vector{Float64}[]
        while !eof(io)
            lineno += 1
            observation_string = readline(io)
            obs = [parse(Float64, observation_string)]
            push!(observations, obs)
        end
        if lineno != nlines
            error("Bad observations file")
        end
        observations
    end
    return observations
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
            model, pf_rng, switch_rng, nparticles, switch_threshold, randomstatesidx
        )
    elseif config["inference"]["likelihood_approx"]["method"] == "particle_filter"
        pf_rng = makerng(config["inference"]["likelihood_approx"]["particle_filter"]["seed"])
        nparticles = config["inference"]["likelihood_approx"]["particle_filter"]["nparticles"]
        ismultithreadded = config["inference"]["likelihood_approx"]["particle_filter"]["multithreadding"]

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

    curr_params = zeros(paramtype(model), 3)
    nparam_per_stage = 3

    loglikelihood = (pars) -> begin # function loglikelihood(pars)
        # pre-itervention
        curr_params[2:nparam_per_stage] .= pars[1:(nparam_per_stage-1)]
        for i in eachindex(param_seq.seq)
            curr_params[1] = pars[i-1 + nparam_per_stage]
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

function Distributions.logpdf(rw::RandomWalkGammaInitialDistR0Prior{F}, R0s::AbstractVector) where F
    ll = logpdf(rw.initial_dist, R0s[1])
    for i in Iterators.drop(eachindex(R0s),1)
        ll += logpdf(Normal(R0s[i-1], rw.randomwalkstddev), R0s[i])
    end
    return ll
end

function makeconstpriordists(config)
    const_prior_dists = Any[
        Gamma(config["inference"]["prior_parameters"]["T_E"]["shape"], 
              config["inference"]["prior_parameters"]["T_E"]["scale"]),
        Gamma(config["inference"]["prior_parameters"]["T_I"]["shape"], 
              config["inference"]["prior_parameters"]["T_I"]["scale"])
    ]
    return tuple(const_prior_dists...)
end

function makeprior(config)
    const_prior_dists = makeconstpriordists(config)

    if config["inference"]["prior_parameters"]["R_0"]["type"]=="random_walk_gamma_initial_dist"
        init_dist = Gamma(config["inference"]["prior_parameters"]["R_0"]["shape"], 
                          config["inference"]["prior_parameters"]["R_0"]["scale"])
        sigma = config["inference"]["prior_parameters"]["R_0"]["sigma"]
        R0prior = RandomWalkGammaInitialDistR0Prior(init_dist, sigma)
        prior_logpdf = (params) -> begin
            val = zero(eltype(params))
            for i in eachindex(const_prior_dists)
                val += logpdf(const_prior_dists[i], params[i])
            end
            val += logpdf(R0prior, params[(length(const_prior_dists)+1):end])
            return val
        end
    elseif config["inference"]["prior_parameters"]["R_0"]["type"]=="gaussian_processes"
        if config["inference"]["prior_parameters"]["R_0"]["covariance_function"]=="exponential"
            cov_fun = GP.ExponentialCovarianceFunction(
                config["inference"]["prior_parameters"]["R_0"]["sigma"]^2,
                config["inference"]["prior_parameters"]["R_0"]["ell"]
            )
            timestamps = Matrix(reshape(Float64.(config["model"]["stateprocess"]["params"]["timestamps"]), 1, :))
            mu = config["inference"]["prior_parameters"]["R_0"]["mu"]
            R0prior = GP.GaussianProcess(timestamps, mu, cov_fun)
        elseif config["inference"]["prior_parameters"]["R_0"]["covariance_function"]=="squared_exponential"
            cov_fun = GP.SquaredExponentialCovarianceFunction(
                config["inference"]["prior_parameters"]["R_0"]["sigma"]^2,
                config["inference"]["prior_parameters"]["R_0"]["ell"]
            )
            timestamps = Matrix(reshape(Float64.(config["model"]["stateprocess"]["params"]["timestamps"]), 1, :))
            mu = config["inference"]["prior_parameters"]["R_0"]["mu"]
            R0prior = GP.GaussianProcess(timestamps, mu, cov_fun)
        else 
            error("Unknown covariance function in config")
        end
        if config["inference"]["prior_parameters"]["R_0"]["transform"]=="log"
            cache = zeros(Float64, length(timestamps))
            gpmemcache = GP.gp_logpdf_memcache(R0prior, cache)
            prior_logpdf = (params) -> begin
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
        elseif config["inference"]["prior_parameters"]["R_0"]["transform"]=="none"
            cache = zeros(Float64, length(timestamps))
            gpmemcache = GP.gp_logpdf_memcache(R0prior, cache)
            prior_logpdf = (params) -> begin
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
            error("Unkown R_0 prior specification, expected \"log\" or \"none\", got $(config["inference"]["prior_parameters"]["R_0"]["transform"])")
        end
    else
        error("Unknown prior R0 specififcation")
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