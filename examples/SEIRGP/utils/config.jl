function setenvironment!(config)
    if "env" in keys(config) && "blas_num_threads" in keys(config["env"])
        LinearAlgebra.BLAS.set_num_threads(config["env"]["blas_num_threads"])
    end
    return
end

function pathtodailycases(path, cases_idx)
    cumulative_cases = [[state[cases_idx]] for state in path]
    daily_cases = diff(cumulative_cases)
    cases = [[cumulative_cases[1]]; daily_cases]
    return cases
end

function convertseirparamstorates(R_0, T_E, T_I, E_state_count, I_state_count)
    # rate of symptom onset
    delta = E_state_count / T_E
    # rate of recovery
    lambda = I_state_count / T_I
    # rate of infection
    beta = R_0 / T_I
    return delta, lambda, beta
end

function param_map!(
    mtbpparams, E_state_count, I_state_count, seir_params, immigration,
    convert_to_rates=true
)
    if convert_to_rates
        R_0, T_E, T_I = seir_params
        # rate of symptom onset
        delta = one(T_E) / T_E
        # rate of recovery
        lambda = one(T_I) / T_I
        # rate of infection
        beta = R_0 * lambda
    else
        delta, lambda, beta = seir_params
    end

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
    mtbpparams.rates[end-1] = zero(eltype(mtbpparams.rates))
    mtbpparams.rates[end] = sum(immigration)

    p = beta / (beta + lambda)
    one_ = one(eltype(mtbpparams.rates))
    for i in infectious_states
        mtbpparams.cdfs[i][1] = p
        mtbpparams.cdfs[i][2] = one_
    end

    if mtbpparams.rates[end] == zero(eltype(immigration))
        mtbpparams.cdfs[end] .= range(zero(eltype(immigration)), one(eltype(immigration)), length(immigration))
    else
        mtbpparams.cdfs[end] .= cumsum(immigration)
        mtbpparams.cdfs[end] ./= mtbpparams.cdfs[end][end]
    end
    return mtbpparams
end

function makerng(seed)
    rng = Xoshiro()
    Random.seed!(rng, seed)
    return rng
end

function makemodel(config)
    seirconfig = config["model"]["stateprocess"]["params"]
    delta, lambda, beta = convertseirparamstorates(
        first(seirconfig["R_0"]), first(seirconfig["T_E"]), first(seirconfig["T_I"]),
        seirconfig["E_state_count"], seirconfig["I_state_count"],
    )
    seir = SEIR(
        seirconfig["E_state_count"], seirconfig["I_state_count"],
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

    param_seq = MTBPParamsSequence(MTBPParams{paramtype(model),Vector{paramtype(model)}}[])
    if seirconfig["is_time_homogeneous"]
        mtbpparams = MTBPParams(seir)
        beta, lambda, delta = seirconfig["infection_rate"], seirconfig["infectious_stage_chage_rate"], seirconfig["exposed_stage_chage_rate"]
        immigration = seirconfig["immigration_rate"]
        param_map!(
            mtbpparams,
            seirconfig["E_state_count"], seirconfig["I_state_count"],
            (delta, lambda, beta), immigration, false
        )
        push!(param_seq.seq, mtbpparams)
    else
        for i in eachindex(seirconfig["timestamps"])
            mtbpparams = MTBPParams(seir)
            paramtimestamp = seirconfig["timestamps"][i]
            R_0 = seirconfig["R_0"][i]
            T_E = seirconfig["T_E"][i]
            T_I = seirconfig["T_I"][i]
            immigration = seirconfig["immigration_rate"]
            param_map!(
                mtbpparams,
                seirconfig["E_state_count"], seirconfig["I_state_count"],
                (R_0, T_E, T_I),
                immigration
            )
            mtbpparams.time = paramtimestamp
            push!(param_seq.seq, mtbpparams)
        end
    end

    return model, param_seq
end

function makepriordists(config)
    cts_prior_dists = Any[
        Gamma(config["inference"]["prior_parameters"]["R_0"]["shape"],
            config["inference"]["prior_parameters"]["R_0"]["scale"]),
        Gamma(config["inference"]["prior_parameters"]["T_E"]["shape"],
            config["inference"]["prior_parameters"]["T_E"]["scale"]),
        Gamma(config["inference"]["prior_parameters"]["T_I"]["shape"],
            config["inference"]["prior_parameters"]["T_I"]["scale"]),
    ]
    discrete_prior_dists = Any[]

    if "intervention" in keys(config["inference"]["prior_parameters"])
        push!(
            cts_prior_dists,
            Beta(config["inference"]["prior_parameters"]["intervention"]["effect"]["alpha"],
                config["inference"]["prior_parameters"]["intervention"]["effect"]["beta"]),
        )
        push!(
            discrete_prior_dists,
            DiscreteUniform(config["inference"]["prior_parameters"]["intervention"]["time"]["lower"],
                config["inference"]["prior_parameters"]["intervention"]["time"]["upper"]),
        )
    end
    return cts_prior_dists, discrete_prior_dists
end

function makeprior(config)
    cts_prior_dists, discrete_prior_dists = makepriordists(config)

    cts_prior_dists = tuple(cts_prior_dists...)
    discrete_prior_dists = tuple(discrete_prior_dists...)

    function prior_logpdf(params)
        val = zero(eltype(params))
        for i in eachindex(cts_prior_dists)
            val += logpdf(cts_prior_dists[i], params[i])
        end
        for i in eachindex(discrete_prior_dists)
            val += logpdf(discrete_prior_dists[i], round(Int, params[i+length(cts_prior_dists)]))
        end
        return val
    end
    return prior_logpdf
end

function makeproposal(config)
    propconfig = config["inference"]["proposal_parameters"]
    mu = propconfig["mean"]
    sigma = reshape(propconfig["cov"], length(mu), length(mu))
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

function makeloglikelihood(model, param_seq, observations, config)
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

        approx = ParticleFilterApproximation(model, pf_rng, nparticles)

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
            mtbpparams, seirconfig["E_state_count"], seirconfig["I_state_count"],
            param, seirconfig["immigration_rate"], true
        )
    end
    if "intervention" in keys(config["inference"]["prior_parameters"])
        pre_intervention_params = zeros(paramtype(model), 3)
        post_intervention_params = zeros(paramtype(model), 3)
        loglikelihood = (pars) -> begin # function loglikelihood(pars)
            # pre-itervention
            pre_intervention_params .= pars[1:3]
            llparam_map!(param_seq[1], pre_intervention_params)

            # post-itervention and intervention time
            post_intervention_params .= pre_intervention_params
            post_intervention_params[1] *= pars[4]
            llparam_map!(param_seq[2], post_intervention_params)
            param_seq[2].time = round(typeof(param_seq[2].time), pars[5])

            return logpdf!(model, param_seq, observations, approx, reset_obs_state_iter_setup!)
        end
    else
        loglikelihood = (pars) -> begin # function loglikelihood(pars)
            llparam_map!(only(param_seq), pars)
            return logpdf!(model, param_seq, observations, approx, reset_obs_state_iter_setup!)
        end
    end
    return loglikelihood
end