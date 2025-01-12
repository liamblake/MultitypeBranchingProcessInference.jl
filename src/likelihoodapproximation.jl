# logpdf!
function logpdf!(
    model::StateSpaceModel, paramseq::MTBPParamsSequence, observations::Observations, 
    approx::MTBPLikelihoodApproximationMethods, customitersetup=nothing, verbose=true,
)
    # initialise
    iteration = 1

    obs = first(observations)
    currtime = gettime(obs)
    obs_value = getvalue(obs)

    currparams, nextparamidx = iterate(paramseq)
    paramtime = gettime(currparams)
    if paramtime > currtime
        error("Initial parameter timestamp must before the timestamp of the first observation.")
    end
    if !insupport(currparams)
        return -Inf
    end
    setparams!(model, currparams)
    nextparamtime = nextparamidx <= length(paramseq) ? gettime(paramseq[nextparamidx]) : Inf
    
    use_prev_iter_params = true
    dt = nothing
    itersetup!(approx, model, dt, obs_value, iteration, use_prev_iter_params)
    if customitersetup !== nothing
        customitersetup(approx, model, nothing, obs_value, iteration, use_prev_iter_params)
    end

    loglikelihood = init!(approx, model, obs_value)

    prevtime = currtime
    prevdt = dt

    # iterate
    for obs in Iterators.drop(observations, 1)
        iteration += 1
        currtime = gettime(obs)

        paramschanged = currtime == nextparamtime
        
        if paramschanged
            currparams, nextparamidx = iterate(paramseq, nextparamidx)
            if !insupport(currparams)
                return -Inf
            end
            nextparamtime = nextparamidx <= length(paramseq) ? gettime(paramseq[nextparamidx]) : Inf

            setparams!(model, currparams)
        elseif currtime > nextparamtime
            println(paramseq)
            error("Parameters at timestamp $nextparamtime. Parameter timestamp must equal an observation timestamp.")
        end

        dt = currtime - prevtime
        obs_value = getvalue(obs)

        use_prev_iter_params = (!paramschanged) && (dt==prevdt)
        itersetup!(approx, model, dt, obs_value, iteration, use_prev_iter_params)
        if customitersetup !== nothing
            customitersetup(approx, model, dt, obs_value, iteration, use_prev_iter_params)
        end

        loglikelihood += iterate!(approx, model, dt, obs_value)
        prevtime = currtime
        prevdt = dt

        if isinf(loglikelihood)
            # if verbose 
            #     println("Negative log-loglikelihood with params: ")
            #     println(currparams)
            #     println("obs value:")
            #     println(obs_value)
            # end
            return loglikelihood
        end
    end

    # Warn unused params
    if verbose && nextparamidx != length(paramseq)+1
        println("[WARN] Unused parameters in paramseq.")
    end
    return loglikelihood
end

## Interface methods for logpdf
function logpdf!(
    model::StateSpaceModel, observations::Observations, approx=nothing, customitersetup=nothing, verbose=true,
)
    params = MTBPParams(model.stateprocess)
    paramseq = MTBPParamsSequence([params])
    return logpdf!(model, paramseq, observations, approx, customitersetup, verbose)
end

# default filter behaviour
function logpdf!(
    model::StateSpaceModel, paramseq::MTBPParamsSequence, 
    observations::Observations, approx::Nothing, 
    customitersetup=nothing, nparticles=256, rng=Random.default_rng(),
    verbose=true,
)
    pf = ParticleFilterApproximation(model, nparticles, rng)
    return logpdf!(model, paramseq, observations, pf, customitersetup, verbose)
end