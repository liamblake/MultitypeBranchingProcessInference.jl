# logpdf!
function logpdf!(
    model::StateSpaceModel, paramseq::MTBPParamsSequence, observations::Observations, 
    approx::MTBPLikelihoodApproximationMethods, customitersetup=nothing, verbose=true,
)
    iteration = 0
    currtime = zero(gettime(first(observations)))
    nextparamtime = gettime(first(paramseq))
    iszero(nextparamtime) || error("First paramseq timestamp must be 0, got $(nextparamtime).")
    nextparamidx = firstindex(paramseq)
    loglikelihood = zero(paramtype(model))
    prevdt = nothing

    # iterate
    for obs in observations
        # set params
        # this occurs at the time of the previous observation timestamp 
        # (or at timestamp 0 if this is the first iteration)
        updateparams = (currtime == nextparamtime)
        if updateparams
            currparams, nextparamidx = iterate(paramseq, nextparamidx)
            if !insupport(currparams)
                return -Inf
            end
            setparams!(model, currparams)

            nextparamtime = nextparamidx <= length(paramseq) ? gettime(paramseq[nextparamidx]) : Inf
        elseif currtime > nextparamtime
            println(paramseq)
            error("Parameters at timestamp $nextparamtime. Parameter timestamp must equal an observation timestamp.")
        end

        # now move to the time of the observation
        prevtime = currtime
        iteration += 1
        currtime = gettime(obs)
        dt = currtime - prevtime
        obs_value = getvalue(obs)

        # if the parameters have not been update and the timestep, dt, has not changed
        # then we can save some computations
        use_prev_iter_params = (!updateparams) && (dt==prevdt)
        
        loglikelihood += iterate!(approx, model, dt, obs_value, iteration, use_prev_iter_params, customitersetup)
        
        if isinf(loglikelihood) && loglikelihood < zero(loglikelihood)
            # loglikelihood is negative infinity, so we can shortcut further computations and return
            return loglikelihood
        end
        prevdt = dt
    end

    # Warn unused params
    if verbose && nextparamidx != length(paramseq)+1
        @warn "Unused parameters in paramseq."
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