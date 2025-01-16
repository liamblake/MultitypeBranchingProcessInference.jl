function particlefilter!(
    rng::AbstractRNG, 
    particles::ParticleStore, 
    model,
    observations::Observations,
    callback=nothing,
    thread_info=SingleThreadded(),
    check_inputs=true,
)
    check_inputs && particlefilter_checks(rng, model, particles)
    
    # initialise
    init!(rng, model, particles, thread_info)
    if callback!==nothing
        callback(particles, model, observations, iteration)
    end

    # iterate
    iteration = 0
    prevtime = zero(gettime(first(observations.data)))
    for obs in observations.data
        iteration += 1
        currtime = gettime(obs)
        dt = currtime - prevtime
        observation = getvalue(obs)
        loglikelihood += iterate!(rng, particles, model, dt, observation, thread_info)
        if callback!==nothing
            callback(particles, model, observations, iteration)
        end
        prevtime = currtime
    end
    return loglikelihood
end

function init!(
    rng::AbstractRNG, 
    particles::ParticleStore, 
    model,
    thread_info=SingleThreadded(),
)
    return initstate!(rng, particles, model, thread_info)
end

function iterate!(
    rng::AbstractRNG, 
    particles::ParticleStore, 
    model,
    dt, 
    observation,
    thread_info=SingleThreadded(),
)
    simulatestate!(rng, particles, model, dt, thread_info)
    loglikelihood = calcweights!(particles, model, observation)
    resample!(rng, particles)
    return loglikelihood
end

function particlefilter_checks(rng, m, p)
    @assert getntypes(m)==getntypes(p) "ParticleStore store and state process must have the same state size"
    @assert variabletype(m)==elementtype(p)
    @assert statetype(m)==particletype(p)
    return
end
