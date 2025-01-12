function init!(rng::AbstractRNG, state::AbstractVector, m::MultitypeBranchingProcess)
    rand!(rng, m.initial_state, state)
    return state
end

init!(state::AbstractVector, m::MultitypeBranchingProcess) = init!(Random.default_rng(), state, m)
init!(m::MultitypeBranchingProcess) = init!(Random.default_rng(), m.state, m)
init!(rng::AbstractRNG, m::MultitypeBranchingProcess) = init!(rng, m.state, m)

function transition_event_params!(
    m::MultitypeBranchingProcess, state::AbstractVector, deathdistribution=similar(m._deathdistribution)
)
    # state contains counts of individuals of each type, so
    # the total rate is the sum of the rate of the individuals
    total_rate = zero(eltype(m.rates))
    for i in eachindex(state)
        total_rate += state[i]*m.rates[i]
        deathdistribution[i] = total_rate
    end
    return total_rate
end

function rand_time_increment(rng::AbstractRNG, total_rate::T) where {T}
    return -log(rand(rng, T))/total_rate
end

rand_time_increment(r) = rand_time_increment(Random.default_rng(), r)

function simulate!(
    rng::AbstractRNG, 
    state::AbstractVector, 
    m::MultitypeBranchingProcess, 
    t,
    deathdistribution=m._deathdistribution
)
    # rate of next event
    total_rate = transition_event_params!(m, state, deathdistribution)
    # next event time
    T = rand_time_increment(rng, total_rate)
    while T < t
        # determine progeny event type
        death_idx = rand_idx(rng, deathdistribution)
        state .+= rand(rng, m.progeny[death_idx])
        # racalculate event rate to account for death and new progeny
        total_rate = transition_event_params!(m, state, deathdistribution)
        # next event time 
        T += rand_time_increment(rng, total_rate)
    end
    return
end

simulate!(s::AbstractVector, m::MultitypeBranchingProcess, t, d=m._deathdistribution) = 
    simulate!(Random.default_rng(), s, m, t, d)
simulate!(m::MultitypeBranchingProcess, t, d=m._deathdistribution) = 
    simulate!(Random.default_rng(), m.state, m, t, d)
simulate!(rng::AbstractRNG, m::MultitypeBranchingProcess, t, d=m._deathdistribution) = 
    simulate!(rng, m.state, m, t, d)
