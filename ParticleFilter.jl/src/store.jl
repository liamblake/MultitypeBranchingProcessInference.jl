# Weights

struct Weights{W<:AbstractVector{<:AbstractFloat}}
    count::Int
    values::W
    logarithm::W
    cumulative::W
    function Weights{W}(count::Int, values::W, logarithm::W, cumulative::W) where {W<:AbstractVector{<:AbstractFloat}}
        @assert length(values)==length(logarithm)==length(cumulative)==count "Weights fields must have the same length"
        return new{W}(count, values, logarithm, cumulative)
    end
    Weights(count::Int, values::W, logarithm::W, cumulative::W) where {W<:AbstractVector{<:AbstractFloat}} = 
        Weights{W}(count, values, logarithm, cumulative)
end

function paramtype(w::Weights)
    return eltype(w.values)
end

function Weights(T, count)
    values = Vector{T}(undef, count)
    logarithm = Vector{T}(undef, count)
    cumulative = Vector{T}(undef, count)
    return Weights(count, values, logarithm, cumulative)
end

# TODO: Extend to multiple obs 
# returns the loglikelihood of the obs
function calcweights!(weights::Weights, particles, statespacemodel, obs)
    for i in 1:weights.count
        setstate!(statespacemodel, particles[i])
        weights.logarithm[i] = logpdf(statespacemodel, obs)
    end
    # weights = exp(logarithm - shift)
    shift = shifted_exp!(weights.values, weights.logarithm)
    cumsum!(weights.cumulative, weights.values)
    # likelihood = sum(exp(logarithm))/N 
    #            = sum(exp(logarithm - shift + shift))/N
    #            = sum(exp(logarithm - shift)*exp(shift))/N
    #            = sum(weights*exp(shift))/N
    #            = exp(shift)*sum(weights)/N
    # loglikelihood = log( exp(shift)*sum(weights)/N )
    #               = shift + log(sum(weights)) - log(N)
    return shift + log(weights.cumulative[end]) - log(weights.count)
end

function shifted_exp!(out, x)
    shift = maximum(x)
    out .= x
    out .-= shift
    out .= exp.(out)
    return shift
end

function Random.rand(rng::AbstractRNG, weights::Weights)
    r = rand(rng, eltype(weights.cumulative))
    r *= weights.cumulative[end]
    i = 1
    while weights.cumulative[i] <= r
        i += 1
    end
    return i
end

Random.rand(weights::Weights) = rand(Random.default_rng(), weights)

function ess(weights::Weights)
    s2 = zero(paramtype(weights))
    s = weights.cumulative[end]
    for wi in weights.values
        r = wi/s
        s2 += r^2
    end
    return one(paramtype(weights))/s2
end

###############################################################################

struct ParticleStore{S<:AbstractVector, W<:AbstractArray}
    # a vector of particles
    count::Int
    store::S
    resample_store::S
    weights::Weights{W}
    function ParticleStore{S, W}(
        count::Int, store::S, resample_store::S, weights::Weights{W},
    ) where {S<:AbstractVector, W<:AbstractArray}
        @assert (
            count==length(store)==length(resample_store)==weights.count
        ) "ParticleStore fields must have the same length"
        statesize = length(first(store))
        for s in store
            @assert length(s)==statesize "All elements of store must be the same length"
        end
        for s in resample_store
            @assert length(s)==statesize "All resample_store of store must be the same length"
        end
        @assert eltype(store)==eltype(resample_store) "store must have the same element types"
        return new{S,W}(count, store, resample_store, weights)
    end
    ParticleStore(c::Int, s::S, rs::S, w::Weights{W}) where {S<:AbstractVector, W<:AbstractArray} = 
        ParticleStore{S, W}(c, s, rs, w)
end

function paramtype(s::ParticleStore)
    return paramtype(s.weights)
end

function elementtype(s::ParticleStore)
    return eltype(first(s.store))
end

function particletype(s::ParticleStore)
    return eltype(s.store)
end

function ParticleStore(weights_type::Type, state, count::Integer)
    store = [similar(state) for _ in 1:count]
    resample_store = [similar(state) for _ in 1:count]
    weights = Weights(weights_type, count)
    return ParticleStore(count, store, resample_store, weights)
end
ParticleStore(state, count) = 
    ParticleStore(Float64, state, count)

function getntypes(p::ParticleStore)
    return length(first(p.store))
end

# returns the loglikelihood of the obs
function calcweights!(store::ParticleStore, statespacemodel, obs)
    return calcweights!(store.weights, store.store, statespacemodel, obs)
end

function ess(store::ParticleStore)
    return ess(store.weights)
end

## Simulation for particle filter
abstract type AbstractThreadInfo end
struct SingleThreadded <: AbstractThreadInfo end
struct MultiThreadded{S} <: AbstractThreadInfo
    memalloc::S
end

function MultiThreadded(mtbp::MultitypeBranchingProcess)
    return MultiThreadded([similar(mtbp._deathdistribution) for _ in 1:Threads.nthreads()])
end

function initstate!(rng::AbstractRNG, particles::ParticleStore, statespacemodel, thread_info::SingleThreadded=SingleThreadded())
    for particle in particles.store
        rand!(rng, statespacemodel, particle)
    end
    return 
end
function chunkup(n::Int, nchunks::Int)
    smallchunksize = n÷nchunks
    bigchunksize = smallchunksize + 1
    nbigchunks = n - smallchunksize*nchunks
    chunks = Vector{UnitRange{Int}}(undef, nchunks)
    chunkstart = 1
    for i in 1:nbigchunks
        chunks[i] = chunkstart:(chunkstart+bigchunksize-1)
        chunkstart += bigchunksize
    end
    for i in (nbigchunks+1):nchunks
        chunks[i] = chunkstart:(chunkstart+smallchunksize-1)
        chunkstart += smallchunksize
    end
    return chunks
end
function initstate!(rng::AbstractRNG, particles::ParticleStore, statespacemodel, thread_info::MultiThreadded)
    chunks = chunkup(particles.count, Threads.nthreads())
    Threads.@threads for chunk in chunks
        for i in chunk
            rand!(rng, statespacemodel, particles.store[i])
        end
    end
    return 
end
initstate!(particles::ParticleStore, statespacemodel, thread_info=SingleThreadded()) = 
    initstate!(Random.default_rng(), particles, statespacemodel, thread_info)

function simulatestate!(rng::AbstractRNG, particles::ParticleStore, statespacemodel, t, thread_info::SingleThreadded=SingleThreadded())
    for particle in particles.store
        simulatestate!(rng, particle, statespacemodel, t)
    end
end
function simulatestate!(rng::AbstractRNG, particles::ParticleStore, statespacemodel, t, thread_info::MultiThreadded,
)
    chunks = chunkup(particles.count, Threads.nthreads())
    Threads.@threads for chunkid in eachindex(chunks)
        memalloc = thread_info.memalloc[chunkid]
        chunk = chunks[chunkid]
        for i in chunk
            simulatestate!(rng, particles.store[i], statespacemodel, t, memalloc)
        end
    end
end
simulatestate!(particles::ParticleStore, statespacemodel, t, thread_info=SingleThreadded()) = 
    simulatestate!(Random.default_rng(), particles::ParticleStore, statespacemodel, t, thread_info)

## NEED TO RESAMPLE WEIGHTS TOO!! TO CALC LIKELIHOOD
function resample!(rng::AbstractRNG, particles::ParticleStore)
    for particle in particles.resample_store
        resample_idx = rand(rng, particles.weights)
        particle .= particles.store[resample_idx]
    end
    particles.store .= particles.resample_store
    return 
end

resample!(particles) = resample!(Random.default_rng(), particles)

function mean!(mu, store::ParticleStore)
    mu .= zero(eltype(mu))
    for particle in store.store
        mu .+= particle
    end
    mu ./= store.count
    return mu
end
mean(store::ParticleStore) = mean!(zeros(paramtype(store), getntypes(store)), store)

function vcov!(c, store::ParticleStore, mu=mean(store), cache=similar(mu))
    c .= zero(eltype(c))
    for particle in store.store
        mul!(c, particle, particle', 1, 1)
    end
    for i in eachindex(mu)
        cache .= mu
        cache .*= (mu[i]*store.count)
        c[:,i] .-= cache
    end
    c ./= store.count-1
    return c
end
vcov(store::ParticleStore, mu=mean(store), cache=similar(mu)) = 
    vcov!(zeros(paramtype(store), getntypes(store), getntypes(store)), store, mu, cache)

function summarise!(mu, vcov, store::ParticleStore, cache=similar(mu))
    mean!(mu, store)
    vcov!(vcov, store, mu, cache)
    return mu, vcov
end