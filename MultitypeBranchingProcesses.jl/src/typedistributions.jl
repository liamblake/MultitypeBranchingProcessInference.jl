struct MTBPDiscreteDistribution{D<:AbstractVector{<:Real}, E<:AbstractVector{<:AbstractVector}, V<:AbstractVector, C<:AbstractMatrix}
    ntypes::Int
    nevents::Int
    # cdf of progeny type distribution idx
    distribution::D
    # array of progeny events
    # [e1 e2 ... eN]
    # where ei is a vector of containing the progeny increments for progeny type i
    events::E
    # characterisation
    first_moments::V
    second_moments::C
    function MTBPDiscreteDistribution{D,E,V,C}(
        ntypes::Int, nevents::Int, distribution::D, events::E, first_moments::V, second_moments::C
    ) where {D<:AbstractVector, E<:AbstractVector, V<:AbstractVector, C<:AbstractMatrix}
        @assert issorted(distribution) "distribution is not monotonic"
        if length(distribution)>0 
            @assert distribution[1]>=zero(eltype(distribution)) "distribution is not non-negative"
            @assert distribution[end]==one(eltype(distribution)) "distribution is defective"
            @assert length(first_moments)==ntypes "first_moments must be the same length as the number of types"
            @assert size(second_moments)==(ntypes, ntypes) "second_moments must be square with size equal to the number of types"
        end
        @assert length(distribution)==length(events)==nevents "number of events does not match length of distribution"
        for e in events
            @assert length(e)==ntypes "all events must have the same size"
        end
        return new{D,E,V,C}(ntypes, nevents, distribution, events, first_moments, second_moments)
    end
    MTBPDiscreteDistribution(
        ntypes::Int, nevents::Int, distribution::D, events::E, first_moments::V, second_moments::C
    ) where {D<:AbstractVector, E<:AbstractVector, V<:AbstractVector, C<:AbstractMatrix} = 
        MTBPDiscreteDistribution{D,E,V,C}(ntypes, nevents, distribution, events, first_moments, second_moments)
end

function dummy_mtbp_discrete_distribution(ntypes, statetype, partype)
    return MTBPDiscreteDistribution(
        ntypes, 
        0,                            # n events
        zeros(partype, 0),            # progeny cdf
        statetype[],                  # progeny events
        zeros(partype, ntypes),       # first moment
        zeros(partype, ntypes, ntypes) # second moments
    )
end

function variabletype(d::MTBPDiscreteDistribution)
    if d.nevents==0
        return nothing
    end
    return eltype(d.events[1])
end

function paramtype(d::MTBPDiscreteDistribution)
    return eltype(d.distribution)
end

function statetype(d::MTBPDiscreteDistribution)
    return eltype(d.events)
end

function getntypes(d::MTBPDiscreteDistribution)
    return d.ntypes
end

function MTBPDiscreteDistribution(d::D, e::E) where {E<:AbstractVector, D<:AbstractVector}
    ntypes = length(first(e))
    nevents = length(d)
    ex = similar(first(e), eltype(d))
    m2 = similar(first(e), eltype(d), (ntypes, ntypes))
    firstmoment!(ex, e, d)
    secondmoment!(m2, e, d)
    dist = MTBPDiscreteDistribution(ntypes, nevents, d, e, ex, m2)
    return dist
end

function rand_idx(rng::AbstractRNG, cdf)
    r = rand(rng, eltype(cdf))*cdf[end]
    i = 1
    while cdf[i] <= r
        i += 1
    end
    return i
end

function firstmoment!(out, progeny::MTBPDiscreteDistribution)
    return firstmoment!(out, progeny.events, progeny.distribution)
end

function firstmoment!(out, events, cdf)
    out .= zero(eltype(cdf))
    q = zero(eltype(cdf))
    for eventidx in eachindex(events)
        out .+= events[eventidx]*(cdf[eventidx] - q)
        q = cdf[eventidx]
    end
    return
end

function secondmoment!(out, progeny::MTBPDiscreteDistribution)
    return secondmoment!(out, progeny.events, progeny.distribution)
end

function secondmoment!(out, events, cdf)
    out .= zero(eltype(out))
    q = zero(eltype(cdf))
    for eventidx in eachindex(events)
        event = events[eventidx]
        p = (cdf[eventidx] - q)
        for idx in eachindex(event)
            out[:,idx] .+= event*event[idx]*p
        end
        q = cdf[eventidx]
    end
    return
end

function setcdf!(dist, cdf)
    dist.distribution .= cdf
    firstmoment!(dist.first_moments, dist)
    secondmoment!(dist.second_moments, dist)
    return
end

function rand(rng::AbstractRNG, progeny::MTBPDiscreteDistribution, nevents::Val)
    return progeny.events[rand_idx(rng, progeny.distribution)]
end
function rand!(rng::AbstractRNG, progeny::MTBPDiscreteDistribution, nevents::Val, out)
    out .= progeny.events[rand_idx(rng, progeny.distribution)]
    return out
end
function rand(rng::AbstractRNG, progeny::MTBPDiscreteDistribution, nevents::Val{0})
    error("The variable \"nevents\" is 0, there are no events to sample from.")
    return 
end
function rand!(rng::AbstractRNG, progeny::MTBPDiscreteDistribution, nevents::Val{0}, out)
    error("The variable \"nevents\" is 0, there are no events to sample from.")
    return 
end
function rand(rng::AbstractRNG, progeny::MTBPDiscreteDistribution, nevents::Val{1})
    return only(progeny.events)
end
function rand!(rng::AbstractRNG, progeny::MTBPDiscreteDistribution, nevents::Val{1}, out)
    out .= only(progeny.events)
    return out
end

rand(rng::AbstractRNG, progeny::MTBPDiscreteDistribution) = 
    rand(rng::AbstractRNG, progeny::MTBPDiscreteDistribution, Val(progeny.nevents))
rand(p::MTBPDiscreteDistribution) = rand(Random.default_rng(), p)
rand!(rng::AbstractRNG, progeny::MTBPDiscreteDistribution, out) = 
    rand!(rng::AbstractRNG, progeny::MTBPDiscreteDistribution, Val(progeny.nevents), out)
rand!(p::MTBPDiscreteDistribution, out) = rand!(Random.default_rng(), p, out)

const ProgenyDistribution = MTBPDiscreteDistribution
const InitialStateDistribution = MTBPDiscreteDistribution