const StateProcess = Union{MultitypeBranchingProcess}

struct StateSpaceModel{S<:StateProcess, O<:AbstractObservationModel}
    stateprocess::S
    observation_model::O
    function StateSpaceModel{S,O}(s::S,o::O) where {S<:StateProcess, O<:AbstractObservationModel}
        @assert paramtype(s)==paramtype(o)
        return new{S,O}(s,o)
    end
    StateSpaceModel(s::S,o::O) where {S<:StateProcess, O<:AbstractObservationModel} = 
        StateSpaceModel{S,O}(s,o)
end

getntypes(s::StateSpaceModel) = getntypes(s.stateprocess)

function paramtype(s::StateSpaceModel)
    return paramtype(s.stateprocess)
end

function variabletype(s::StateSpaceModel)
    return variabletype(s.stateprocess)
end

function statetype(s::StateSpaceModel)
    return statetype(s.stateprocess)
end

function setstate!(s::StateSpaceModel, state)
    return setstate!(s.stateprocess, state)
end

function getstate(s::StateSpaceModel)
    return getstate(s.stateprocess)
end

function rand!(rng::AbstractRNG, m::StateSpaceModel, out)
    return init!(rng, out, m.stateprocess)
end

function simulatestate!(rng::AbstractRNG, state::AbstractVector, m::StateSpaceModel, t, memcache=m.stateprocess._deathdistribution) 
    return simulate!(rng, state, m.stateprocess, t, memcache)
end

function logpdf(m::StateSpaceModel, y)
    return logpdf(m.observation_model, y, getstate(m.stateprocess))
end