struct Observation{T<:Real,D}
    time::T
    data::D
end

gettime(obs::Observation) = obs.time
getvalue(obs::Observation) = obs.data

struct Observations{T,D}
    data::Vector{Observation{T,D}}
    function Observations{T,D}(data::Vector{Observation{T,D}}) where {T,D}
        prevtime = -Inf
        for observation in data
            @assert gettime(observation) > prevtime "Expected observation times to be in increasing order."
            prevtime = gettime(observation)
        end
        return new{T,D}(data)
    end
    Observations(data::Vector{Observation{T,D}}) where {T,D} = Observations{T,D}(data)
end

function Observations(t::AbstractVector{T}, d::AbstractVector{S}) where {T,S}
    @assert isconcretetype(T)
    @assert isconcretetype(S)
    return Observations([Observation(ti, di) for (ti, di) in zip(t,d)])
end

Base.getindex(p::Observations, i) = p.data[i]
Base.iterate(p::Observations, i=1) = iterate(p.data, i)
Base.firstindex(p::Observations) = firstindex(p.data)
Base.lastindex(p::Observations) = lastindex(p.data)
Base.first(p::Observations) = first(p.data)
Base.last(p::Observations) = last(p.data)
Base.length(p::Observations) = length(p.data)
