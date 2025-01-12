mutable struct MTBPParams{F<:Real,V<:AbstractVector}
    time::F
    const rates::V
    const cdfs::Vector{V}
end

function MTBPParams(nrates::Integer, ncdf_vec::AbstractVector{Integer}, T::Type=Float64)
    @assert nrates == length(ncdf_vec)
    rates = zeros(T, nrates)
    cdfs = [zeros(T, n) for n in ncdf_vec]
    return MTBPParams(-Inf, rates, cdfs)
end

function MTBPParams(model::MultitypeBranchingProcess)
    rates = copy(model.rates)
    cdfs = [copy(progeny.distribution) for progeny in model.progeny]
    return MTBPParams(-Inf, rates, cdfs)
end 

gettime(p::MTBPParams) = p.time

function insupport(p::MTBPParams)
    # rates are non-negative
    for rate in p.rates
        if rate < zero(rate)
            return false
        end
    end
    # cdfs are non-decreasing, non-negative and last entry is 1
    for cdf in p.cdfs
        p_prev = zero(eltype(cdf))
        p_next = one(p_prev)
        for p_next in cdf
            if p_next - p_prev < zero(p_prev)
                return false
            end
            p_prev = p_next
        end
        if !isone(p_next)
            return false
        end
    end
    return true
end

struct MTBPParamsSequence{P<:MTBPParams}
    seq::Vector{P}
    function MTBPParamsSequence{P}(seq::Vector{P}) where {P<:MTBPParams}
        if length(seq)>0
            prevtime = gettime(first(seq))
            for params in Iterators.drop(seq, 1)
                currtime = gettime(params)
                @assert currtime > prevtime "Expected param sequence to be in increasing time order."
                prevtime = currtime
            end
        end
        return new{P}(seq)
    end
    MTBPParamsSequence(seq::Vector{P}) where {P<:MTBPParams} = MTBPParamsSequence{P}(seq)
end

Base.getindex(p::MTBPParamsSequence, i) = p.seq[i]
Base.iterate(p::MTBPParamsSequence, i=1) = iterate(p.seq, i)
Base.firstindex(p::MTBPParamsSequence) = firstindex(p.seq)
Base.lastindex(p::MTBPParamsSequence) = lastindex(p.seq)
Base.first(p::MTBPParamsSequence) = first(p.seq)
Base.last(p::MTBPParamsSequence) = last(p.seq)
Base.length(p::MTBPParamsSequence) = length(p.seq)

function setparams!(model::MultitypeBranchingProcess, params::MTBPParams)
    setrates!(model, params.rates)
    for i in eachindex(model.progeny)
        params.cdfs[i]===nothing && continue
        setprogenycdf!(model, params.cdfs[i], i)
    end
    return
end
setparams!(model::StateSpaceModel, params::MTBPParams) = setparams!(model.stateprocess, params)

function similar_params(model::StateSpaceModel)
    return similar_params(model.stateprocess)
end
