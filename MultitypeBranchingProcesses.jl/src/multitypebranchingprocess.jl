struct MultitypeBranchingProcess{S<:AbstractVector,P<:ProgenyDistribution,R<:AbstractVector,D<:AbstractVector}
    ntypes::Int
    state::S
    initial_state::P
    progeny::Vector{P}
    rates::R
    _deathdistribution::D
    function MultitypeBranchingProcess{S,P,R,D}(
        ntypes::Int, state::S, initial::P, progeny::Vector{P}, rates::R, deathdistribution::D
    ) where {S<:AbstractVector,P<:ProgenyDistribution,R<:AbstractVector,D<:AbstractVector}
        @assert ntypes == length(rates) == length(deathdistribution) == length(state) "model types count and rates parametrs mismatch, got $(ntypes), $(length(rates)), $(length(deathdistribution)), $(length(state))"
        @assert getntypes(initial) == ntypes "initial and progeny distribution must have same state size"
        @assert ntypes == length(progeny) "initial and progeny distribution must have same state size, got $(ntypes) and $(length(progeny))"
        @assert eltype(deathdistribution) == eltype(rates) == paramtype(initial) "eltypes of parameters must match"
        for p in progeny
            @assert eltype(rates) == paramtype(p) "eltypes of parameters must match"
            @assert ntypes == getntypes(p) "model types count and progeny events dimension mismatch"
            if variabletype(p) !== nothing
                @assert variabletype(initial) == variabletype(p) "initial state, progeny distributions events and rates must have the same element types"
            end
            @assert statetype(initial) == statetype(p) "got $(statetype(initial)), $(statetype(p))"
            @assert typeof(state) == typeof(similar(first(initial.events))) "got $(statetype(state)), $(similar(first(initial.events)))"
        end
        return new{S,P,R,D}(
            ntypes, state, initial, progeny, rates, deathdistribution
        )
    end
    MultitypeBranchingProcess(
        ntypes::Int, state::S, initial::P, progeny::Vector{P}, rates::R, deathdistribution::D
    ) where {S<:AbstractVector,P<:ProgenyDistribution,R<:AbstractVector,D<:AbstractVector} =
        MultitypeBranchingProcess{S,P,R,D}(ntypes, state, initial, progeny, rates, deathdistribution)
end

function MTBPMomentsOperator(bp::MultitypeBranchingProcess)
    return MTBPMomentsOperator(getntypes(bp), paramtype(bp))
end
function mean(mtbp::MultitypeBranchingProcess, t)
    op = MTBPMomentsOperator(getntypes(mtbp), variabletype(mtbp))
    moments!(op, mtbp, t)
    return mean(op, m.state)
end
function variance_covariance(mtbp::MultitypeBranchingProcess, t)
    op = MTBPMomentsOperator(getntypes(mtbp), variabletype(mtbp))
    moments!(op, mtbp, t)
    return variance_covariance(op, m.state)
end

function setstate!(m::MultitypeBranchingProcess, state)
    return m.state .= state
end

function getstate(m::MultitypeBranchingProcess)
    return m.state
end

function getntypes(m::MultitypeBranchingProcess)
    return m.ntypes
end

function variabletype(m::MultitypeBranchingProcess)
    return eltype(getstate(m))
end

function paramtype(m::MultitypeBranchingProcess)
    return eltype(m.rates)
end

function statetype(m::MultitypeBranchingProcess)
    return typeof(getstate(m))
end

function MultitypeBranchingProcess(ntypes, initial, progeny, rates)
    state = similar(first(initial.events))
    deathdistribution = similar(rates)
    deathdistribution .= zero(eltype(deathdistribution))
    mtbp = MultitypeBranchingProcess(ntypes, state, initial, progeny, rates, deathdistribution)
    init!(mtbp)
    return mtbp
end

function setrates!(bp, rates)
    bp.rates .= rates
    return
end

function setprogenycdf!(bp, dist, progeny_idx)
    setcdf!(bp.progeny[progeny_idx], dist)
    return
end

function characteristicmatrix!(m, bp)
    for typeidx in 1:bp.ntypes
        firstmoment!(@view(m[:, typeidx]), bp.progeny[typeidx])
        m[:, typeidx] .*= bp.rates[typeidx]
    end
    return
end

function getcharacteristicmatrix(op)
    return op.characteristicmatrix
end

function getmeanoperator(op)
    p = getntypes(op)
    return op.generator[end-p+1:end, end-p+1:end]
end

function Base.show(io::IO, mtbp::MultitypeBranchingProcess)
    println(io, "Multitype Branching Process")
    println(io, "Number of types: $(getntypes(mtbp))")
    println(io, "State prototype: $(getstate(mtbp))")
    println(io, "Initial distribution: $(mtbp.initial_state.distribution)")

    println("Progeny:")
    for i in eachindex(mtbp.progeny)
        println("\tType $(i):")
        println(io, "\t\t Death rate: $(mtbp.rates[i])")
        println(io, "\t\t Progeny distribution: ")

        prog_cdf = diff([0.0; mtbp.progeny[i].distribution])

        for j in eachindex(mtbp.progeny[i].events)
            println(io, "\t\t\t $(mtbp.progeny[i].events[j]) with probability $(prog_cdf[j])")
        end
    end
end