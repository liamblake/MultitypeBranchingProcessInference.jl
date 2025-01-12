abstract type SymmetricProposalDistribution end

mutable struct MutableMvNormal{D<:MvNormal} <: SymmetricProposalDistribution
    distribution::D
end

function MutableMvNormal(mu, cov)
    return MutableMvNormal(MvNormal(mu, cov))
end

function setstate!(proposal, value)
    error("setstate! requires a custom implementation for your proposal")
    return 
end

function adapt!(proposal, samples_buffer)
    error("adapt! requires a custom implementation for your proposal")
    return 
end

function MetropolisHastings.setstate!(mvn_proposal::MutableMvNormal, value)
    mvn_proposal.distribution.μ .= value
    return 
end

function MetropolisHastings.adapt!(mvn_proposal::MutableMvNormal, param_samples, scale)
    cov_mat = cov(param_samples, dims=2)
    if cov_mat == zero(cov_mat) || !isposdef(cov_mat)
        # not enough samples
        return 
    end
    cov_mat .*= scale
    mvn_proposal.distribution = MvNormal(mvn_proposal.distribution.μ, cov_mat)
    return 
end

function Random.rand(rng::AbstractRNG, mvn_proposal::MutableMvNormal)
    return rand(rng, mvn_proposal.distribution)
end
