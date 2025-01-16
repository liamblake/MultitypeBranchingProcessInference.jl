mutable struct SwitchParams{R<:AbstractRNG, T<:Real, I<:AbstractVector{<:Integer}, NT<:NamedTuple}
    const rng::R
    const threshold::T
    const randomstatesidx::I
    const memcache::NT
    currfiltername::Symbol
end

function SwitchParams(model, rng, threshold, randomstatesidx)
    memcache = makeswitchcache(paramtype(model), getntypes(model), length(randomstatesidx))
    return SwitchParams(rng, threshold, randomstatesidx, memcache, :pfapprox)
end

struct HybridFilterApproximation{K<:MTBPKalmanFilterApproximation, P<:ParticleFilterApproximation, S<:SwitchParams}
    kfapprox::K
    pfapprox::P
    switchparams::S
end

function HybridFilterApproximation(
    model, pfrng::AbstractRNG, switchrng::AbstractRNG, 
    nparticles::Integer, threshold::Real, randomstatesidx=1:getntypes(model),
)
    kfapprox = MTBPKalmanFilterApproximation(model)
    pfapprox = ParticleFilterApproximation(model, pfrng, nparticles)
    switchparams = SwitchParams(model, switchrng, threshold, randomstatesidx)
    return HybridFilterApproximation(kfapprox, pfapprox, switchparams)
end

function HybridFilterApproximation(
    model, rng::AbstractRNG,
    nparticles::Integer, threshold::Real, randomstatesidx,
)
    return HybridFilterApproximation(
        model, rng, rng, nparticles, threshold, randomstatesidx,
    )
end

function HybridFilterApproximation(
    model, nparticles::Integer, threshold::Real, randomstatesidx,
)
    rng = Random.default_rng()
    return HybridFilterApproximation(
        model, rng, rng, nparticles, threshold, randomstatesidx,
    )
end

function mean!(out, kfapprox::MTBPKalmanFilterApproximation)
    return mean!(out, kfapprox.kalmanfilter)
end
function mean(kfapprox::MTBPKalmanFilterApproximation)
    return mean(kfapprox.kalmanfilter)
end

function mean!(out, pfapprox::ParticleFilterApproximation)
    return mean!(out, pfapprox.store)
end
function mean(pfapprox::ParticleFilterApproximation)
    return mean(pfapprox.store)
end

struct RoundedMvNormal{D<:AbstractMvNormal, V<:AbstractVector{<:Integer}, S<:Integer, W<:AbstractVector}
    dist::D
    statesidx::V
    T::Type{S}
    cache::W
end
function RoundedMvNormal(mu::Vector, cov::AbstractMatrix, idx, T, cache=similar(mu))
    return RoundedMvNormal(MvNormal(mu, cov), idx, T, cache)
end

function Random.rand!(rng::AbstractRNG, z::RoundedMvNormal, out)
    rand!(rng, z.dist, z.cache)
    for (cacheidx, outidx) in enumerate(z.statesidx)
        out[outidx] = round(z.T, z.cache[cacheidx])
        if out[outidx] < zero(eltype(out))
            out[outidx] = zero(eltype(out))
        end
    end
    return out
end

function makeswitchcache(T, ntypes, nrandomstates)
    return (
        mu=zeros(T, ntypes), 
        vcovcache=zeros(T, ntypes), 
        mvnormmu=zeros(T, nrandomstates), 
        mvnormvcov=zeros(T, nrandomstates, nrandomstates),
        mvnormcache=zeros(T, nrandomstates), 
    )
end
# randomstatesidx=1:getntypes(model)

const MTBPLikelihoodApproximationMethods = Union{
    MTBPKalmanFilterApproximation,
    ParticleFilterApproximation,
    HybridFilterApproximation,
}

function itersetup!(
    hf::HybridFilterApproximation,
    model::StateSpaceModel, dt::Union{Real,Nothing}, observation, 
    iteration::Real, use_prev_iter_params::Bool,
    customitersetup=nothing,
)
    isfirstiter = (iteration == one(iteration))
    isfirstiter && init!(hf, model)
    
    currfiltername = hf.switchparams.currfiltername
    currfilter = getfield(hf, currfiltername)
    if !isfirstiter
        mean!(hf.switchparams.memcache.mu, currfilter)
    end
    
    newfiltername = :kfapprox
    for i in hf.switchparams.randomstatesidx
        if hf.switchparams.memcache.mu[i] < hf.switchparams.threshold
            newfiltername = :pfapprox
            break
        end
    end
    hf.switchparams.currfiltername = newfiltername

    # No switch of filter or is the init step (iteration == 1)
    if (newfiltername === currfiltername) || isfirstiter
        # no more hf-specific set up to be done
        # newfilter set up is executed by call to iterate for the specific filter
        return false
    end

    newfilter = getfield(hf, newfiltername)

    if (currfilter isa ParticleFilterApproximation 
        && newfilter isa MTBPKalmanFilterApproximation)
        # switch from PF to KF
        pfapprox = currfilter
        pfstore = pfapprox.store
        kfapprox = newfilter
        kf = kfapprox.kalmanfilter

        mean!(kf.state_estimate, pfstore)
        vcov!(kf.state_estimate_covariance, pfstore, hf.switchparams.memcache.mu, hf.switchparams.memcache.vcovcache)
    elseif (currfilter isa MTBPKalmanFilterApproximation 
        && newfilter isa ParticleFilterApproximation)
        # switch from KF to PF
        kfapprox = currfilter
        kf = kfapprox.kalmanfilter
        pfapprox = newfilter
        pfstore = pfapprox.store

        randomstatesidx = hf.switchparams.randomstatesidx
        for j in axes(hf.switchparams.memcache.mvnormvcov, 2)
            hf.switchparams.memcache.mvnormmu[j] = kf.state_estimate[randomstatesidx[j]]
            for i in axes(hf.switchparams.memcache.mvnormvcov, 1)
                hf.switchparams.memcache.mvnormvcov[i,j] = 
                    kf.state_estimate_covariance[randomstatesidx[i], randomstatesidx[j]]
            end
        end

        dist = RoundedMvNormal(
            hf.switchparams.memcache.mvnormmu,
            # wrap in Hermitian type because the KalmanFilter may have rounding errors
            Hermitian(hf.switchparams.memcache.mvnormvcov),
            randomstatesidx, 
            elementtype(pfstore), 
            hf.switchparams.memcache.mvnormcache,
        )
        
        # fill the pfstore by sampling from dist
        initstate!(hf.switchparams.rng, pfstore, dist)
    else 
        error("Unknown filterargs types.")
    end
    if customitersetup !== nothing
        customitersetup(hf, model, dt, observation, iteration, use_prev_iter_params)
    end
    return true
end

function init!(
    f::HybridFilterApproximation, ssm::StateSpaceModel{S,O}, 
) where {S<:MultitypeBranchingProcess, O<:LinearGaussianObservationModel}
    f.switchparams.memcache.mu .= ssm.stateprocess.initial_state.first_moments
    return 
end

function iterate!(
    f::HybridFilterApproximation, ssm::StateSpaceModel{S,O}, dt, observation,
    iteration::Real, use_prev_iter_params::Bool, customitersetup=nothing,
) where {S<:MultitypeBranchingProcess, O<:LinearGaussianObservationModel} 
    isswitch = itersetup!(f, ssm, dt, observation, iteration, use_prev_iter_params, customitersetup)
    filter = getfield(f, f.switchparams.currfiltername)
    return iterate!(filter, ssm, dt, observation, iteration, use_prev_iter_params && !isswitch, customitersetup)
end