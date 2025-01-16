struct ParticleFilterApproximation{R<:AbstractRNG, S<:ParticleStore, T<:AbstractThreadInfo}
    rng::R
    store::S
    thread_info::T
end

function ParticleFilterApproximation(model, rng, nparticles, multithredded=false)
    store = ParticleStore(paramtype(model), getstate(model), nparticles)
    if multithredded
        thread_info = MultiThreadded(model.stateprocess)
    else
        thread_info = SingleThreadded()
    end
    return ParticleFilterApproximation(rng, store, thread_info)
end

function itersetup!(
    f::ParticleFilterApproximation,
    model::StateSpaceModel, dt::Union{Real,Nothing}, observation, 
    iteration::Real, use_prev_iter_params::Bool,
    customitersetup=nothing,
)
    iteration==one(iteration) && init!(f, model)
    if customitersetup!==nothing
        customitersetup(f, model, dt, observation, iteration, use_prev_iter_params)
    end
    return 
end

function init!(f::ParticleFilterApproximation, ssm::StateSpaceModel)
    return init!(f.rng, f.store, ssm, f.thread_info)
end

function iterate!(
    f::ParticleFilterApproximation,
    ssm::StateSpaceModel, dt, observation, 
    iteration::Real, use_prev_iter_params::Bool=false,
    customitersetup=nothing,
)
    itersetup!(f, ssm, dt, observation, iteration, use_prev_iter_params, customitersetup)
    return iterate!(f.rng, f.store, ssm, dt, observation, f.thread_info)
end
