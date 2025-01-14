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
)
    return 
end

function init!(f::ParticleFilterApproximation, ssm::StateSpaceModel, observation)
    println("Init PF")
    ll = init!(f.rng, f.store, ssm, observation, f.thread_info)
    println(
        "ESS: ",
        1/sum((f.store.weights.values/f.store.weights.cumulative[end]).^2)
    )
    return ll
end

function iterate!(
    f::ParticleFilterApproximation, ssm::StateSpaceModel, dt, observation
)
    ll = iterate!(f.rng, f.store, ssm, dt, observation, f.thread_info)
    println(
        "ESS: ",
        1/sum((f.store.weights.values/f.store.weights.cumulative[end]).^2)
    )
    return ll
end
