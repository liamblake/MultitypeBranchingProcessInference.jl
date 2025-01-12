struct ParticleFilterApproximation{R<:AbstractRNG, S<:ParticleStore}
    rng::R
    store::S
end

function ParticleFilterApproximation(model, rng, nparticles)
    store = ParticleStore(paramtype(model), getstate(model), nparticles)
    return ParticleFilterApproximation(rng, store)
end

function itersetup!(
    f::ParticleFilterApproximation,
    model::StateSpaceModel, dt::Union{Real,Nothing}, observation, 
    iteration::Real, use_prev_iter_params::Bool,
)
    return 
end

function init!(f::ParticleFilterApproximation, ssm::StateSpaceModel, observation)
    return init!(f.rng, f.store, ssm, observation)
end

function iterate!(
    f::ParticleFilterApproximation, ssm::StateSpaceModel, dt, observation,
)
    return iterate!(f.rng, f.store, ssm, dt, observation)
end
