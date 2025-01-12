@testset "loglikelihoods" begin
    seir = SEIR(1, 1, 0.4, 0.4, 0.4, 1.0, 0.4, [30;30;30;1])
    obs_model = LinearGaussianObservationModel([0. 0. 1. 0.])
    ssm = StateSpaceModel(seir, obs_model)
    kfapprox = MTBPStateSpaceModelKalmanFilterApproximation(ssm)
    rng = StableRNGs.StableRNG(1234)
    paramtime = -Inf
    rates = seir.rates
    cdfs = [progeny.distribution for progeny in seir.progeny]
    params = MTBPParamsSequence([MTBPParams(paramtime, rates, cdfs)])
    rng = StableRNGs.StableRNG(1234)
    states = [similar(seir.state) for _ in 1:5]
    init!(rng, states[1], seir)
    for i in Iterators.drop(eachindex(states), 1)
        states[i] .= states[i-1]
        simulate!(rng, states[i], seir, 1.0)
    end
    obs = Observations([Observation(Float64(i), [state[3]]) for (i, state) in enumerate(states)])
    kfll = loglikelihood(
        rng, params, ssm, obs, (kfapprox,), 
    )
    pfstore = ParticleStore(getstate(ssm), 256)
    rng = StableRNGs.StableRNG(1234)
    pfll = loglikelihood(
        rng, params, ssm, obs, (pfstore,), 
    )
    @test isapprox(pfll, kfll; rtol=0.07)

    switchmethod_kfonly(filteridx, iteration, filterargs, model, dt, obs) = 
        meanthresholdswitch(rng, filteridx, iteration, filterargs, model, dt, obs, 0, 1:2)
    kfswtichll = loglikelihood(
        rng, params, ssm, obs, (pfstore, kfapprox), nothing, switchmethod_kfonly,
    )
    @test kfll == kfswtichll

    switchmethod_pfonly(filteridx, iteration, filterargs, model, dt, obs) = 
        meanthresholdswitch(rng, filteridx, iteration, filterargs, model, dt, obs, typemax(Int), 1:2)
    rng = StableRNGs.StableRNG(1234)
    pfswtichll = loglikelihood(
        rng, params, ssm, obs, (pfstore, kfapprox), nothing, switchmethod_pfonly,
    )
    @test pfll == pfswtichll

    switchmethod(filteridx, iteration, filterargs, model, dt, obs) = 
        meanthresholdswitch(rng, filteridx, iteration, filterargs, model, dt, obs, 45, 1:2)
    pfkfll = loglikelihood(
        rng, params, ssm, obs, (pfstore, kfapprox), nothing, switchmethod,
    )
    @test isapprox(pfll, pfkfll; rtol=0.07)
    @test isapprox(kfll, pfkfll; rtol=0.07)
    # TODO test param switching
end