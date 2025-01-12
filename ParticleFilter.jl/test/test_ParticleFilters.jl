@testset "ParticleFilter/ParticleFilter.jl" begin
    @testset "Poisson likelihood" begin
        obsdata = [Int32[0], Int32[2], Int32[3], Int32[3], Int32[5]]
        obs = Observations([0f0; 1f0; 2f0; 3f0; 4f0; 5f0], obsdata)
        rng = Random.default_rng()
        Random.seed!(rng, 12345)
        particles_count = 40000
        rate = 1f0
        initial_dist = MTBPDiscreteDistribution([1f0], [Int32[1, 0]])
        progeny_dist = MTBPDiscreteDistribution([1f0], [Int32[0, 1]])
        poisson_process = MultitypeBranchingProcess(2, initial_dist, [progeny_dist, progeny_dist], Float32[rate, 0])
        observation_model = IdentityObservationModel(Float32, [2])
        model = StateSpaceModel(poisson_process, observation_model)
        store = ParticleStore(Float32, getstate(model), particles_count)
        test_ll = sum([logpdf(Poisson(rate), only(getvalue(obs.data[i]))-only(getvalue(obs.data[i-1]))) for i in 2:5])
        @test isapprox(test_ll, particlefilter!(rng, store, model, obs); atol=0.02)
    end
end