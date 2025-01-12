@testset "ParticleFilter/particlestore.jl" begin
    nparticles = 3
    store = ParticleStore(Float32, Array{Float32, 1}(undef, 2), nparticles)
    function ParticleFilter.logpdf(m::MockStateSpaceModel, obs)
        return obs
    end
    function ParticleFilter.setstate!(m::MockStateSpaceModel, state)
        return m.counter += 1
    end
    ll = ParticleFilter.calcweights!(store, MockStateSpaceModel(0), 1.0)
    @test ll ≈ 1.0f0 + 3f0 - 3f0
    @test store.weights.logarithm==ones(nparticles)
    @test store.weights.values==ones(nparticles)
    @test store.weights.cumulative≈[1;2;3]

    rng = MockRNG(0)
    function ParticleFilter.randinitstate!(rng::MockRNG, out, d::MockStateSpaceModel)
        d.counter += 1
        out .= [d.counter; d.counter]
        return out
    end
    ParticleFilter.initstate!(rng, store, MockStateSpaceModel(0))
    @test store.store==[[i;i] for i in 1:nparticles]

    function ParticleFilter.simulatestate!(rng::MockRNG, particle::AbstractVector, m::MockStateSpaceModel, t)
        particle .= [m.counter; m.counter+1]
        m.counter += 2
        return
    end
    ParticleFilter.simulatestate!(rng, store, MockStateSpaceModel(nparticles+1), nothing)
    @test store.store==[[nparticles+i; nparticles+i+1] for i in 1:2:2*nparticles]

    out = zeros(3)
    x = collect(1:3)
    shift = ParticleFilter.shifted_exp!(out, x)
    @test out == exp.(x.-3)
    @test shift == 3

    stablerng = StableRNGs.StableRNG(1234)
    @test rand(stablerng, store.weights)==3
    @test rand(stablerng, store.weights)==1
    @test rand(stablerng, store.weights)==3
    @test rand(stablerng, store.weights)==2

    stablerng = StableRNGs.StableRNG(1234)
    resample!(stablerng, store)
    expected_vals = [[8;9] [4;5] [8;9]]
    @test store.resample_store == [expected_vals[:,i] for i in 1:nparticles]
    @test store.resample_store == store.store
end