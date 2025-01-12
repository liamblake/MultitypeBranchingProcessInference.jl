@testset "MetropolisHastings/MetropolisHastings.jl" begin
    @testset "Poisson posterior" begin
        obsdata = [Int32[0], Int32[2], Int32[3], Int32[3], Int32[5]]
        obs = Observations([0f0; 1f0; 2f0; 3f0; 4f0], obsdata)
        rng = StableRNGs.StableRNG(12345)
        particles_count = 200
        rate = 1f0
        initial_dist = MTBPDiscreteDistribution([1f0], [Int32[1, 0]])
        progeny_dist = MTBPDiscreteDistribution([1f0, 1f0], [Int32[0, 1], Int32[0, 0]])
        rates = Float32[rate, 0]
        poisson_process = MultitypeBranchingProcess(2, initial_dist, [progeny_dist, progeny_dist], rates)
        obs_model = IdentityObservationModel(Float32, [2])
        model = StateSpaceModel(poisson_process, obs_model)
        store = ParticleStore(Float32, getstate(model), particles_count)

        struct LogLikelihood
            rng::StableRNGs.StableRNG
            particles::ParticleStore
            model::StateSpaceModel
            observations::Observations
        end
        loglikelihood = LogLikelihood(rng, store, model, obs)
        function (ll::LogLikelihood)(p)
            if p[1] <= zero(p[1])
                return -Inf
            end
            setrates!(ll.model.stateprocess, [p; zero(Float32)])
            return particlefilter!(ll.rng, ll.particles, ll.model, ll.observations)
        end

        mh_config = MHConfig(
            100, # buffer size
            joinpath(@__DIR__, "data", "poisson_test.f32_array.bin"),
            10_000, # max iters
            1, # n params
            60.0, # max time (sec)
            Float32[1.], # init sample
            true, # verbose
            joinpath(@__DIR__, "data", "poisson_test.info.txt"),
            false, # adaptive
            0, # nadapt
            false, # continue_from_write_file
        )
        rm(mh_config.samples_write_file, force=true)
        rm(mh_config.info_file, force=true)

        prior_distribution = GenericIndepdendentPrior([Gamma(1f0, 1f0)])
        prior_logpdf(x, dist=prior_distribution) = logpdf(dist, x)
        proposal_distribution = MutableMvNormal(zeros(Float32, 1), 1f0*ones(Float32, 1))

        metropolis_hastings(
            rng,
            loglikelihood,
            prior_logpdf,
            proposal_distribution,
            mh_config
        )

        samples = open(joinpath(@__DIR__, "data", "poisson_test.f32_array.bin"), "r") do io
            header = MetropolisHastings.read_binary_array_file_header(io)
            @test length(header)==2
            @test header[1] == mh_config.nparams
            @test header[2] == mh_config.maxiters
            MetropolisHastings.read_binary_array_file(io, Float32)
        end
        @test size(samples, 1)==mh_config.nparams
        @test size(samples, 2)==mh_config.maxiters

        true_posterior = Gamma(
            prior_distribution.distributions[1].α+only(sum(diff(obsdata))), 
            1f0/(prior_distribution.distributions[1].θ+length(diff(obsdata))),
        )
        @test isapprox(mean(true_posterior), mean(samples); atol=0.05)
        @test isapprox(var(true_posterior), var(samples); atol=0.02)
        for x in 0:0.1:3
            @test isapprox(cdf(true_posterior, x), mean(samples.<x); atol=0.05)
        end

        ncontinue = 210
        mh_config_continue = MHConfig(
            100, # buffer size
            joinpath(@__DIR__, "data", "poisson_test.f32_array.bin"),
            ncontinue, # max iters
            1, # n params
            60.0, # max time (sec)
            Float32[1.], # init sample
            true, # verbose
            joinpath(@__DIR__, "data", "poisson_test.info.txt"),
            false, # adaptive
            0, # nadapt
            true, # continue_from_write_file
        )

        metropolis_hastings(
            rng,
            loglikelihood,
            prior_logpdf,
            proposal_distribution,
            mh_config_continue,
        )

        samples_continue = open(joinpath(@__DIR__, "data", "poisson_test.f32_array.bin"), "r") do io
            header = MetropolisHastings.read_binary_array_file_header(io)
            @test length(header)==2
            @test header[1] == mh_config_continue.nparams
            @test header[2] == (mh_config_continue.maxiters+mh_config.maxiters)
            MetropolisHastings.read_binary_array_file(io, Float32)
        end
        @test size(samples_continue, 1)==mh_config_continue.nparams
        @test size(samples_continue, 2)==(mh_config.maxiters+mh_config_continue.maxiters)
        @test samples_continue[:,1]==samples[:,1]
        @test samples_continue[:,mh_config.maxiters]==samples[:,end]
        @test samples_continue[:,mh_config.maxiters+1]!=mh_config.init_sample
        @test samples_continue[:,1:64]==samples[:,1:64]
    end
end