@testset "ParticleFilter/multitypebranchingprocess.jl" begin
    @testset "MTBP Dicrete Distribution one event" begin
        d = MTBPDiscreteDistribution([1.0], [[1]])
        rng = StableRNGs.StableRNG(1234)
        # test until we get both outcomes
        @test MultitypeBranchingProcesses.rand_idx(rng, d.distribution)==1

        StableRNGs.seed!(rng, 1234)
        # should match the sequence above
        @test rand(rng, d)==[1]
    end

    @testset "defaults" begin
        d = MTBPDiscreteDistribution([0.5;1.0], [[1],[2]])
        rng = StableRNGs.StableRNG(1234)
        # test until we get both outcomes
        @test MultitypeBranchingProcesses.rand_idx(rng, d.distribution)==2
        @test MultitypeBranchingProcesses.rand_idx(rng, d.distribution)==2
        @test MultitypeBranchingProcesses.rand_idx(rng, d.distribution)==2
        @test MultitypeBranchingProcesses.rand_idx(rng, d.distribution)==2
        @test MultitypeBranchingProcesses.rand_idx(rng, d.distribution)==1

        StableRNGs.seed!(rng, 1234)
        # should match the sequence above
        @test rand(rng, d)==[2]
        @test rand(rng, d)==[2]
        @test rand(rng, d)==[2]
        @test rand(rng, d)==[2]
        @test rand(rng, d)==[1]

        rates = [1.5]
        bp = MultitypeBranchingProcess(1, d, [d], rates)
        state = [2]
        @test MultitypeBranchingProcesses.transition_event_params!(bp, state)==only(rates)*2
        simulate!(rng, state, bp, 0)
        @test state==[2] # state must be unchanged
        simulate!(rng, state, bp, 1.0)
        @test state==[7]

        p = getntypes(bp)
        mu = 0.0
        cov = 0.0
        nsamples = 50_000
        state0 = [1]
        dt = 0.1
        for i in 1:nsamples
            state .= state0
            simulate!(rng, state, bp, dt)
            mu += only(state)
            cov += only(state)^2
        end
        mu /= nsamples
        cov = cov/nsamples - mu^2
        moments = MTBPMomentsOperator(bp)
        moments!(moments, bp, dt)
        @test isapprox(moments.generator[end-p+1:end,end-p+1:end], [mu]; atol=0.01)
        @test isapprox(moments.generator[1:end-p,end-p+1:end], [cov]; atol=0.04)
        m = zeros(1)
        MultitypeBranchingProcesses.firstmoment!(m, d)
        @test moments.generator[end-p+1:end,end-p+1:end]≈[exp(only(rates)*only(m)*dt)]

        for dt in 0.5:0.5:1.5
            moments!(moments, bp, dt)
            @test moments.generator[end-p+1:end,end-p+1:end]≈[exp(only(rates)*only(m)*dt)]
        end
    end
    
    @testset "f32" begin
        d = MTBPDiscreteDistribution([0.5f0;1.0f0], [[Int16(1)],[Int16(2)]])
        rng = StableRNGs.StableRNG(1234)
        # test until we get both outcomes
        @test MultitypeBranchingProcesses.rand_idx(rng, d.distribution)==2
        @test MultitypeBranchingProcesses.rand_idx(rng, d.distribution)==1

        StableRNGs.seed!(rng, 1234)
        # should match the sequence above
        @test rand(rng, d)==[2]
        @test rand(rng, d)==[1]

        rates = [1.5f0]
        bp = MultitypeBranchingProcess(1, d, [d], rates)

        state = [Int16(2)]
        @test MultitypeBranchingProcesses.transition_event_params!(bp, state)===only(rates)*2
        simulate!(rng, state, bp, 0f0)
        @test state==[Int16(2)]
        simulate!(rng, state, bp, 1.0f0)
        @test state==[Int16(20)]

        p = getntypes(bp)
        mu = 0.0
        cov = 0.0
        nsamples = 50_000
        state0 = [Int16(1)]
        dt = 0.1f0
        for i in 1:nsamples
            state .= state0
            simulate!(rng, state, bp, dt)
            mu += only(state)
            cov += only(state)^2
        end
        mu /= nsamples
        cov = cov/nsamples - mu^2
        moments = MTBPMomentsOperator(bp)
        moments!(moments, bp, dt)
        @test isapprox(moments.generator[end-p+1:end,end-p+1:end], [mu]; atol=0.01)
        @test isapprox(moments.generator[1:end-p,end-p+1:end], [cov]; atol=0.04)
        m = zeros(1)
        MultitypeBranchingProcesses.firstmoment!(m, d)
        @test moments.generator[end-p+1:end,end-p+1:end]≈[exp(only(rates)*only(m)*dt)]

        for dt in 0.5f0:0.5f0:1.5f0
            moments!(moments, bp, dt)
            @test moments.generator[end-p+1:end,end-p+1:end]≈[exp(only(rates)*only(m)*dt)]
        end
    end

    @testset "StaticArrays" begin
        d = MTBPDiscreteDistribution(
            @SArray([0.5f0;1.0f0]), 
            @SArray([@SArray([Int16(1)]);@SArray([Int16(2)])]),
        )
        rng = StableRNGs.StableRNG(1234)
        # test until we get both outcomes
        @test MultitypeBranchingProcesses.rand_idx(rng, d.distribution)==2
        @test MultitypeBranchingProcesses.rand_idx(rng, d.distribution)==1

        StableRNGs.seed!(rng, 1234)
        # should match the sequence above
        @test rand(rng, d)==[2]
        @test rand(rng, d)==[1]

        rates = @SArray([1.5f0])
        bp = MultitypeBranchingProcess(1, d, [d], rates)

        state = @MArray([Int16(2)])
        @test MultitypeBranchingProcesses.transition_event_params!(bp, state)===only(rates)*2
        simulate!(rng, state, bp, 0f0)
        @test state==[Int16(2)]
        simulate!(rng, state, bp, 1.0f0)
        @test state==[Int16(20)]

        p = getntypes(bp)
        mu = 0.0
        cov = 0.0
        nsamples = 50_000
        state0 = @MArray([Int16(1)])
        dt = 0.1f0
        for i in 1:nsamples
            state .= state0
            simulate!(rng, state, bp, dt)
            mu += only(state)
            cov += only(state)^2
        end
        mu /= nsamples
        cov = cov/nsamples - mu^2
        moments = MTBPMomentsOperator(bp)
        moments!(moments, bp, dt)
        @test isapprox(moments.generator[end-p+1:end,end-p+1:end], [mu]; atol=0.01)
        @test isapprox(moments.generator[1:end-p,end-p+1:end], [cov]; atol=0.04)
        m = zeros(1)
        MultitypeBranchingProcesses.firstmoment!(m, d)
        @test moments.generator[end-p+1:end,end-p+1:end]≈[exp(only(rates)*only(m)*dt)]

        for dt in 0.5f0:0.5f0:1.5f0
            moments!(moments, bp, dt)
            @test moments.generator[end-p+1:end,end-p+1:end]≈[exp(only(rates)*only(m)*dt)]
        end
    end
    @testset "poisson" begin
        rate = 0.1f0
        initial_dist = MTBPDiscreteDistribution([1f0], [Int32[1, 0]])
        progeny_dist = MTBPDiscreteDistribution([1f0], [Int32[0, 1]])
        secondmoments = zeros(2,2)
        MultitypeBranchingProcesses.secondmoment!(secondmoments, progeny_dist)
        @test secondmoments[1,2]==secondmoments[2,1]
        poisson_process = MultitypeBranchingProcess(2, initial_dist, [progeny_dist, progeny_dist], Float32[rate, 0])
        p = getntypes(poisson_process)
        for dt in 0.0f0:0.5f0:1.5f0
            moments = MTBPMomentsOperator(poisson_process)
            moments!(moments, poisson_process, dt)
            init!(poisson_process)
            bpmean = mean(moments, poisson_process.state)
            cov = zeros(paramtype(poisson_process), getntypes(poisson_process), getntypes(poisson_process))
            variance_covariance!(cov, moments, poisson_process.state)
            p = getntypes(poisson_process)

            @test moments.generator[end-p+1,end-p+1]≈1
            @test moments.generator[end-p+2,end-p+1]≈dt*rate
            @test moments.generator[end-p+1,end-p+2]≈0
            @test moments.generator[end-p+2,end-p+2]≈1
            @test bpmean≈[1;dt*rate]

            @test cov≈[0 0; 0 dt*rate]
            @test moments.generator[1:end-p,end]≈zeros(4)
        end
    end
    @testset "complex model" begin
        rng = StableRNGs.StableRNG(1234)
        d = MTBPDiscreteDistribution[]
        for i in 1:4
            cdf = rand(rng, 8)
            cdf = cumsum(cdf)
            cdf ./= cdf[end]
            push!(d, 
                MTBPDiscreteDistribution(
                    cdf, 
                    [round.(Int, 4*rand(rng, 4)) for _ in 1:8],
                )
            )
        end
        init_d = MTBPDiscreteDistribution([1.0], [round.(Int, 4*rand(rng, 4))])
        rates = rand(rng, 4)
        bp = MultitypeBranchingProcess(4, init_d, d, rates)

        p = getntypes(bp)
        mu = zeros(Int, 4)
        cov = zeros(Int, 4, 4)
        outer = zeros(Int, 4, 4)
        nsamples = 10_000
        dt = 0.1f0
        for i in 1:nsamples
            init!(rng, bp)
            simulate!(rng, bp, dt)
            mu .+= bp.state
            outer .= bp.state
            outer .*= bp.state'
            cov .+= outer
        end
        mu = mu./nsamples
        cov = cov./nsamples - (mu*mu')
        moments = MTBPMomentsOperator(bp)
        moments!(moments, bp, dt)
        init!(rng, bp)
        bpmean = zeros(paramtype(bp), length(bp.state))
        bpcov = zeros(paramtype(bp), length(bp.state), length(bp.state))
        mean!(bpmean, moments, bp.state)
        variance_covariance!(bpcov, moments, bp.state)
        @test all(isapprox.(bpmean-mu, zeros(4); atol=0.05))
        @test all(isapprox.(bpcov-cov, zeros(4,4); atol=0.2))
    end
end
