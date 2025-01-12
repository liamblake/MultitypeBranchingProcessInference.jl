@testset "KalmanFilters/kalmanfilter.jl" begin
    @testset "loglikelihood!" begin
        obs = [0.0]
        mu = [0.0]
        resid = mu-obs
        cov = [3.0;;]
        chol = cholesky(cov)
        logconst = -0.5*log(2*pi)
        ll = loglikelihood!(resid, chol, logconst)
        @test ll≈logpdf(MvNormal(mu, cov), obs)
    end
    @testset "kf loglikelihood!" begin
        prior_state = Array{Float64,1}([0;0])
        prior_state_cov = Array{Float64,2}([1 0; 0 1])

        state_transition_model = Array{Float64,2}([1 0; 0 1])
        state_transition_covariance = Array{Float64,2}([0 0; 0 0])
        
        observation_model = Array{Float64,2}([1 1])
        observation_covariance = Array{Float64,2}([1;;])
        
        kf = KalmanFilter(
            prior_state,
            prior_state_cov,
            state_transition_model,
            state_transition_covariance,
            observation_model,
            observation_covariance,
        )

        predicted_state, predicted_state_covariance = predict!(kf)
        @test predicted_state≈prior_state
        @test predicted_state_covariance≈prior_state_cov

        n = 1
        rng = StableRNGs.StableRNG(1234)
        data = [[randn(rng)] for _ in 1:n]
        ll = kalmanfilter!(kf, data)
        test_ll = 0.0
        mu = observation_model*prior_state
        cov = observation_model*prior_state_cov*observation_model' .+ observation_covariance
        for xi in data
            test_ll += logpdf(MvNormal(mu, cov), xi)
        end
        @test ll≈test_ll
    end
    @testset "kf" begin
        prior_state = Array{Float64,1}([0])
        prior_state_cov = Array{Float64,2}([1;;])

        state_transition_model = Array{Float64,2}([1;;])
        state_transition_covariance = Array{Float64,2}([0;;])
        
        observation_model = Array{Float64,2}([1;;])
        observation_covariance = Array{Float64,2}([1;;])
        
        kf = KalmanFilter(
            prior_state,
            prior_state_cov,
            state_transition_model,
            state_transition_covariance,
            observation_model,
            observation_covariance,
        )

        predicted_state, predicted_state_covariance = predict!(kf)
        @test predicted_state≈prior_state
        @test predicted_state_covariance≈prior_state_cov

        n = 8
        rng = StableRNGs.StableRNG(1234)
        data = [[randn(rng)] for _ in 1:n]
        ll = kalmanfilter!(kf, data)
        test_ll = 0.0
        mu = prior_state
        tau = 1/only(prior_state_cov)
        for i in 1:n
            test_ll += logpdf(MvNormal(mu, [1/tau + 1;;]), data[i])
            mu = (data[i]+tau*mu)/(1+tau)
            tau = tau + 1
        end
        @test ll≈test_ll
        @test kf.state_estimate≈(n*mean(data).+prior_state)./(n+1)
        @test kf.state_estimate_covariance≈[1.0./(n+1);;]
    end
end