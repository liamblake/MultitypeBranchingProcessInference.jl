struct KalmanFilter{M<:AbstractMatrix, V<:AbstractVector, F<:AbstractFloat}
    statesize::Int
    obssize::Int

    # x(k) = Fx(k-1) + w(k)
    # y(k) = Hx(k) + v(k)
    state_transition_model::M      # F
    state_transition_covariance::M # Q = cov(w(k))
    observation_model::M           # H 
    observation_covariance::M      # R = cov(v(k))

    state_estimate::V               # x̂(k|k) = E[x(k)|y(k),...,y(0)]
    state_estimate_covariance::M    # P(k|k) = cov(x(k)|y(k),...,y(0))
    predicted_state::V              # x̂(k|k-1) = E[x(k)|y(k-1),...,y(0)]
    predicted_state_covariance::M   # P(k|k-1) = cov(x(k)|y(k-1),...,y(0))
    _residual::V                    # ỹ(k) = z(k) - Hx̂(k|k-1)
    _residual_covariance::M         # S(k) = cov(ỹ(k)|y(k-1),...,y(0))
    _gain::M                        # K(k)

    _state_cache::M
    _obs_cache::M

    logconst::F

    function KalmanFilter{M, V, F}(
        statesize::Int,
        obssize::Int,
        state_transition_model::M,
        state_transition_covariance::M,
        observation_model::M,
        observation_covariance::M,
        state_estimate::V,
        state_estimate_covariance::M,
        predicted_state::V,
        predicted_state_covariance::M,
        _residual::V,
        _residual_covariance::M,
        _gain::M,
        _state_cache::M,
        _obs_cache::M,
        logconst::F,
    ) where {M<:AbstractMatrix, V<:AbstractVector, F<:AbstractFloat} 
        @assert (statesize
                ==size(state_transition_model,1)
                ==size(state_transition_model,2)
                ==size(observation_model,2)
                ==length(state_estimate)
                ==size(state_estimate_covariance,1)
                ==size(state_estimate_covariance,2)
                ==length(predicted_state)
                ==size(predicted_state_covariance,1)
                ==size(predicted_state_covariance,2)
                ==size(_gain,1)
                ==size(_state_cache,1)
                ==size(_state_cache,2)
                ==size(_obs_cache,2))
        @assert (obssize
                    ==size(observation_model,1)
                    ==size(observation_covariance,1)
                    ==size(observation_covariance,2)
                    ==length(_residual)
                    ==size(_residual_covariance,1)
                    ==size(_residual_covariance,2)
                    ==size(_gain,2)
                    ==size(_obs_cache,1))
        return new{M, V, F}(
            statesize,
            obssize,
            state_transition_model,
            state_transition_covariance,
            observation_model,
            observation_covariance,
            state_estimate,
            state_estimate_covariance,
            predicted_state,
            predicted_state_covariance,
            _residual,
            _residual_covariance,
            _gain,
            _state_cache,
            _obs_cache,
            logconst)
    end
    KalmanFilter(
        statesize::Int,
        obssize::Int,
        state_transition_model::M,
        state_transition_covariance::M,
        observation_model::M,
        observation_covariance::M,
        state_estimate::V,
        state_estimate_covariance::M,
        predicted_state::V,
        predicted_state_covariance::M,
        _residual::V,
        _residual_covariance::M,
        _gain::M,
        _state_cache::M,
        _obs_cache::M,
        logconst::F,
    ) where {M<:AbstractMatrix, V<:AbstractVector, F<:AbstractFloat} =
        KalmanFilter{M, V, F}(
            statesize,
            obssize,
            state_transition_model,
            state_transition_covariance,
            observation_model,
            observation_covariance,
            state_estimate,
            state_estimate_covariance,
            predicted_state,
            predicted_state_covariance,
            _residual,
            _residual_covariance,
            _gain,
            _state_cache,
            _obs_cache,
            logconst,)
end

function paramtype(kf::KalmanFilter)
    return eltype(kf.observation_covariance)
end

function KalmanFilter(
    prior_state::V,
    prior_state_covariance::M,
    state_transition_model::M, 
    state_transition_covariance::M,
    observation_model::M,
    observation_covariance::M,
) where {V, M}
    obssize, statesize = size(observation_model)

    _state_cache = similar(state_transition_covariance, statesize, statesize)
    _obs_cache = similar(state_transition_covariance, obssize, statesize)

    state_estimate = similar(state_transition_model, statesize)
    state_estimate .= prior_state

    state_estimate_covariance = similar(state_transition_covariance)
    state_estimate_covariance .= prior_state_covariance

    predicted_state = similar(state_transition_model, statesize)

    predicted_state_covariance = similar(state_transition_covariance)

    _residual = similar(state_transition_model, obssize)
    _residual_covariance = similar(state_transition_covariance, obssize, obssize)
    _gain = similar(state_transition_covariance, statesize, obssize)

    logconst = eltype(state_transition_covariance)(-obssize/2*log(2*pi))

    kf = KalmanFilter(
        statesize,
        obssize,
        state_transition_model,
        state_transition_covariance,
        observation_model,
        observation_covariance,
        state_estimate,
        state_estimate_covariance,
        predicted_state,
        predicted_state_covariance,
        _residual,
        _residual_covariance,
        _gain,
        _state_cache,
        _obs_cache,
        logconst)
    
    predict!(kf)
    return kf
end

function reset!(kf::KalmanFilter, 
    prior_state=nothing,
    prior_state_covariance=nothing,
    state_transition_model=nothing, 
    state_transition_covariance=nothing,
    observation_model=nothing,
    observation_covariance=nothing,
)
    if prior_state !== nothing
        kf.state_estimate .= prior_state
    end
    if prior_state_covariance !== nothing
        kf.state_estimate_covariance .= prior_state_covariance
    end
    if state_transition_model !== nothing
        kf.state_transition_model .= state_transition_model
    end
    if state_transition_covariance !== nothing
        kf.state_transition_covariance .= state_transition_covariance
    end
    if observation_model !== nothing
        kf.observation_model .= observation_model
    end
    if observation_covariance !== nothing
        kf.observation_covariance .= prior_state
    end
    if loglikelihood !==nothing
        kf.loglikelihood .= loglikelihood
    end
    return 
end

function update!(kf::KalmanFilter, obs::AbstractVector, returnloglikelihood=true)
    # x(k) = Fx(k-1) + w(k)
    # y(k) = Hx(k) + v(k)
    
    # state_transition_model      # F
    # state_transition_covariance # Q = cov(w(k))
    # observation_model           # H 
    # observation_covariance      # R = cov(v(k))

    # state_estimate              # x̂(k|k) = E[x(k)|y(k),...,y(0)]
    # state_estimate_covariance   # P(k|k) = cov(x(k)|y(k),...,y(0))
    # predicted_state             # x̂(k|k-1) = E[x(k)|y(k-1),...,y(0)]
    # predicted_state_covariance  # P(k|k-1) = cov(x(k)|y(k-1),...,y(0))
    # _residual                   # ỹ(k) = z(k) - Hx̂(k|k-1)
    # _residual_covariance        # S(k) = cov(ỹ(k)|y(k-1),...,y(0))
    # _gain                       # K(k)

    # ỹ(k) = z(k) - Hx̂(k|k-1)
    # Hx̂(k|k-1)
    mul!(@view(kf._obs_cache[:,1]), kf.observation_model, kf.predicted_state)
    # z(k) - Hx̂(k|k-1)
    kf._residual .= obs
    kf._residual .-= kf._obs_cache[:,1]

    # S(k) = HP(k|k-1)H' + R
    # HP(k|k-1)
    mul!(kf._obs_cache, kf.observation_model, kf.predicted_state_covariance)
    # HP(k|k-1)H'
    mul!(kf._residual_covariance, kf._obs_cache, kf.observation_model')
    # HP(k|k-1)H' .+ R
    kf._residual_covariance .+= kf.observation_covariance
    # K(k) = P(k|k-1)H'/S(k)
    # P(k|k-1)H'
    mul!(kf._gain, kf.predicted_state_covariance, kf.observation_model')
    # P(k|k-1)H'/S(k)
    chol = cholesky!(kf._residual_covariance)
    rdiv!(kf._gain, chol)

    # x̂(k|k) = x̂(k|k-1) + K(k)ỹ(k)
    # K(k)ỹ(k)
    mul!(kf.state_estimate, kf._gain, kf._residual)
    # K(k)ỹ(k) + x̂(k|k-1) 
    kf.state_estimate .+= kf.predicted_state

    # P(k|k) = (I-K(k)H)P(k|k-1)
    # -K(k)
    kf._gain .*= -one(eltype(kf._gain))
    # -K(k)H
    mul!(kf._state_cache, kf._gain, kf.observation_model)
    # I-K(k)H
    for i in 1:kf.statesize
        kf._state_cache[i,i] += one(eltype(kf._state_cache))
    end
    # (I-K(k)H)P(k|k-1)
    mul!(kf.state_estimate_covariance, kf._state_cache, kf.predicted_state_covariance)
    
    if returnloglikelihood
        return loglikelihood!(kf._residual, chol, kf.logconst)
    end
    return 
end

function chol_lower(a::Cholesky)
    return a.uplo === 'L' ? LowerTriangular(a.factors) : LowerTriangular(a.factors')
end

# destroys its kf_residual input
function loglikelihood!(kf_residual, kf_resid_cov::Cholesky, logconst)
    # residual = (x-mu)
    # (x-mu)' * sigma^-1 * (x-mu) = (x-mu)' * (L*L')^-1 * (x-mu)
    #                             = ((x-mu)' * L'^-1) * (L^-1 * (x-mu))
    #                             = ||L^-1 * (x-mu)||^2
    # !overwrites kf_residual
    ldiv!(chol_lower(kf_resid_cov), kf_residual)
    value = -sum(abs2, kf_residual)

    # -log(det(sigma))
    value -= logdet(kf_resid_cov)
    # * 0.5
    value *= eltype(kf_residual)(0.5)
    # + -(k/2)*log(2π)
    value += logconst
    return value
end

function predict!(kf::KalmanFilter)
    # x(k) = Fx(k-1) + w(k)
    # y(k) = Hx(k) + v(k)
    
    # state_transition_model      # F
    # state_transition_covariance # Q = cov(w(k))
    # observation_model           # H 
    # observation_covariance      # R = cov(v(k))

    # state_estimate              # x̂(k|k) = E[x(k)|y(k),...,y(0)]
    # state_estimate_covariance   # P(k|k) = cov(x(k)|y(k),...,y(0))
    # predicted_state             # x̂(k|k-1) = E[x(k)|y(k-1),...,y(0)]
    # predicted_state_covariance  # P(k|k-1) = cov(x(k)|y(k-1),...,y(0))
    # _residual                   # ỹ(k) = z(k) - Hx̂(k|k-1)
    # _residual_covariance        # S(k) = cov(ỹ(k)|y(k-1),...,y(0))
    # _gain                       # K(k)

    # x̂(k|k-1) = Fx̂(k|k-1)
    mul!(kf.predicted_state, kf.state_transition_model, kf.state_estimate)

    # P(k|k-1) = FP(k-1|k-1)F' + Q
    # FP(k-1|k-1)
    mul!(kf._state_cache, kf.state_transition_model, kf.state_estimate_covariance)
    # FP(k-1|k-1)F'
    mul!(kf.predicted_state_covariance, kf._state_cache, kf.state_transition_model')
    # FP(k-1|k-1)F' + Q
    kf.predicted_state_covariance .+= kf.state_transition_covariance
    return kf.predicted_state, kf.predicted_state_covariance
end

function kalmanfilter!(kf::KalmanFilter, observations, returnloglikelihood=true)
    iteration = 0
    loglikelihood = zero(paramtype(kf))
    while iteration < length(observations)
        iteration += 1
        ll = update!(kf, observations[iteration])
        predict!(kf)
        if ll !== nothing
            loglikelihood += ll
        end
    end
    if returnloglikelihood
        return loglikelihood
    end
    return
end

function mean(kf::KalmanFilter)
    return kf.state_estimate
end

function mean!(out, kf::KalmanFilter)
    out .= kf.state_estimate
    return out
end
