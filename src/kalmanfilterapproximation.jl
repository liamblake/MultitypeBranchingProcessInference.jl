struct MTBPKalmanFilterApproximation{KF<:KalmanFilter, M<:MTBPMomentsOperator}
    kalmanfilter::KF
    moments_operator::M
end

function MTBPKalmanFilterApproximation(
    ssm::StateSpaceModel{S,O}
) where {S<:MultitypeBranchingProcess, O<:LinearGaussianObservationModel}
    mtbp = ssm.stateprocess
    
    op = MTBPMomentsOperator(mtbp)
    moments!(op, mtbp)

    initial_state_mean = mtbp.initial_state.first_moments
    initial_state_cov = (
        mtbp.initial_state.second_moments
        - mtbp.initial_state.first_moments*mtbp.initial_state.first_moments'
    )
    
    kf_obs_model = ssm.observation_model.obs_map
    
    # wrap in Matrix because obs_model.cov is of type Cholesky
    kf_obs_cov = Matrix(ssm.observation_model.cov)

    kf = KalmanFilter(
        initial_state_mean,
        initial_state_cov,
        getmeanoperator(op),
        variance_covariance(op, initial_state_mean),
        kf_obs_model,
        kf_obs_cov,
    )

    return MTBPKalmanFilterApproximation(kf, op)
end

function init!(
    f::MTBPKalmanFilterApproximation, ssm::StateSpaceModel{S,O}, observation, 
) where {S<:MultitypeBranchingProcess, O<:LinearGaussianObservationModel}
    mtbp = ssm.stateprocess
    kf = f.kalmanfilter

    kf.predicted_state .= mtbp.initial_state.first_moments
    
    # E[ZZ'] - E[Z]E[Z']
    for i in axes(kf._state_cache, 1)
        kf._state_cache[i, 1] = -kf.predicted_state[i]
    end
    mul!(kf.predicted_state_covariance, @view(kf._state_cache[:, 1]), kf.predicted_state')
    kf.predicted_state_covariance .+= mtbp.initial_state.second_moments

    ll = update!(kf, observation, true)

    for x in kf.state_estimate
        if x < zero(x)
            return -Inf
        end
    end
    return ll 
end

function iterate!(
    f::MTBPKalmanFilterApproximation, ssm::StateSpaceModel{S,O}, dt, observation,
) where {S<:MultitypeBranchingProcess, O<:LinearGaussianObservationModel} 
    kf = f.kalmanfilter

    predict!(kf)

    ll = update!(kf, observation, true)
    
    for x in kf.state_estimate
        if x < zero(x)
            return -Inf
        end
    end
    return ll 
end

function itersetup!(
    f::MTBPKalmanFilterApproximation,
    model::StateSpaceModel, dt::Union{Real,Nothing}, observation, 
    iteration::Real, use_prev_iter_params::Bool,
)
    kf = f.kalmanfilter
    op = f.moments_operator

    if !use_prev_iter_params
        mtbp = model.stateprocess
        moments!(op, mtbp, dt)
    end

    kf.state_transition_model .= getmeanoperator(op)

    variance_covariance!(
        kf.state_transition_covariance, op, kf.state_estimate
    )
    # ensure symmetric
    kf._state_cache .= kf.state_transition_covariance
    kf.state_transition_covariance .+= kf._state_cache'
    kf.state_transition_covariance ./= 2
    return 
end
