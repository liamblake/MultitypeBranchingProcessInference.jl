struct MTBPKalmanFilterApproximation{KF<:KalmanFilter, M<:MTBPMomentsOperator}
    kalmanfilter::KF
    moments_operator::M
end

function array_to_marray(a)
    return MArray{Tuple{size(a)...}}(a)
end

function MTBPKalmanFilterApproximation(
    ssm::StateSpaceModel{S,O}
) where {S<:MultitypeBranchingProcess, O<:LinearGaussianObservationModel}
    mtbp = ssm.stateprocess
    
    op = MTBPMomentsOperator(mtbp)
    moments!(op, mtbp)

    initial_state_mean = array_to_marray(mtbp.initial_state.first_moments)
    initial_state_cov = array_to_marray(
        mtbp.initial_state.second_moments
        - mtbp.initial_state.first_moments*mtbp.initial_state.first_moments'
    )
    
    kf_obs_model = array_to_marray(ssm.observation_model.obs_map)
    
    # wrap in Matrix because obs_model.cov is of type Cholesky
    kf_obs_cov = array_to_marray(Matrix(ssm.observation_model.cov))

    kf = KalmanFilter(
        initial_state_mean,
        initial_state_cov,
        array_to_marray(getmeanoperator(op)),
        array_to_marray(variance_covariance(op, initial_state_mean)),
        kf_obs_model,
        kf_obs_cov,
    )

    return MTBPKalmanFilterApproximation(kf, op)
end

function init!(
    f::MTBPKalmanFilterApproximation, ssm::StateSpaceModel{S,O}, 
) where {S<:MultitypeBranchingProcess, O<:LinearGaussianObservationModel}
    mtbp = ssm.stateprocess
    kf = f.kalmanfilter

    kf.state_estimate .= mtbp.initial_state.first_moments
    
    # E[ZZ'] - E[Z]E[Z']
    for i in axes(kf._state_cache, 1)
        kf._state_cache[i, 1] = -kf.state_estimate[i]
    end
    mul!(kf.state_estimate_covariance, @view(kf._state_cache[:, 1]), kf.state_estimate')
    kf.state_estimate_covariance .+= mtbp.initial_state.second_moments
    return 
end

function iterate!(
    f::MTBPKalmanFilterApproximation, 
    ssm::StateSpaceModel{S,O}, dt, observation, 
    iteration::Real, use_prev_iter_params::Bool,
    customitersetup=nothing,
) where {S<:MultitypeBranchingProcess, O<:LinearGaussianObservationModel} 
    itersetup!(f, ssm, dt, observation, iteration, use_prev_iter_params, customitersetup)

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
    model::StateSpaceModel, dt::Real, observation, 
    iteration::Real, use_prev_iter_params::Bool,
    customitersetup=nothing,
)
    isfirstiter = iteration==one(iteration)
    isfirstiter && init!(f, model)

    kf = f.kalmanfilter
    op = f.moments_operator

    if !use_prev_iter_params || isfirstiter
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

    if customitersetup !== nothing
        customitersetup(f, model, dt, observation, iteration, use_prev_iter_params)
    end
    return 
end