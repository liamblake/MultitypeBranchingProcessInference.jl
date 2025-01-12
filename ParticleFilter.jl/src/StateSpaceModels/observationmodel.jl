abstract type AbstractObservationModel end

struct LinearGaussianObservationModel{M<:AbstractMatrix{<:Real}, V<:AbstractVector, C<:Cholesky, F<:AbstractFloat} <: AbstractObservationModel
    obs_map::M
    mu::V
    cov::C
    _residual_cache::V
    logconst::F
    function LinearGaussianObservationModel{M, V, C, F}(om::M, mu::V, c::C, r::V, lc::F) where {M<:AbstractMatrix{<:Real}, V<:AbstractVector, C<:Cholesky, F<:AbstractFloat}
        dim = size(om, 1)
        @assert dim==length(mu)
        @assert size(c)==(dim, dim)
        @assert dim==length(r)
        @assert F==eltype(mu)==eltype(c)==eltype(r)
        return new{M, V, C, F}(om, mu, c, r, lc)
    end
    LinearGaussianObservationModel(om::M, mu::V, c::C, r::V, lc::F) where {M<:AbstractMatrix{<:Real}, V<:AbstractVector, C<:Cholesky, F<:AbstractFloat} =
        LinearGaussianObservationModel{M, V, C, F}(om, mu, c, r, lc)
end

function LinearGaussianObservationModel(om::M) where {M<:AbstractMatrix{<:Real}}
    dim = size(om, 1)
    mu = zeros(eltype(om), dim)
    r = similar(mu, dim)
    cov = zeros(eltype(om), dim, dim)
    for i in 1:dim
        cov[i,i] = one(eltype(om))
    end
    chol = cholesky!(cov)
    logconst = eltype(r)(-0.5*log(2*pi))
    return LinearGaussianObservationModel(om, mu, chol, r, logconst)
end

function LinearGaussianObservationModel(om, mu, cov)
    dim = size(om, 1)
    r = similar(mu, dim)
    chol = cholesky!(cov)
    logconst = eltype(r)(-0.5*log(2*pi))
    return LinearGaussianObservationModel(om, mu, chol, r, logconst)
end

function chol_lower(a::Cholesky)
    return a.uplo === 'L' ? LowerTriangular(a.factors) : LowerTriangular(a.factors')
end

# destroys its residual input
function loglikelihood!(residual, resid_cov::Cholesky, logconst)
    # residual = (x-mu)
    # (x-mu)' * sigma^-1 * (x-mu) = (x-mu)' * (L*L')^-1 * (x-mu)
    #                             = ((x-mu)' * L'^-1) * (L^-1 * (x-mu))
    #                             = ||L^-1 * (x-mu)||^2
    # !overwrites residual
    ldiv!(chol_lower(resid_cov), residual)
    value = -sum(abs2, residual)

    # -log(det(sigma))
    value -= logdet(resid_cov)
    # * 0.5
    value *= eltype(residual)(0.5)
    # + -(k/2)*log(2π)
    value += logconst
    return value
end

function logpdf(m::LinearGaussianObservationModel, y::Union{AbstractVector{<:Number}, Tuple{<:Number}}, x)
    mul!(m.mu, m.obs_map, x)
    m._residual_cache .= m.mu
    m._residual_cache .-= y
    return loglikelihood!(m._residual_cache, m.cov, m.logconst)
end

function paramtype(m::LinearGaussianObservationModel)
    return eltype(m.mu)
end

function logpdf(m::LinearGaussianObservationModel, y, x)
    m.mu .= m.obs_map*x
    s = zero(paramtype(m))
    for yi in y
        m._residual_cache .= m.mu
        m._residual_cache .-= yi
        s += loglikelihood!(m._residual_cache, m.cov, m.logconst)
    end
    return s
end

## Identity observation model
struct IdentityObservationModel{T<:AbstractFloat, V<:AbstractVector{<:Integer}} <: AbstractObservationModel
    neg_inf_return_type::T
    observed_states_idx::V
end

function paramtype(m::IdentityObservationModel)
    return typeof(m.neg_inf_return_type)
end

function IdentityObservationModel(T::Type, observed_states_idx)
    return IdentityObservationModel(T(-Inf), observed_states_idx)
end

function logpdf(m::IdentityObservationModel, x, y)
    return x==y[m.observed_states_idx] ? zero(m.neg_inf_return_type) : m.neg_inf_return_type
end
