abstract type CovarianceFunction{F<:AbstractFloat} end

function _d2(x, y)
    size(x,1)==size(y,1) || error("Inputs must be the same length")
    T = promote_type(eltype(x), eltype(y))
    d2 = zero(T)
    for i in eachindex(x)
        d = x[i]-y[i]
        d2 += d*d
    end
    return d2
end
function _d(x, y)
    return sqrt(_d2(x,y))
end

# K(x,y) = δ(x,y)σ²
mutable struct WhiteNoiseCovarianceFunction{F<:AbstractFloat} <: CovarianceFunction{F}
    sigma2::F
    function WhiteNoiseCovarianceFunction{F}(sigma2) where F
        sigma2 >= zero(sigma2) || error("Parameter sigma2 must be non-negative")
        return new{F}(sigma2)
    end
    function WhiteNoiseCovarianceFunction(sigma2)
        T = typeof(sigma2)
        return WhiteNoiseCovarianceFunction{T}(sigma2)
    end
end
function (cf::WhiteNoiseCovarianceFunction)(x,y)
    return (iszero(_d2(x,y))) ? cf.sigma2 : zero(cf.sigma2)
end

# K(x,y) = c 
mutable struct ConstantCovarianceFunction{F<:AbstractFloat} <: CovarianceFunction{F}
    c::F
end
function (cf::ConstantCovarianceFunction)(x, y)
    return cf.c
end

# K(x,y) = σ²x⋅y 
mutable struct LinearCovarianceFunction{F<:AbstractFloat} <: CovarianceFunction{F}
    sigma2::F
    function LinearCovarianceFunction{F}(sigma2) where F
        sigma2 >= zero(sigma2) || error("Parameter sigma2 must be non-negative")
        return new{F}(sigma2)
    end
    function LinearCovarianceFunction(sigma2)
        T = typeof(sigma2)
        return LinearCovarianceFunction{T}(sigma2)
    end
end
function (cf::LinearCovarianceFunction)(x, y)
    return cf.sigma2 * dot(x,y)
end

# K(x,y) = σ²exp(-|x-y|² / (2ℓ²))
mutable struct SquaredExponentialCovarianceFunction{F<:AbstractFloat} <: CovarianceFunction{F}
    sigma2::F
    ell::F
    function SquaredExponentialCovarianceFunction{F}(sigma2, ell) where F
        sigma2 >= zero(sigma2) || error("Parameter sigma2 must be non-negative")
        ell >= zero(ell) || error("Parameter ell must be non-negative")
        return new{F}(sigma2, ell)
    end
    function SquaredExponentialCovarianceFunction(sigma2, ell)
        T = promote_type(typeof(sigma2), typeof(ell))
        return SquaredExponentialCovarianceFunction{T}(sigma2, ell)
    end
end
function (cf::SquaredExponentialCovarianceFunction)(x,y)
    return cf.sigma2 * exp(-_d2(x,y)/(2*cf.ell^2))
end

# nu = 1//2 
#   K(x,y) = σ²exp(-|x-y|/ℓ)
# nu = 3//2 
#   K(x,y) = σ²(1 + √3|x-y|/ℓ)exp(-√3|x-y|/ℓ)
# nu = 5//2 
#   K(x,y) = σ²(1 + √5|x-y|/ℓ + 5|x-x'|²/(3ℓ²))exp(-√5|x-y|/ℓ)
mutable struct MaternCovarianceFunction{NU,F<:AbstractFloat} <: CovarianceFunction{F}
    sigma2::F
    ell::F
    function MaternCovarianceFunction{NU,F}(sigma2,ell) where {NU,F}
        (NU isa Rational && NU ∈ (1//2, 3//2, 5//2)) || error("Matern implementations for nu parameter in (1//2, 3//2, 5//2)")
        sigma2 >= zero(sigma2) || error("Parameter sigma2 must be non-negative")
        ell >= zero(ell) || error("Parameter ell must be non-negative")
        return new{NU,F}(sigma2,ell)
    end
end
function MaternCovarianceFunction(n, sigma2, ell)
    T = promote_type(typeof(sigma2), typeof(ell))
    n = Rational(n)
    return MaternCovarianceFunction{n,T}(sigma2, ell)
end
function (cf::MaternCovarianceFunction{1//2,F})(x,y) where {F<:AbstractFloat}
    return cf.sigma2*exp(-_d(x,y)/cf.ell)
end
function (cf::MaternCovarianceFunction{3//2,F})(x,y) where {F<:AbstractFloat}
    sqrt3_d_div_ell = sqrt(3)*_d(x,y)/cf.ell
    return cf.sigma2*(one(eltype(x)) + sqrt3_d_div_ell)*exp(-sqrt3_d_div_ell)
end
function (cf::MaternCovarianceFunction{5//2,F})(x,y) where {F<:AbstractFloat}
    sqrt5_d_div_ell = sqrt(5)*_d(x,y)/cf.ell
    return cf.sigma2*(one(eltype(x)) + sqrt5_d_div_ell + sqrt5_d_div_ell^2/3)*exp(-sqrt5_d_div_ell)
end
const ExponentialCovarianceFunction{F} = MaternCovarianceFunction{1//2,F}
const OrnsteinUhlenbeckCovarianceFunction{F} = MaternCovarianceFunction{1//2,F}
function ExponentialCovarianceFunction(sigma2, ell)
    T = promote_type(typeof(sigma2), typeof(ell))
    return ExponentialCovarianceFunction{T}(sigma2, ell)
end
function OrnsteinUhlenbeckCovarianceFunction(sigma2, ell)
    T = promote_type(typeof(sigma2), typeof(ell))
    return OrnsteinUhlenbeckCovarianceFunction{T}(sigma2, ell)
end

# K(x,y) = σ²exp(-2sin²(|x-y|/p)/ℓ²)
mutable struct PeriodicCovarianceFunction{F<:AbstractFloat} <: CovarianceFunction{F}
    sigma2::F
    ell::F
    p::F
    function PeriodicCovarianceFunction{F}(sigma2, ell, p) where F
        sigma2 >= zero(sigma2) || error("Parameter sigma2 must be non-negative")
        ell >= zero(ell) || error("Parameter ell must be non-negative")
        p >= zero(p) || error("Parameter p must be non-negative")
        return new{F}(sigma2, ell, p)
    end
    function PeriodicCovarianceFunction(sigma2, ell, p)
        T = promote_type(typeof(sigma2), typeof(ell), typeof(p))
        return PeriodicCovarianceFunction{T}(sigma2, ell, p)
    end
end
function (cf::PeriodicCovarianceFunction)(x,y)
    d = _d(x,y)
    return cf.sigma2*exp(-2*sin(pi*d/cf.p)^2/cf.ell^2)
end

# K(x,y) = 1/(1 + |x-y|²)ᵃ
mutable struct RationalQuadraticCovarianceFunction{F<:AbstractFloat} <: CovarianceFunction{F}
    sigma2::F
    a::F
    function RationalQuadraticCovarianceFunction{F}(sigma2, a) where F
        sigma2 >= zero(sigma2) || error("Parameter sigma2 must be non-negative")
        a >= zero(a) || error("Parameter a must be non-negative")
        return new{F}(sigma2, a)
    end
    function RationalQuadraticCovarianceFunction(a) 
        T = promote_type(typeof(sigma2), typeof(a))
        return RationalQuadraticCovarianceFunction{T}(sigma2, a)
    end
end
function (cf::RationalQuadraticCovarianceFunction)(x,y)
    return cf.sigma2*(one(eltype(x)) + _d2(x,y))^-cf.a
end

struct LinearCombinationCovarianceFunction{F<:AbstractFloat} <: CovarianceFunction{F}
    weights::Vector{F}
    covfns::Vector{CovarianceFunction{F}}
end
function (cf::LinearCombinationCovarianceFunction)(x,y)
    T = promote_type(eltype(x), eltype(y))
    out = zero(T)
    for (i, covfn) in enumerate(cf.covfns)
        out += cf.weights[i]*covfn(x,y)
    end
    return out
end

# TODO: implement mutable version, perhaps
struct GaussianProcess{F<:AbstractFloat, CF<:CovarianceFunction{F}, M<:AbstractMatrix{F}, V<:AbstractVector{F}}
    x::M
    mu::V
    cov::PDMat{F,M}
    covfun::CF
    function GaussianProcess{F,CF,M,V}(x,mu,cov,covfun) where {F<:AbstractFloat, CF<:CovarianceFunction{F}, M<:AbstractMatrix{F}, V<:AbstractVector{F}}
        size(x,2) == size(cov,2) == size(cov, 1) == length(mu) || error("Parameter dimension mismatch")
        return new{F,CF,M,V}(x,mu,cov,covfun)
    end
    function GaussianProcess(x::M,mu::V,cov::PDMat{F,M},covfun::CF) where {F<:AbstractFloat, CF<:CovarianceFunction{F}, M<:AbstractMatrix{F}, V<:AbstractVector{F}}
        return GaussianProcess{F,CF,M,V}(x,mu,cov,covfun)
    end
end

function GaussianProcess(x::AbstractMatrix, mu::AbstractVector, covfun::CovarianceFunction)
    xcount = size(x,2)
    F = eltype(x)
    cov = Matrix{F}(undef, xcount, xcount)
    for i in axes(x,2)
        cov[i,i] = covfun(x[:,i], x[:,i])
        for j in Iterators.drop(axes(x,2), i)
            cov[i,j] = covfun(x[:,i], x[:,j])
            cov[j,i] = cov[i,j]
        end
    end
    cov = PDMat(cov)
    return GaussianProcess(x,mu,cov,covfun)
end

function gp_logpdf_memcache(gp, y)
    return (
        similar(y, promote_type(eltype(y), eltype(gp.cov))),
        similar(y, promote_type(eltype(y), eltype(gp.cov)))
    )
end

function chol_lower(a)
    return a.uplo === 'L' ? LowerTriangular(a.factors) : LowerTriangular(a.factors')
end

function Distributions.logpdf(gp::GaussianProcess, y, cache=gp_logpdf_memcache(gp,y))
    N = length(y)
    if N != length(gp.mu)
        error("Data vector y must have the same length as gp.x")
    end
    # y'C⁻¹y = y'(LL')⁻¹y = (y'(L')⁻¹)(L⁻¹y) = ||(L⁻¹y)||
    cache[1] .= y .- gp.mu
    lower = chol_lower(gp.cov.chol)
    ldiv!(cache[2], lower, cache[1])
    quad = zero(eltype(cache[2]))
    for ci in cache[2]
        quad += ci^2
    end
    F = eltype(gp.x)
    halfsumlogdet = zero(eltype(gp.cov))
    for i in axes(lower,1)
        halfsumlogdet += log(lower[i,i])
    end
    return -F(0.5)*(N*(log(F(2*pi))) + quad) - halfsumlogdet
 end

function Random.rand!(rng::AbstractRNG, gp::GaussianProcess, out::AbstractVector, cache=similar(out))
    randn!(rng, cache)
    mul!(out, gp.cov.chol.L, cache)
    out .+= gp.mu
    return out
end