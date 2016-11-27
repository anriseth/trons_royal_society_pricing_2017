    using Distributions
import Distributions.rand, Distributions.mean, Distributions.std,
Distributions.var, Distributions.pdf, Distributions.cdf

type Chisqshift{T<:Real} <: Distributions.Distribution{Univariate,Continuous}
    μ::T
    σ::T
    rv::Distributions.Chisq{T}
    mirror::Bool
end

function getabc(d::Chisqshift)
    c = (d.mirror == true ? -1 : 1)
    b = d.σ/std(d.rv)
    a = d.μ- c*b*mean(d.rv)
    (a,b,c)
end

Chisqshift(μ,σ,v::Int) = Chisqshift(μ,σ,Chisq(v),false)
Chisqshift(μ,σ,v::Int,mirror::Bool) = Chisqshift(μ,σ,Chisq(v),mirror)
mean(d::Chisqshift) = d.μ
std(d::Chisqshift) = d.σ
var(d::Chisqshift) = d.σ^2

function pdf{T<:Real}(d::Chisqshift{T}, x::T)
    a,b,c = getabc(d)
    pdf(d.rv, c*(x-a)/b)/b
end

function cdf{T<:Real}(d::Chisqshift{T}, x::T)
    a,b,c = getabc(d)
    if c == 1
        return cdf(d.rv, (x-a)/b)
    else
        return 1-cdf(d.rv, (a-x)/b)
    end
end

function rand(d::Chisqshift)
    a,b,c = getabc(d)
    a + c*b*rand(d.rv)
end

function rand(d::Chisqshift, n::Int)
    a,b,c = getabc(d)
    a + c*b*rand(d.rv, n)
end

#rand(d::Chisqshift, n::Int) = rand(sampler(d), n)


# type Chisqshiftsampler{T <: Real} <: Sampleable{Univariate,Continuous}
#     a::T
#     b::T
#     rv::Distributions.Chisq{T}
# end
# rand(s::Chisqshiftsampler) = s.a + s.b*rand(s.rv)
# rand(s::Chisqshiftsampler, n::Int) = s.a + s.b*rand(s.rv,n)

# function sampler(d::Chisqshift)
#     b = d.σ/std(d.rv)
#     a = d.μ-b*mean(d.rv)
#     Chisqshiftsampler(a,b,d.rv)
# end
