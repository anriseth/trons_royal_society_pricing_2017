using DiscreteControl
using Optim, LineSearches
using Distributions
import Distributions: rand,mean
using LaTeXStrings
using Plots, StatPlots

if !isdefined(:Betashift)
    type Betashift{T} <: Distributions.Distribution{Univariate,Continuous}
        rv::Distributions.Beta{T}
        shift::T
    end

    function rand(d::Betashift)
        d.shift + rand(d.rv)
    end
    function rand(d::Betashift, dims::Int...)
        d.shift + rand(d.rv, dims...)
    end

    function mean(d::Betashift)
        d.shift + mean(d.rv)
    end
end

srand(0) # For reproducibility

q2 = 3; q1 = exp(q2*2/3)/3 # Gives same 0/1st order properties for Q as d(a)=1-a at a = 2/3
Q(a) = q1*exp(-q2*a)

f(t,x,a,w) = x-min(Q(a)*w,x)
U(t,x,a,w) = a*min(Q(a)*w,x)
C = 1.
Ubar(x) = -C*x

γ = 5e-2
T = 3
amin = 0.; amax = 1.
system = DynamicSystem1D(f,U,Ubar,T,amin,amax)
xmin = 0.; xmax = 1.
K = 201
xarr = collect(linspace(xmin,xmax,K))
xtup = (xarr,)

numsamples = 1000 # Samples per time step
betavar = 1./(8γ^2)-0.5  # Sets variance equal to γ^2

ωdist = Betashift(Beta(betavar,betavar), 0.5) # shift by 0.5 to get mean 1
#ωdist = Normal(1.,γ)

ω = rand(ωdist,numsamples,T)
# Value and policy function arrays
v = zeros(K, T+1)
α = zeros(K, T)

@time solvebellman!(v, α, system, xtup, ω)
αolfc = copy(α)


# Simulations
x0 = xarr[end]
bellman = OfflineSystemControl1D(system, x0, xarr, α)

olfc = OLFCSystem1D(system, x0, -1)
#olfcw = fill(mean(ωdist),T,1)
@time solveolfc!(αolfc,olfc,xtup,ω';
                 optimizer=OACCEL())

olfccon = OfflineSystemControl1D(system, x0, xarr, αolfc)

numsimulations = 10000

srand(1) # Reproducibility indep of bellmansamples etc
(bellmantrajectories,
 olfctrajectories) = simulatetrajectories([bellman, olfccon],
                                          ωdist, x0,
                                          numsimulations)

bellmanvals = [traj.value[end] for traj in bellmantrajectories]
olfcvals = [traj.value[end] for traj in olfctrajectories]

using JLD
savefiles = true
if savefiles == true
    @save "./data/markdown_bellman_olfc_$(now()).jld" bellmanvals olfcvals
    writecsv("./data/markdown_bellman_olfc_vals_1000.csv", [bellmanvals olfcvals])
end
