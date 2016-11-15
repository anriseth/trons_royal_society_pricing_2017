using DiscreteControl
using Optim
using Distributions
import Distributions.rand
using LaTeXStrings
using Plots, StatPlots

#srand(0) # For reproducibility

#Q(a) = 1.-a
q2 = 3; q1 = exp(q2*2/3)/3 # Gives same 0/1st order properties for Q as d(a)=1-a at a = 2/3
#q1 = 1.1; q2 = 2
Q(a) = q1*exp(-q2*a)
#Qhat(a) = Q1*exp(-Q2*a)
#L(a) = (1+exp(-10a))*(1+exp(10(a-1))) # Discourage extreme pricing values
#Q(a) = Qhat(a)/L(a)
#q1 = 1.1; q2 = 2
#Q(a) = q1*exp(-q2*a)

f(t,x,a,w) = x-min(Q(a)*w,x)
U(t,x,a,w) = a*min(Q(a)*w,x)
C = 1
Ubar(x) = -C*x

γ = 5e-2
T = 2
amin = 0.; amax = 1.
system = DynamicSystem1D(f,U,Ubar,T,amin,amax)
xmin = 0.; xmax = 1.
K = 301
xarr = collect(linspace(xmin,xmax,K))
xtup = (xarr,)

numsamples = 1000 # Samples per time step
betavar = 1./(8γ^2)-0.5  # Sets variance equal to γ^2
if !isdefined(:Betashift)
    type Betashift{T} <: Distributions.Distribution{Univariate,Continuous}
        rv::Distributions.Beta{T}
        shift::T
    end

    function rand(d::Betashift)
        d.shift + rand(d.rv)
    end
    function rand(d::Betashift, dims::Int...)
        d.shift + rand(d.rv, dims)
    end
end
ωdist = Betashift(Beta(betavar,betavar), 0.5) # shift by 0.5 to get mean 1
#ωdist = Normal(1.,γ)

ω = rand(ωdist,numsamples,T)
# Value and policy function arrays
v = zeros(K, T+1)
α = zeros(K, T)

@time solvebellman!(v, α, system, xtup, ω)

# Simulations

function gettrajectoryarrs(mpcnumscenariosarr, numsimulations, bellman, x0)
    bellmantrajectoriesarr = Vector{Vector}()
    mpctrajectoriesarr = Vector{Vector}()
    for mpcnumscenarios in mpcnumscenariosarr
        @show mpcnumscenarios
        mpc = MPCSystem1D(system, x0, mpcnumscenarios)

        @time (bellmantrajectories,
               mpctrajectories) = simulatetrajectories(mpc, bellman,
                                                       ωdist, x0,
                                                       numsimulations,
                                                       mpcnumscenarios == 1;
                                                       verbose = false,
                                                       optimizer = GradientDescent(),
                                                       aguessinit = fill(α[end,1],system.T))
        push!(bellmantrajectoriesarr, bellmantrajectories)
        push!(mpctrajectoriesarr, mpctrajectories)
    end
    return bellmantrajectoriesarr, mpctrajectoriesarr
end

x0 = xarr[end]
bellman = OfflineSystemControl1D(system, x0, xarr, α)
mpcnumscenariosarr = [5,20,100]
numsimulations = 1000

(bellmantrajectoriesarr,
 mpctrajectoriesarr) = gettrajectoryarrs(mpcnumscenariosarr,
                                         numsimulations, bellman, x0)

bellmanvals = [[traj.value[end] for traj in trajarr] for trajarr in bellmantrajectoriesarr]
mpcvals = [[traj.value[end] for traj in trajarr] for trajarr in mpctrajectoriesarr]

using JLD
savefiles = true
if savefiles == true
    @save "./data/markdown_bellman_mpc_$(now()).jld" bellmanvals mpcvals
    writecsv("./data/markdown_bellman_mpc_5_20_100.csv", [bellmanvals... mpcvals...])
end

#density(bellmanvals-mpcvals)

#==
@load("./data/bellman_rollout.jld")
diffarr = bellmanplotvals-mpcplotvals
writecsv("./data/markdown_bellman_mpc_1_20_100.csv", diffarr)
==#

#==
# Remember to load the file in the following way:
using DiscreteControl
using Optim
using Distributions
@load "./data/markdown_bellman_mpc_<date>.jld"
==#

#cdensity((bellmanvals-mpcvals)[1]); for i = 2:4; cdensity!((bellmanvals-mpcvals)[i]); end
#gui()


# using StatPlots
# density(bellmanvals-mpcvals,
#         title = "Bellman vs. MPC"
#         xlabel="Value of Bellman vs. MPC",
#         ylabel = "Density",
#         label="")
