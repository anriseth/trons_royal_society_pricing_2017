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
K = 301
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

function simulatevalues(olfc::OLFCSystem1D,
                        osc::OfflineSystemControl1D,
                        ωmodel::UnivariateDistribution,
                        x0,
                        numsimulations::Int = 1000,
                        includemean::Bool = true;
                        ωtrue::UnivariateDistribution = ωmodel,
                        verbose::Bool = false,
                        optimizer::Optim.Optimizer = GradientDescent(linesearch=LineSearches.bt3!),
                        aguessinit::Vector = fill(0.5*(olfc.system.amin+olfc.system.amax), olfc.system.T))
    @assert olfc.system === osc.system
    system = olfc.system

    oscvals = zeros(numsimulations)
    olfcvals = zeros(numsimulations)

    for sim = 1:numsimulations
        trajosc = DynamicSystemTrajectory1D(system)
        trajolfc = DynamicSystemTrajectory1D(system)
        initializestate!(trajosc, x0)
        initializestate!(trajolfc, x0)
        aguess = copy(aguessinit)
        atolfc = aguess

        for t = 0:system.T-1
            numscenarios = olfc.numscenarios[t+1]
            includemean = includemean || (numscenarios == 1) # Always predict the future using the mean when only one sample
            if t > 0
                aguess = atolfc[2:end]
            end

            atosc = osc.policy(t, trajosc.state[t+1])
            atolfc = onlinedecision(olfc, trajolfc.state[t+1], t, aguess, ωmodel, includemean;
                                    verbose = verbose, optimizer = optimizer)

            # Update guess for next time
            w = rand(ωtrue)
            step!(trajosc, atosc, t, w)
            step!(trajolfc, atolfc[1], t, w)
        end
        oscvals[sim] = trajosc.value[end]
        olfcvals[sim] = trajolfc.value[end]
    end

    return oscvals, olfcvals
end

function getvalsarrs(olfcnumscenariosarr, numsimulations, bellman, x0)
    bellmanvalsarr = Vector{Vector}()
    olfcvalsarr = Vector{Vector}()
    for olfcnumscenarios in olfcnumscenariosarr
        @show olfcnumscenarios
        olfc = OLFCSystem1D(system, x0, olfcnumscenarios)

        @time (bellmanvals,
               olfcvals) = simulatevalues(olfc, bellman,
                                          ωdist, x0,
                                          numsimulations,
                                          olfcnumscenarios == 1;
                                          verbose = false,
                                          optimizer = GradientDescent(linesearch=LineSearches.bt3!),#LBFGS(linesearch=LineSearches.bt3!),
                                          aguessinit = fill(α[end,1],system.T))
        push!(bellmanvalsarr, bellmanvals)
        push!(olfcvalsarr, olfcvals)
    end
    return bellmanvalsarr, olfcvalsarr
end

# Simulations

x0 = xarr[end]
bellman = OfflineSystemControl1D(system, x0, xarr, α)
olfcnumscenariosarr = [100]
numsimulations = 1000

(bellmanvals,
 olfcvals) = getvalsarrs(olfcnumscenariosarr,
                         numsimulations, bellman, x0)

using JLD
savefiles = false
if savefiles == true
    @save "./data/markdown_bellman_olfc_$(now()).jld" bellmanvals olfcvals
    writecsv("./data/markdown_bellman_olfc_5_20_100.csv", [bellmanvals... olfcvals...])
end

#density(bellmanvals-olfcvals)

#==
@load("./data/bellman_rollout.jld")
diffarr = bellmanplotvals-olfcplotvals
writecsv("./data/markdown_bellman_olfc_20_100_500.csv", diffarr)
==#

#==
# Remember to load the file in the following way:
using DiscreteControl
using Optim
using Distributions
@load "./data/markdown_bellman_olfc_<date>.jld"
==#

#cdensity((bellmanvals-olfcvals)[1]); for i = 2:4; cdensity!((bellmanvals-olfcvals)[i]); end
#gui()


# using StatPlots
# density(bellmanvals-olfcvals,
#         title = "Bellman vs. OLFC"
#         xlabel="Value of Bellman vs. OLFC",
#         ylabel = "Density",
#         label="")
