using DiscreteControl
using Optim
using Distributions
import Distributions.rand
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
end

Carr = [0.25, 0.5, 1.]
γarr = [5e-2, 1e-1]
q1arr = linspace(0.8,3, 50)
q2arr = linspace(1.5, 5, 50)

amin = 0.; amax = 1.
xmin = 0.; xmax = 1.
K = 101
xarr = collect(linspace(xmin,xmax,K))
xtup = (xarr,)
T = 3
bellsamples = 300 # Samples per time step
numsimulations = 1000
x0 = 1.0

srand(0) # For reproducibility
q(a,q1,q2) = q1*exp(-q2*a)

function creatediffarrays()
    counter = 0
    diffarrays = Vector{Array{Float64,2}}(length(Carr)*length(γarr))

    for C in Carr, γ in γarr
        counter += 1
        @show counter

        betavar = 1./(8γ^2)-0.5  # Sets variance equal to γ^2
        Ubar(x) = -C*x
        ωdist = Betashift(Beta(betavar,betavar), 0.5) # shift by 0.5 to get mean 1

        srand(0) # For reproducibility
        ω = rand(ωdist,bellsamples,T)

        diffarray = zeros(length(q2arr),length(q1arr))
        for i = 1:length(q1arr)
            q1 = q1arr[i]
            for j = 1:length(q2arr)
                q2 = q2arr[j]

                f(t,x,a,w) = x-min(q(a,q1,q2)*w,x)
                U(t,x,a,w) = a*min(q(a,q1,q2)*w,x)

                system = DynamicSystem1D(f,U,Ubar,T,amin,amax)



                # Value and policy function arrays
                v = zeros(K, T+1)
                α = zeros(K, T)

                @time solvebellman!(v, α, system, xtup, ω)

                # Simulations
                αcec(t,s) = min(amax,max(log(q1*(T-t)./s)/q2,1/q2-C,amin))

                bellman = OfflineSystemControl1D(system, x0, xarr, α)
                deterministic = OfflineSystemControl1D(system,x0,αcec)

                srand(0) # Use same samples for each trajectory
                (bellmantrajectories,
                 dettrajectories) = simulatetrajectories([bellman, deterministic],
                                                         ωdist, x0,
                                                         numsimulations)

                bellmanvals = [traj.value[end] for traj in bellmantrajectories]
                detvals = [traj.value[end] for traj in dettrajectories]

                diffarray[j,i] = norm(detvals-bellmanvals, 2)#/sqrt(numsimulations) # L2-norm
                diffarray[j,i] = diffarray[j,i] / norm(bellmanvals,2)
            end
        end

        diffarrays[counter] = diffarray
    end
    return diffarrays
end

diffarrays = creatediffarrays()

using JLD

savefiles = true
if savefiles == true
    @save "./data/bellman_det_distance_$(now()).jld"
end

q2 = 3; q1 = exp(q2*2/3)/3
Plots.scalefontsizes(1.3)
titles = [L"$(C,\gamma)=(0.25,0.05)$" L"$(C,\gamma)=(0.25,0.1)$" L"$(C,\gamma)=(0.5,0.05)$" L"$(C,\gamma)=(0.5,0.1)$" L"$(C,\gamma)=(1,0.05)$" L"$(C,\gamma)=(1,0.1)$"]
plt = plot(q1arr, q2arr, diffarrays, st=:heatmap, margin=0mm, ratio=0.4,
           xlabel=L"$q_1$", ylabel=L"$q_2$",
           title=titles,layout=@layout [ p11 p12 ; p21 p22; p31 p32])
scatter!(plt.spmap[:p31], [q1],[q2], label="", color=:white, markersize=10)
oldxlims = xlims(plt.spmap[:p11])
oldylims = ylims(plt.spmap[:p11])
xlims!(plt.spmap[:p31], oldxlims...) # Fix scatter changing xlims,ylims
ylims!(plt.spmap[:p31], oldylims...)
xticks!(plt.spmap[:p31], [1,2,3])
yticks!(plt.spmap[:p31], [2,3,4,5])

q1s = [1+1/3, 2, 2+2/3]
q2s = [2+2/3, 4]
q1scatter = [q1s[k] for k in [2,1,3,2,1,3]]
q2scatter = [q2s[k] for k in [2,1,2,1,2,1]]

for k = 1:length(q1scatter)
    q1 = q1scatter[k]; q2 = q2scatter[k]
    splt = plt.subplots[k]
    scatter!(splt, [q1], [q2], label="",
             color = :grey, markersize = 10)
    # TODO: add scatter for each plt.subplots[k]
    xlims!(splt, oldxlims...) # Fix scatter changing xlims,ylims
    ylims!(splt, oldylims...)
    xticks!(splt, [1,2,3])
    yticks!(splt, [2,3,4,5])
end

plt
