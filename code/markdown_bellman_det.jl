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
        d.shift + rand(d.rv, dims)
    end
end

srand(0) # For reproducibility

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

# Simulations

x0 = xarr[end]
αcec(t,s) = min(amax,max(log(q1*(T-t)./s)/q2,1/q2-C,amin))
vcec(t,s) = (T-t)*(αcec(t,s)+C).*min(Q(αcec(t,s)),s/(T-t))-C*s

vcecarr = zeros(v)
αcecarr = zeros(α)
for ti = 1:T
    αcecarr[:,ti] = αcec(ti-1,xarr)
    vcecarr[:,ti] = vcec(ti-1,xarr)
end
vcecarr[:,T+1] = vcec(T,xarr)

plot(xarr,α-αcecarr)
#==
bellman = OfflineSystemControl1D(system, x0, α)
deterministic = OfflineSystemControl1D(system,x0,αcec)
numsimulations = 1000

@time (bellmantrajectories,
       dettrajectories) = simulatetrajectories([bellman, deterministic],
                                               ωdist, x0,
                                               numsimulations)

bellmanvals = [traj.value[end] for traj in bellmantrajectories]
detvals = [traj.value[end] for traj in dettrajectories]
bellmancontrols = zeros(numsimulations, T)
detcontrols = zeros(numsimulations, T)

for ti = 1:T
    bellmancontrols[:,ti] = [traj.control[ti] for traj in bellmantrajectories]
    detcontrols[:,ti] = [traj.control[ti] for traj in dettrajectories]
end



## Plotting stuff
using JLD
savefiles = false
if savefiles == true
    @save "./data/markdown_bellman_det_$(now()).jld" bellmanvals detvals xarr α v αcecarr bellmancontrols detcontrols
    writecsv("./data/markdown_bellman_det_vals.csv", [bellmanvals detvals])
    writecsv("./data/markdown_bellman_det_policies.csv", [bellmancontrols detcontrols]')
    writecsv("./data/markdown_bellman_det_val_policy.csv", ([xarr v α αcecarr])[2:end,:])
end

#plot!([0,1,2], [quantile(detcontrols[:,1],0.1),quantile(detcontrols[:,2],0.1),quantile(detcontrols[:,3],0.1)])
==#
