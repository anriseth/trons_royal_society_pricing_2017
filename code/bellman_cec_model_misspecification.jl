using DiscreteControl
using Optim
using LaTeXStrings
using Plots, StatPlots

if !isdefined(:Chisqshift)
    include("customdistributions.jl")
end

srand(0) # For reproducibility

Q2 = 3; Q1 = exp(Q2*2/3)/3 # Gives same 0/1st order properties for Q as d(a)=1-a at a = 2/3
Q(a) = Q1*exp(-Q2*a)

f(t,x,a,w) = x-min(Q(a)*w,x)
U(t,x,a,w) = a*min(Q(a)*w,x)
C = 1.
Ubar(x) = -C*x

γmodel = 5e-2
T = 3
amin = 0.; amax = 1.
system = DynamicSystem1D(f,U,Ubar,T,amin,amax)
xmin = 0.; xmax = 1.
K = 301
xarr = collect(linspace(xmin,xmax,K))
xtup = (xarr,)

numsamples = 1000 # Samples per time step
ωmodel = Normal(1.,γmodel) # Basically equivalent to the Beta-shifted distribution

w = rand(ωmodel, numsamples, T)

# Value and policy function arrays
v = zeros(K, T+1)
α = zeros(K, T)

@time solvebellman!(v, α, system, xtup, w)

# Simulations
x0 = xarr[end]
αcec(t,s) = min(amax,max(log(q1*(T-t)./s)/q2,1/q2-C,amin))

bellman = OfflineSystemControl1D(system, x0, xarr, α)
deterministic = OfflineSystemControl1D(system,x0,αcec)
numsimulations = 1000

function misstrajectories(ωtrue)
    (bellmantrajectories,
     cectrajectories) = simulatetrajectories([bellman, deterministic],
                                             ωmodel, x0,
                                             numsimulations;
                                             ωtrue = ωtrue)
end

ωmissmean = Normal(0.95mean(ωmodel), std(ωmodel))
(bellmantrajectoriesmissmean,
 cectrajectoriesmissmean) = misstrajectories(ωmissmean)
bellmanvalsmissmean = [traj.value[end] for traj in bellmantrajectoriesmissmean]
cecvalsmissmean = [traj.value[end] for traj in cectrajectoriesmissmean]

ωmisstd = Normal(mean(ωmodel), 2*std(ωmodel))
(bellmantrajectoriesmisstd,
 cectrajectoriesmisstd) = misstrajectories(ωmisstd)
bellmanvalsmisstd = [traj.value[end] for traj in bellmantrajectoriesmisstd]
cecvalsmisstd = [traj.value[end] for traj in cectrajectoriesmisstd]

chi2degs = 5
ωchi2 = Chisqshift(mean(ωmodel), std(ωmodel), chi2degs)
(bellmantrajectorieschi2,
 cectrajectorieschi2) = misstrajectories(ωchi2)
bellmanvalschi2 = [traj.value[end] for traj in bellmantrajectorieschi2]
cecvalschi2 = [traj.value[end] for traj in cectrajectorieschi2]

ωchi2rev = Chisqshift(mean(ωmodel), std(ωmodel), chi2degs, true)
(bellmantrajectorieschi2rev,
 cectrajectorieschi2rev) = misstrajectories(ωchi2rev)
bellmanvalschi2rev = [traj.value[end] for traj in bellmantrajectorieschi2rev]
cecvalschi2rev = [traj.value[end] for traj in cectrajectorieschi2rev]

using JLD
savefiles = true
if savefiles == true
    savearr = [bellmanvalsmissmean bellmanvalsmisstd bellmanvalschi2 bellmanvalschi2rev cecvalsmissmean cecvalsmisstd cecvalschi2 cecvalschi2rev]
    @save "./data/markdown_bellman_cec_model_misspecification_$(now()).jld" bellmanvalsmissmean bellmanvalsmisstd bellmanvalschi2 bellmanvalschi2rev cecvalsmissmean cecvalsmisstd cecvalschi2 cecvalschi2rev
    writecsv("./data/markdown_bellman_cec_model_misspecification.csv", savearr)
end

#pltmisstd = histogram(bellmanvalsmisstd-cecvalsmisstd)
#pltmissmean = histogram(bellmanvalsmissmean-cecvalsmissmean)


#plot(pltmissmean, pltmisstd)
#==
# Remember to load the file in the following way:
using DiscreteControl
using Optim
using Distributions
@load "./data/markdown_bellman_cec_<date>.jld"
==#
