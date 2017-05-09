using DiscreteControl
using Optim
using Distributions
import Distributions: rand, mean
using LaTeXStrings

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


if !isdefined(:DemandFun)
    type DemandFun{T}
        q1::T
        q2::T
    end
end

(d::DemandFun{T}){T}(a) = d.q1*exp(-d.q2*a)
q = DemandFun(1.0,1.0)

function main(;bellsamples = 1000, olfcsamples = -1)
    # Paramarr
    Carr = [0.25, 0.25, 0.5, 0.5, 1.0, 1.0]
    γarr = [5e-2, 1e-1,5e-2, 1e-1,5e-2, 1e-1]
    q1s = [1+1/3, 2, 2+2/3]
    q2s = [2+2/3, 4]
    q1arr = [q1s[k] for k in [2,1,3,2,1,3]]
    q2arr = [q2s[k] for k in [2,1,2,1,2,1]]

    amin = 0.; amax = 1.
    xmin = 0.; xmax = 1.
    K = 201
    xarr = collect(linspace(xmin,xmax,K))
    xtup = (xarr,)
    T = 3

    numsimulations = 10000
    x0 = 1.0

    srand(0) # For reproducibility

    function createsamples(bellvalarrs, olfcvalarrs)
        counter = 0

        for k = 1:length(Carr)
            C = Carr[k]; γ = γarr[k]; q1 = q1arr[k]; q2 = q2arr[k]
            q.q1 = q1; q.q2 = q2

            counter += 1
            @show counter

            betavar = 1./(8γ^2)-0.5  # Sets variance equal to γ^2
            Ubar(x) = -C*x

            f(t,x,a,w) = x-min(q(a)*w,x)
            U(t,x,a,w) = a*min(q(a)*w,x)

            system = DynamicSystem1D(f,U,Ubar,T,amin,amax)

            ωdist = Betashift(Beta(betavar,betavar), 0.5) # shift by 0.5 to get mean 1

            srand(0) # For reproducibility
            ω = rand(ωdist,bellsamples,T)
            # Value and policy function arrays
            v = zeros(K, T+1)
            α = zeros(K, T)
            αolfc = copy(α)

            @time solvebellman!(v, α, system, xtup, ω)

            # Simulations
            bellman = OfflineSystemControl1D(system, x0, xarr, α)

            olfc = OLFCSystem1D(system, x0, olfcsamples)
            if olfcsamples == -1
                @time solveolfc!(αolfc,olfc,xtup,ω';
                                 optimizer=GradientDescent, linesearch = LineSearches.bt3!)
            else
                wolfc = rand(ωdist, T, olfcsamples)
                @time solveolfc!(αolfc,olfc,xtup,wolfc;
                                 optimizer=GradientDescent, linesearch = LineSearches.bt3!)
            end

            olfccon = OfflineSystemControl1D(system, x0, xarr, αolfc)

            srand(1) # start with same underlying samples each time

            (bellmantrajectories,
             olfctrajectories) = simulatetrajectories([bellman, olfccon],
                                                      ωdist, x0,
                                                      numsimulations)

            bellmanvals = [traj.value[end] for traj in bellmantrajectories]
            olfcvals = [traj.value[end] for traj in olfctrajectories]


            bellvalarrs[:,k] = bellmanvals
            olfcvalarrs[:,k] = olfcvals

        end
    end

    function printtable(bellvalarrs, olfcvalarrs)
        numentries = size(bellvalarrs,2)
        diffarr = 1.0 - olfcvalarrs./bellvalarrs
        diffnorms = [norm((olfcvalarrs-bellvalarrs)[:,k],2)/norm(bellvalarrs[:,k],2)
                     for k = 1:numentries]
        vals = [quantile(diffarr[:,k], q)
                for q = [0.05,0.5,0.95], k = 1:numentries]
        vals = round(1000*[vals; diffnorms'], 1) # Show % to nearest 1/10000

        str = """
\\begin{tabular}{llllcccc}
  \$C\$ & \$\\gamma\$ & \$q_1\$ & \$q_2\$ & \$\\mathcal Q_{0.05}\$
  &Median & \$\\mathcal Q_{0.95}\$ &\$L^2\$\\\\
  \\toprule
"""
        for k = 1:numentries
            C = round(Carr[k],2); γ = round(γarr[k],2); q1 = round(q1arr[k],2); q2 = round(q2arr[k],2)
            valstrs = prod(["& \$$v\$ " for v = vals[:,k]])
            str *= "\$$C\$ & \$$γ\$ & \$$q1\$ & \$$q2\$ " *
                valstrs * "\\\\\n"
        end
        str *= """
  &&&&\$\\times 10^{-3}\$&\$\\times 10^{-3}\$&\$\\times 10^{-3}\$&\$\\times 10^{-3}\$\\\\
  \\bottomrule
\\end{tabular}
"""
    end

    bellvalarrs = Array{Float64,2}(numsimulations, length(Carr))
    olfcvalarrs = Array{Float64,2}(numsimulations, length(Carr))
    createsamples(bellvalarrs, olfcvalarrs)
    tblstr = printtable(bellvalarrs, olfcvalarrs)

    return bellvalarrs, olfcvalarrs, tblstr
end

bellvalarrs, olfcvalarrs, tblstr = main()

print(tblstr)

function plothistos(bellvalarrs, olfcvalarrs)
    Base.error("Not finished")
end
