using Optim
using Distributions
using LaTeXStrings
using Plots, StatPlots

#Carr = [1./3., 1.]
#γarr = [5e-2, 1e-1]
Carr = [1.]
γarr = [5e-2]

q1arr = linspace(0.5,3, 20)
q2arr = linspace(1., 8, 20)

amin = 0.; amax = 1.
xmin = 0.; xmax = 1.
K = 201
xarr = collect(linspace(xmin,xmax,K)[2:end])
xtup = (xarr,)

α = zeros(xarr)
αcecarr = zeros(α)

q(a,q1,q2) = q1*exp(-q2*a)
Y(a,s,γ,q1,q2) = (s-q(a,q1,q2))/(γ*q(a,q1,q2))
Z = Normal()
Φ(x) = cdf(Z,x)
F(a,s,C,γ,q1,q2) = Φ(Y(a,s,γ,q1,q2))*(a+C)*(q(a,q1,q2)-s)+a*s

# Value and policy function arrays
function creatediffarrays()
    diffarrays = Vector{Array{Float64,2}}(length(Carr)*length(γarr))
    counter = 0
    for C in Carr, γ in γarr
        diffarray = zeros(length(q2arr),length(q1arr))

        for i = 1:length(q1arr)
            q1 = q1arr[i]

            for j = 1:length(q2arr)
                q2 = q2arr[j]

                @time @simd for k = 1:length(xarr)
                    objective(a) = -F(a,xarr[k],C,γ,q1,q2)
                    res = optimize(objective, amin, amax)
                    α[k] = Optim.minimizer(res)
                end

                αcec(s) = min(amax,max(log(q1./s)/q2,1/q2-C,amin))
                αcecarr[:] = αcec(xarr)

                diffarray[j,i] = norm(αcecarr-α)
            end
        end
        counter += 1
        diffarrays[counter] = diffarray
    end
    return diffarrays
end

diffarrays = creatediffarrays()

using JLD

savefiles = true
if savefiles == true
    @save "./data/bellman_det_comparison_$(now()).jld"
end
