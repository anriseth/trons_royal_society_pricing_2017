using Optim
using Distributions
using LaTeXStrings
using Plots, StatPlots

Carr = [0.25, 0.5, 1.]
#γarr = [2.5e-2, 5e-2, 1e-1]
#Carr = [0.5, 1.]
γarr = [5e-2, 1e-1]
q1arr = linspace(0.5,3, 50)
q2arr = linspace(1., 5, 50)

amin = 0.; amax = 1.
xmin = 0.; xmax = 1.
K = 201
xarr = collect(linspace(xmin,xmax,K)[2:end])
xtup = (xarr,)

α = zeros(xarr)
αcecarr = zeros(α)
v = zeros(α)
vcecarr = zeros(v)

q(a,q1,q2) = q1*exp(-q2*a)
Y(a,s,γ,q1,q2) = (s-q(a,q1,q2))/(γ*q(a,q1,q2))
Z = Normal()
Φ(x) = cdf(Z,x)
F(a,s,C,γ,q1,q2) = Φ(Y(a,s,γ,q1,q2))*(a+C)*(q(a,q1,q2)-s)+a*s-(a+C)*γ*q(a,q1,q2)/(sqrt(2π))*exp(-Y(a,s,γ,q1,q2)^2/2)

# numsamples = 1000
# zarr = rand(Z,numsamples)
# α2 = zeros(α)
# Q(s,a,w,q1,q2,γ) = min(q(a,q1,q2)*w,s)
# function monteF(a,s,C,γ,q1,q2)
#     retval = 0.
#     for z in zarr
#         retval += (a+C)*Q(s,a,1.+γ*z,q1,q2,γ) - C*s
#     end
#     return retval / numsamples
# end
# Value and policy function arrays

function creatediffarrays()
    counter = 0
    diffarrays = Vector{Array{Float64,2}}(length(Carr)*length(γarr))
    for C in Carr, γ in γarr
        counter += 1
        @show counter

        diffarray = zeros(length(q2arr),length(q1arr))
        for i = 1:length(q1arr)
            q1 = q1arr[i]
            for j = 1:length(q2arr)
                q2 = q2arr[j]

                @simd for k = 1:length(xarr)
                    objective(a) = -F(a,xarr[k],C,γ,q1,q2)
                    res = optimize(objective, amin, amax)
                    α[k] = Optim.minimizer(res)
                    #v[k] = -Optim.minimum(res)
                end

                αcec(s) = min(amax,max(log(q1./s)/q2,1/q2-C,amin))
                #vcec(s) = (αcec(s)+C).*min(q(αcec(s),q1,q2),s) - C*s
                αcecarr[:] = αcec(xarr)
                # for k = 1:length(vcecarr)
                #     vcecarr[k] = vcec(xarr[k])
                # end
                #diffarray[j,i] = maximum(v-vcecarr)
                diffarray[j,i] = norm(αcecarr-α, 2)/sqrt(length(α)) # Normalised 2-norm
                #diffarrayinf[j,i] = norm(αcecarr-α, Inf) # Inf-norm
            end
        end

        diffarrays[counter] = diffarray
    end
    return diffarrays
end

diffarrays = creatediffarrays()


function diffv(v,vcecarr,α,αcecarr,C,γ,q1,q2)
    @simd for k = 1:length(xarr)
        objective(a) = -F(a,xarr[k],C,γ,q1,q2)
        res = optimize(objective, amin, amax)
        v[k] = -Optim.minimum(res)
        α[k] = Optim.minimizer(res)
    end


    αcec(s) = min(amax,max(log(q1./s)/q2,1/q2-C,amin))
    vcec(s) = (αcec(s)+C).*min(q(αcec(s),q1,q2),s) - C*s
    for k = 1:length(vcecarr)
        vcecarr[k] = vcec(xarr[k])
        αcecarr[k] = αcec(xarr[k])
    end
end



using JLD

savefiles = true
if savefiles == true
    @save "./data/bellman_det_comparison_$(now()).jld"
end


Plots.scalefontsizes(1.5)
titles = [L"$(C,\gamma)=(0.25,0.05)$" L"$(C,\gamma)=(0.25,0.1)$" L"$(C,\gamma)=(0.5,0.05)$" L"$(C,\gamma)=(0.5,0.1)$" L"$(C,\gamma)=(1,0.05)$" L"$(C,\gamma)=(1,0.1)$"]
plt = plot(q1arr, q2arr, diffarrays, st=:heatmap, margin=0mm, ratio=0.4,
           xlabel=L"$q_1$", ylabel=L"$q_2$",
           title=titles,layout=@layout [ p11 p12 ; p21 p22; p31 p32])
scatter!(plt.spmap[:p31], [q1],[q2], label="", color=:white, markersize=10)
xlims!(plt.spmap[:p31], minimum(q1arr), maximum(q1arr))
ylims!(plt.spmap[:p31], minimum(q2arr), maximum(q2arr))
xticks!(plt.spmap[:p31], [1,2,3])
