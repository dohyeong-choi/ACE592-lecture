## Setup
U(W) = log(W); U(1) #utility function
μ(σ::Float64) = log(100) - ( (σ^2) / 2) # parameter

## A. Monte Carlo Integration for Expected Utility

using Distributions
function integrate_mc(μ, σ, num_draws)
  zs = rand(Normal(μ, σ), num_draws)
  expectation = mean(zs)
end

mc = [];
for σ in 0.1:0.2:0.7, N in [100, 1000, 10000, 100000]
  app = integrate_mc(μ(σ), σ, N)
  push!(mc, (σ, N, app))
end;
mc

#=
The numerical integration using Monte Carlo simulation has an advantage that it is pretty simple to understand.
However, it has a limitation that its estimate stabilizes slower and it shows even fluctuation when increasing the number of random draws.
That is undesirable behavior.   
=#

ce_from_mc = []
for i in 4:4:16
  a = mc[i][1]
  b = exp(mc[i][3])
  push!(ce_from_mc, (a,b))
end;
ce_from_mc
## B. Quadrature Method Integration for Expected Utility

function eu_hermite(n, μ, σ)
    z, w = qnwnorm(n, μ, σ^2)
    return sum(w .* z)
end;

qm = []
for σ in 0.1:0.2:0.7, n in [2, 5, 7, 11]
    eu = eu_hermite(n, μ(σ), σ)
    push!(qm, (σ, n, eu))
end
qm

#=
The quadrature method has an advantage that it can estimate integrations well even for general functions.
And it is stabilizing much faster than Monte Carlo method. With only two nodes, it returns almost accurate estimates. 
=#

ce_from_qm = []
for i in 3:3:12
  a = qm[i][1]
  b = exp(qm[i][3])
  push!(ce_from_qm, (a,b))
end;
ce_from_qm

## C. Analysis and Discussion

#=
As risk σ_z increases, the corresponding certainty equivalent decreases.
This is because when an agent is risk averse, they assign greater value on an asset with lower risks if the expected return remains same.
=#

#=
Answers for the second bullet point are at the end of the corresponding codes above.
=#