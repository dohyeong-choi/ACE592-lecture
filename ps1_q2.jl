## Setup
U(W) = log(W); U(1) #utility function
μ_z(μ_w, σ_z) = log(μ_w) - (σ_z^2 / 2) # parameter


# setup the problem
using QuantEcon, Plots, Distributions, LaTeXStrings;

[μ_z(100, σ_z) for σ_z = 0.1:0.2:0.7]


ϕ = pdf.(Normal(), W)


## A. Monte Carlo Integration for Expected Utility

using Distributions
function integrate_mc(f, upper, lower, num_draws)
  ws = rand(Uniform(upper, lower), num_draws)
  μ_z(100, 0.1), 0.1
  expectation = mean(f.(ws))*(upper - lower)
end

f(μ_z, σ_z) = LogNormal(μ_z, σ_z);
integrate_mc(f, 0, 10, 1000)