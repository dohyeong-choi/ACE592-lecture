#####################################################################################
######################## Lecture 2.3 - Numerical integration ########################
#####################################################################################

## ------------------- Monte Carlo integration
# Package for drawing random numbers
using Distributions
# Define a function to do the integration for an arbitrary function
function integrate_mc(f, lower, upper, num_draws)
  # Draw from a uniform distribution
  xs = rand(Uniform(lower, upper), num_draws)
  # Expectation = mean(x)*volume
  expectation = mean(f(xs))*(upper - lower)
end

f(x) = x.^2;
integrate_mc(f, 0, 10, 1000)

## ------------------- Newton-Cotes quadrature (mid-point)
# Generic function to integrate with midpoint
function integrate_midpoint(f, a, b, N)
    # Calculate h given the interval [a,b] and N nodes
    h = (b-a)/(N-1)
    
    # Calculate the nodes starting from a+h/2 til b-h/2
    x = collect(a+h/2:h:b-h/2)
    
    # Calculate the expectation    
    expectation = sum(h*f(x))
end;

f(x) = x.^2;

println("Integrating with 5 nodes:$(integrate_midpoint(f, 0, 10, 5))")
println("Integrating with 50 nodes:$(integrate_midpoint(f, 0, 10, 50))")
println("Integrating with 100 nodes:$(integrate_midpoint(f, 0, 10, 100))")

## ------------------- Gaussian quadrature
using QuantEcon;
# Our generic function to integrate with Gaussian quadrature
function integrate_gauss(f, a, b, N)
    # This function get nodes and weights for Gauss-Legendre quadrature
    x, w = qnwlege(N, a, b)

    # Calculate expectation
    expectation = sum(w .* f(x))
end;

f(x) = x.^2;
println("Integrating with 1 node:$(integrate_gauss(f, 0, 10, 1))")
println("Integrating with 2 nodes:$(integrate_gauss(f, 0, 10, 2))")

## ------------------- Practice
f(x) = 1.0./(x.^2 .+ 1.0);

# 1. Use `quadrect` to integrate using Monte Carlo with 1000 nodes
quadrect(f, 1000, 0.0, 1.0, "R")

# 2. Use `qnwtrap` to integrate using the Trapezoid rule quadrature with 7 nodes
x, w = qnwtrap(7, 0.0, 1.0);
# Perform quadrature approximation
sum(w .* f(x))

# 3. Use `qnwlege` to integrate using Gaussian quadrature with 7 nodes
x, w = qnwlege(7, 0.0, 1.0);
# Perform quadrature approximation
sum(w .* f(x))


## ------------------- Applied example: Expected utility under quality uncertainty
# setup the problem
using QuantEcon, Plots, Distributions, LaTeXStrings;

# Parameters
α = 0.4
x₁, x₂ = 2.0, 3.0
μ, σ = 0.0, 0.5

# The integrand: g(z) * ϕ(z) where g(z) = x₁^α * (exp(z)*x₂)^(1-α) and ϕ is Normal pdf
g(z) = x₁^α * (exp.(z) .* x₂).^(1-α)

# Analytic solution via MGF of normal distribution
EU_true = x₁^α * x₂^(1-α) * exp((1-α)*μ + 0.5*(1-α)^2*σ^2)
println("True expected utility: $(round(EU_true, digits=8))")

## ------------------- Method 1: Trapezoid rule
function eu_trapezoid(n)
    a, b = μ - 4σ, μ + 4σ  # Truncated domain
    x, w = qnwtrap(n, a, b)
    # We must include the normal pdf in the integrand
    ϕ = pdf.(Normal(μ, σ), x)
    return sum(w .* g(x) .* ϕ)
end;

for n in [5, 11, 21, 51]
    eu = eu_trapezoid(n)
    err = abs(eu - EU_true)
    println("Trapezoid n=$n: EU=$(round(eu, digits=8)),  error=$(round(err, sigdigits=3))")
end
## ------------------- Method 2: Gauss-Legendre
function eu_legendre(n)
    a, b = μ - 4σ, μ + 4σ
    x, w = qnwlege(n, a, b)
    ϕ = pdf.(Normal(μ, σ), x)
    return sum(w .* g(x) .* ϕ)
end;

for n in [5, 11, 21, 51]
    eu = eu_legendre(n)
    err = abs(eu - EU_true)
    println("Legendre n=$n: EU=$(round(eu, digits=8)),  error=$(round(err, sigdigits=3))")
end

## ------------------- Method 3: Gauss-Hermite
function eu_hermite(n)
    # qnwnorm gives nodes/weights for N(μ, σ²) directly
    x, w = qnwnorm(n, μ, σ^2)
    return sum(w .* g(x))
end;

for n in [3, 5, 7, 11]
    eu = eu_hermite(n)
    err = abs(eu - EU_true)
    println("Hermite  n=$n: EU=$(round(eu, digits=8)),  error=$(round(err, sigdigits=3))")
end

## ------------------- Convergence comparison
ns_trap = 3:2:51  # odd numbers for compatibility
ns_lege = 3:2:51
ns_herm = 3:2:11

errs_trap = [abs(eu_trapezoid(n) - EU_true) for n in ns_trap]
errs_lege = [abs(eu_legendre(n) - EU_true) for n in ns_lege]
errs_herm = [abs(eu_hermite(n) - EU_true) for n in ns_herm]

plot(ns_trap, errs_trap, label="Trapezoid", lw=2, marker=:circle, ms=3, yscale=:log10,
     xlabel="Number of nodes", ylabel="Absolute error (log scale)",
     title="Convergence: Expected utility of Cobb-Douglas",
     legend=:bottomright, size=(700, 400), ylims=(1e-16, 1e0))
plot!(ns_lege, errs_lege, label="Gauss-Legendre", lw=2, marker=:square, ms=3)
plot!(ns_herm, errs_herm, label="Gauss-Hermite", lw=2, marker=:diamond, ms=3)
hline!([eps()], label="Machine epsilon", ls=:dash, color=:gray)

## ------------------- Visualizing nodes and weights: n=5
# Get nodes and weights for n=5
x_t, w_t = qnwtrap(5, μ-4σ, μ+4σ)
x_l, w_l = qnwlege(5, μ-4σ, μ+4σ)
x_h, w_h = qnwnorm(5, μ, σ^2)

# For Trapezoid and Legendre, effective weight = w * ϕ(x) (contribution to final sum)
eff_t = w_t .* pdf.(Normal(μ, σ), x_t)
eff_l = w_l .* pdf.(Normal(μ, σ), x_l)
# For Hermite, the weights already incorporate the density
eff_h = w_h

p1 = bar(x_t, eff_t, bar_width=0.03, label="Trapezoid", alpha=0.8, color=:steelblue,
         title="Effective weights (wᵢ × ϕ(xᵢ) or wᵢ for Hermite)", ylabel="Effective weight",
         xlabel=L"z = \ln\theta", legend=:topright, size=(700, 400))
bar!(x_l, eff_l, bar_width=0.03, label="Gauss-Legendre", alpha=0.8, color=:coral)
bar!(x_h, eff_h, bar_width=0.03, label="Gauss-Hermite", alpha=0.8, color=:seagreen)
## ------------------- Visualizing nodes against the integrand
# Plot the integrand g(z)*ϕ(z) and overlay the quadrature nodes
zz = range(μ - 4σ, μ + 4σ, length=200)
integrand = g(zz) .* pdf.(Normal(μ, σ), zz)

p = plot(zz, integrand, lw=2.5, label=L"g(z)\phi(z)", color=:black,
         xlabel=L"z = \ln\theta", ylabel="Integrand value",
         title="Quadrature nodes overlaid on integrand (n=5)",
         size=(700, 400), legend=:topright)

# Trapezoid nodes
scatter!(x_t, g(x_t) .* pdf.(Normal(μ, σ), x_t), ms=8, marker=:circle,
         color=:steelblue, label="Trapezoid nodes")

# Legendre nodes  
scatter!(x_l, g(x_l) .* pdf.(Normal(μ, σ), x_l), ms=8, marker=:square,
         color=:coral, label="Legendre nodes")

# Hermite nodes (plotted at g(z)*ϕ(z) for visual comparison)
scatter!(x_h, g(x_h) .* pdf.(Normal(μ, σ), x_h), ms=8, marker=:diamond,
         color=:seagreen, label="Hermite nodes")
## ------------------- Visualizing the Gauss-Hermite “discretization” 
# Compare the continuous normal pdf with the Hermite discrete approximation for n = 3, 5, 9
ns_plot = [5, 9, 21]
p_disc = plot(layout=(1,3), size=(800, 400), 
              plot_title="Gauss-Hermite discretization of N($(μ), $(σ))")

for (idx, n) in enumerate(ns_plot)
    xn, wn = qnwnorm(n, μ, σ^2)
    plot!(p_disc[idx], zz, pdf.(Normal(μ, σ), zz), lw=2, label="Normal pdf", color=:black)
    bar!(p_disc[idx], xn, wn ./ 0.05, bar_width=0.05, alpha=0.6, 
         label="Hermite (n=$n)", color=:seagreen,
         xlabel=L"z", title="n = $n", legend=:topleft)
    # Scale: divide weights by bar width to get comparable density height
end
p_disc