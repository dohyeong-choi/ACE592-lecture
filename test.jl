# * ============================================
# * Lecture 1.1.
# * ============================================
## We know solution is between .1 and .2
x = collect(range(.1, stop = .2, length = 10)); # generate evenly spaced grid

# generate equal length vector of qd=2
q_d = ones(size(x)).*2; 

# Define price function and Get corresponding quantity values at these prices
price(p) = p.^(-0.2)/2 .+ p.^(-0.5)/2;
y = price(x);

##
using Plots;
plot(x, [y q_d],
    linestyle = [:solid :dot],
    linewidth = [3 3],
    linecolor = [:red :blue],
    tickfontsize = 12,
    grid = :no,
    xlabel = "p",
    ylabel = "q(p)",
    label = ["q(p)" "Quantity Demanded"])

## Coding Newton's Method
# Define demand functions
demand(p) = p^(-0.2)/2 + p^(-0.5)/2 - 2;     # demanded quantity minus target
demand_grad(p) = .1*p^(-1.2) + .25*p^(-1.5); # demand gradient

function find_root_newton(demand, demand_grad)
    p = .3        # initial guess
    deltap = 1e10 # initialize stepsize

    while abs(deltap) > 1e-4
        deltap = demand(p)/demand_grad(p)
        p += deltap
        println("Intermediate guess of p = $(round(p,digits=3)).")
    end
    println("The solution is p = $(round(p,digits=3)).")
    return p
end;

## Solve for price
find_root_newton(demand, demand_grad)

## How many acres get planted with a price floor?
using Statistics
# Function iteration method to find a root
function find_root_fi(mn, variance)

    y = randn(1000)*sqrt(variance) .+ mn # draws of the random variable
    a = 1.                               # initial guess
    differ = 100.                        # initialize error
    exp_price = 1.                       # initialize expected price

    while differ > 1e-4
        a_old = a                      # save old acreage
        p = max.(1, 3 .- 2 .*a.*y)     # compute price at all distribution points
        exp_price = mean(p)            # compute expected price
        a = 1/2 + 1/2*exp_price        # get new acreage planted given new price
        differ= abs(a - a_old)         # change in acreage planted
        println("Intermediate acreage guess: $(round(a,digits=3))")
    end

    return a, exp_price
end;
##
acreage, expected_price = find_root_fi(1, 0.1);
println("The optimal number of acres to plant is $(round(acreage, digits = 3)).\nThe expected price is $(round(expected_price, digits = 3)).")
##
# * ============================================
# * Lecture 1.3.
# * ============================================

## Types: boolean
x = true;
typeof(x)
@show y = 1 > 2 # @show is a Julia macro for showing the operation.
1.0000000001 ≈ 1  # \approx<TAB>

## Types: numbers
typeof(1)
typeof(1.0)
converted_int = convert(Float32, 1.0);
typeof(converted_int)
a = 2; b = 1.0;
@show 4a + 3b^2 # In Julia, you dont need * in between numeric literals (numbers) and variables

## Types: strings
x = "Hello World!";
typeof(x)
x = 10; y = 20; println("x + y =  $(x+y).") # Use $ to interpolate a variable/expression
a = "Aww"; b = "Yeah!!!"; println(a * " " * b) # Use * to concatenate strings

## Types: arrays
a1 = [1 2 ; 3 4 ; 5 6]; typeof(a1)
a1[1,1] = 5; a1
a2 = [1 2 3 4]
a3 = [1, 2, 3, 4]

## Types: Tuples
a4 = (1, 2, 3, 4); typeof(a4)
try
  a4[1] = 5;
catch
  println("Error, can't change value of a tuple.")
end

a5 = 5, 6; typeof(a5)

a5_x, a5_y = a5;
a5_x

## Types: Named Tuples
nt = (x = 10, y = 11); typeof(nt)
nt.x
nt[:x]

## Types: Dictionary
d1 = Dict("class" => "ACE592", "grade" => 97);
typeof(d1) # the key are strings and the values are any kind of type
d1["class"]
keys_d1 = keys(d1)
values_d1 = values(d1)

## Creating a new type called FoobarNoType
struct FoobarType # This will be immutable by default
  a::Float64 # You should always declare types for the fields of a new composite type
  b::Int     # This lets the compiler generate efficient code 
  c::String  # because it knows the types of the fields when you construct a FoobarType
end
newfoo = FoobarNoType(1.3, 2, "plzzz");
typeof(newfoo)
newfoo.a

## Functions 
function F(x)
  result = sin(x^2)
  return result # it’s a good practice to make the return value explicit
end;
F(1)

F(x) = sin(x^2) # you can use shorthand notation like this
F(1)

#========================== Just-in-time compilation ==========================
When we run the function for the first time,
it may take longer because of this compilation step
This is one of the reasons why putting your code inside functions is important
Julia can optimize functions better than code outside functions
===============================================================================#

## iteration
for count in 1:10
  random_number = rand()
  if random_number > 0.2
    println("We drew a $random_number.")
  end
end

x = 1;
while x < 50
  x = x * 2
  println("After doubling, x is now equal to $x.")
end

# An Iterable is something you can loop over, like arrays
actions = ["codes well", "skips class"];
for action in actions
    println("Charlie $action")
end

# The type Iterator is a particularly convenient subset of Iterables
for key in keys(d1)
  println(d1[key])
end

## Iterating on Iterators is more memory efficient than iterating on arrays
function show_array_speed()
  m = 1
  for i = [1, 2, 3, 4, 5, 6]
    m = m*i
  end
end;

function show_iterator_speed()
  m = 1
  for i = 1:6
    m = m*i
  end
end;

using BenchmarkTools
@btime show_array_speed()
@btime show_iterator_speed()

## Neat looping
f(x) = x^2;
x_values = 0:20:100;
for (index, x) in enumerate(x_values)
  println("f(x) at value $index is $(f(x)).")
end

for x in 1:3, y in 3:-1:1
  println("$x minus $y is $(x-y)")
end

## Comprehensions
squared = [y^2 for y in 1:2:11]
squared_2 = [(y+z)^2 for y in 1:2:11, z in 1:6]

## Dot syntax: broadcasting/vectorization
g(x) = x^2;
squared_2 = g.(1:2:11) # apply function g to each element in vector [1, 3, 5, 7, 9, 11]
try
  g(1:2:11)
catch e
  println(e)
end

## Multiple dispatch
length(methods(/))
methods(/)[1:4]
##
x = (1e-20 + 1) - 1;
y = (1e-20) + (1 - 1);
x == y # they are different

println("Machine epsilon ϵ is $(eps(Float64))")
println("Is 1 + ϵ/2 > 1? $(1.0 + eps(Float64)/2 > 1.0)")

println("The smallest representable number larger than 1.0 is $(nextfloat(1.0))")
println("The largest representable number smaller than 1.0 is $(prevfloat(1.0))")

x = 1.2 - 1.1
x == 0.1


deriv_x_squred = function (x, h)

    return (((x+h)^2 - x^2) / h)
end


deriv_x_squred(2,1e-1)
deriv_x_squred(2,1e-12)
deriv_x_squred(2,1e-30)


# * ============================================
# * Lecture 2.2.
# * ============================================

## Finite Differenciation
using FiniteDifferences; # Import package
my_forward_diff  = forward_fdm(2, 1) # Forward difference operator with two points for 1st order derivative
my_central_diff  = central_fdm(2, 1) # Central difference operator
my_central_diff_6pts  = central_fdm(6, 1) # Central difference operator
my_central_diff_2nd_order  = central_fdm(3, 2) # Central difference operator for 2nd order derivative

##
f(x) = x^2;
my_forward_diff(f, 10.0);
my_central_diff(f, 10.0);

##
my_central_diff(cos, 0.0)
my_central_diff_2nd_order(cos, 0.0)

##
g(x) = x[1]^2 + 2*x[1] + x[2]^2 - 2*x[2] + 4*x[1]*x[2];
grad(my_central_diff, g, [1.0, 1.0])
jacobian(my_central_diff, g, [1.0, 1.0]) # Again, [1] gets is the first element of the tuple

##
function G(x)
    G = similar(x) # Initializes a vector with same dimensions as x (2x1)
    G[1] = x[1]^2 + 2.0*x[1] + x[2]^2 - 2.0*x[2] + 4.0*x[1]*x[2]
    G[2] = x[1]^2 - 2.0*x[1] + x[2]^2 + 2.0*x[2] - 4.0*x[1]*x[2]
    return G
end;

jacobian(my_central_diff, G, [1.0 1.0])[1] #[1] returns the first element of the output tuple (a matrix)

##
function hessian(fdm, f, x)
    # Calculate the Jacobian of function f
    f_i(x) = jacobian(fdm, f, x)[1] #[1] gets the 1st element in the tuple output, which is a vector of partial derivatives
    # Calculate the Jacobian of vector function f_i
    H = jacobian(fdm, f_i, x)[1]
    return H
end;
hessian(my_central_diff, g, [1.0, 1.0])
hessian(my_central_diff_6pts, g, [1.0, 1.0])


# * ============================================
# * Lecture 2.3.
# * ============================================

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