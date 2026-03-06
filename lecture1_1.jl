###################################################################################
######################## Lecture 1.1 - Course introduction ########################
###################################################################################

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
