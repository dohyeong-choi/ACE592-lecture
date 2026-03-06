#################################################################################
######################## Lecture 3.2 - Nonlinear Systems ########################
#################################################################################

## ------------------- Derivative-free methods -------------------

## Bisection method
function bisection(f, lo, up)
    tolerance = 1e-3          # tolerance for solution
    mid = (lo + up)/2         # initial guess, bisect the interval
    difference = (up - lo)/2  # initialize bound difference

    while difference > tolerance         # loop until convergence
        println("Intermediate guess: $mid")
        if sign(f(lo)) == sign(f(mid)) # if the guess has the same sign as the lower bound
            lo = mid                   # a solution is in the upper half of the interval
            mid = (lo + up)/2
        else                           # else the solution is in the lower half of the interval
            up = mid
            mid = (lo + up)/2
        end
        difference = (up - lo)/2       # update the difference 
    end
    println("The root of f(x) is $mid")
end;

f(x) = -x^(-2) + x - 1;
bisection(f, 0.2, 4.0)

# wrong interval (i.e, there’s no root in there)
bisection(f, 2.0, 4.0)

## function_iteration

function function_iteration(f, initial_guess)
    tolerance = 1e-3   # tolerance for solution
    difference = Inf   # initialize difference
    x = initial_guess  # initialize current value
    
    while abs(difference) > tolerance # loop until convergence
        println("Intermediate guess: $x")
        x_prev = x  # store previous value
        x = x_prev - f(x_prev) # calculate next guess
        difference = x - x_prev # update difference
    end
    println("The root of f(x) is $x")
end;

f(x) = -x^(-2) + x - 1;
function_iteration(f, 0.1)

## ------------------- Derivative-based methods -------------------

## Newton’s method

function newtons_method(f, f_prime, initial_guess)
  tolerance = 1e-3   # tolerance for solution
  difference = Inf   # initialize difference
  x = initial_guess  # initialize current value
  
  while abs(difference) > tolerance # loop until convergence
      println("Intermediate guess: $x")
      x_prev = x  # store previous value
      x = x_prev - f(x_prev)/f_prime(x_prev) # calculate next guess
      # ^ this is the only line that changes from function iteration
      difference = x - x_prev # update difference
  end
  println("The root of f(x) is $x")
end;

f(x) = -x^(-2) + x - 1;
f_prime(x) = 2x^(-3) + 1;
newtons_method(f, f_prime, 1.0)

## Newton’s method: a duopoly example

epsilon = 1.6; c = [0.6; 0.8]; # column vector
function f(q)
    Q = sum(q)
    F = Q^(-1/epsilon) .- (1/epsilon)Q^(-1/epsilon-1) .*q .- c .*q
end;
f([0.2; 0.2])

using LinearAlgebra
function f_jacobian(q)
    Q = sum(q)
    A = (1/epsilon)Q^(-1/epsilon-1)
    B = (1/epsilon+1)Q^(-1)
    J = -A .* [2 1; 1 2] + (A*B) .* [q q] - LinearAlgebra.Diagonal(c)
end;
f_jacobian([0.2; 0.2])

function newtons_method_multidim(f, f_jacobian, initial_guess) 
    tolerance = 1e-3   
    difference = Inf   
    x = initial_guess  
    
    while norm(difference) > tolerance # <=== Changed here
        println("Intermediate guess: $x")
        x_prev = x  
        x = x_prev - f_jacobian(x_prev)\f(x_prev) # <=== and here
        difference = x - x_prev 
    end
    println("The root of f(x) is $x")
    return x
  end;

x = newtons_method_multidim(f, f_jacobian, [0.2; 0.2])
f(x) # Let’s check our solution


## Broyden’s method: a duopoly example

using NLsolve
NLsolve.nlsolve(f, [1.0; 2.0], method=:broyden,
                xtol=:1e-8, ftol=:0.0, iterations=:1000, show_trace=:true)

NLsolve.nlsolve(f, f_jacobian, [1.0; 2.0], method=:newton,
                xtol=:1e-8, ftol=:0.0, iterations=:1000)

NLsolve.nlsolve(f, [1.0; 2.0], method=:newton,
                xtol=:1e-8, ftol=:0.0, iterations=:1000)

NLsolve.nlsolve(f, [1.0; 2.0], method=:newton, autodiff=:forward,
                xtol=:1e-8, ftol=:0.0, iterations=:1000)

## Quick detour
function f(q)
    Q = sum(q)
    F = Q^(-1/epsilon) .- (1/epsilon)Q^(-1/epsilon-1) .*q .- c .*q
end;

function f!(F, q) # more efficient
    F[1] = sum(q)^(-1/epsilon) - (1/epsilon)sum(q)^(-1/epsilon-1)*q[1] - c[1]*q[1]
    F[2] = sum(q)^(-1/epsilon) - (1/epsilon)sum(q)^(-1/epsilon-1)*q[2] - c[2]*q[2]
end;
F = zeros(2) # This allocates a 2-vector with elements equal to zero
f!(F, [0.2; 0.2]); # Note the first argument is a pre-allocated vector
F

NLsolve.nlsolve(f!, [1.0; 2.0], method=:newton,
                xtol=:1e-8, ftol=:0.0, iterations=:1000)

## ------------------- When solvers fail: convergence problems -------------------

## Failure mode 1: Cycling

f(x) = x^3 - 2*x + 2;
f_prime(x) = 3*x^2 - 2;

function newtons_method_maxiter(f, f_prime, initial_guess; maxiter=20)
  tolerance = 1e-3   # tolerance for solution
  difference = Inf   # initialize difference
  x = initial_guess  # initialize current value
  
  for k in 1:maxiter
      println("Iteration $k: x = $(round(x, digits=6))")
      x_prev = x
      x = x_prev - f(x_prev)/f_prime(x_prev)
      difference = x - x_prev
      abs(difference) < tolerance && return x
  end
  println("Did not converge after $maxiter iterations")
  return x
end;

newtons_method_maxiter(f, f_prime, 0.0; maxiter=10);

using Plots; gr()

f(x) = x^3 - 2*x + 2;
f_prime(x) = 3*x^2 - 2;

xgrid = range(-2.5, 2.0, length=300)
p = plot(xgrid, f.(xgrid), label="f(x) = x³ - 2x + 2", color=:black, linewidth=2)
hline!([0], color=:gray, linestyle=:dash, label=nothing)

# Newton steps: tangent from (0, f(0)) to (1, 0), then (1, f(1)) to (0, 0)
xs_cycle = [0.0, 1.0, 0.0, 1.0, 0.0]
colors_cycle = [:royalblue, :crimson]
for k in 1:4
    xk = xs_cycle[k]; xk1 = xs_cycle[k+1]
    plot!([xk, xk1], [f(xk), 0.0], color=colors_cycle[mod1(k,2)],
          linestyle=:dot, linewidth=1.5, label=nothing)
    plot!([xk1, xk1], [0.0, f(xk1)], color=colors_cycle[mod1(k,2)],
          linestyle=:dot, linewidth=1.5, label=nothing)
end
scatter!([0.0, 1.0], [f(0.0), f(1.0)], color=:red, markersize=6, label="Iterates")
title!("Cycling: Newton bounces between x = 0 and x = 1")
xlabel!("x"); ylabel!("f(x)")
p

## Failure mode 2: Divergence

f(x) = sign(x) * abs(x)^(1/3);
f_prime(x) = (1/3) * abs(x)^(-2/3);

newtons_method_maxiter(f, f_prime, 0.5; maxiter=8);

f(x) = sign(x) * abs(x)^(1/3);
f_prime(x) = (1/3) * abs(x)^(-2/3);

# Collect iterates
xs_div = [0.5]
for k in 1:6
    xk = xs_div[end]
    push!(xs_div, xk - f(xk)/f_prime(xk))
end

xlim = maximum(abs.(xs_div)) * 1.2
xgrid = range(-xlim, xlim, length=500)
p = plot(xgrid, f.(xgrid), label="f(x) = sign(x)|x|^(1/3)", color=:black, linewidth=2)
hline!([0], color=:gray, linestyle=:dash, label=nothing)

for k in 1:min(length(xs_div)-1, 5)
    xk = xs_div[k]; xk1 = xs_div[k+1]
    plot!([xk, xk1], [f(xk), 0.0], color=:royalblue,
          linestyle=:dot, linewidth=1.5, label=nothing, alpha=0.4+0.12*k)
    plot!([xk1, xk1], [0.0, f(xk1)], color=:royalblue,
          linestyle=:dot, linewidth=1.5, label=nothing, alpha=0.4+0.12*k)
end
scatter!(xs_div[1:6], f.(xs_div[1:6]), color=:red, markersize=5, label="Iterates")
title!("Divergence: each step doubles the distance from the root")
xlabel!("x"); ylabel!("f(x)")
p

## Failure mode 3: Slow convergence

f(x) = x^3;
f_prime(x) = 3*x^2;

newtons_method_maxiter(f, f_prime, 1.0; maxiter=50);

# Collect errors for x^3 (repeated root)
f_slow(x) = x^3; fp_slow(x) = 3*x^2;
x = 1.0; errors_slow = [abs(x)]
for k in 1:50
    x = x - f_slow(x)/fp_slow(x)
    push!(errors_slow, abs(x))
end

# Collect errors for x^3 - 1 (simple root)
f_fast(x) = x^3 - 1; fp_fast(x) = 3*x^2;
x = 2.0; errors_fast = [abs(x - 1.0)]
for k in 1:15
    x = x - f_fast(x)/fp_fast(x)
    push!(errors_fast, abs(x - 1.0))
end

p = plot(0:length(errors_slow)-1, errors_slow, marker=:circle, markersize=3,
     color=:crimson, label="x³ (repeated root, linear rate)",
     yscale=:log10, xlabel="Iteration", ylabel="Error (log scale)",
     ylims=(1e-16, 10), linewidth=1.5)
plot!(0:length(errors_fast)-1, errors_fast, marker=:diamond, markersize=3,
     color=:royalblue, label="x³ - 1 (simple root, quadratic rate)", linewidth=1.5)
title!("Convergence comparison: repeated vs simple root")
p