#########################################################################################
######################## Lecture 2.2 - Numerical differentiation ########################
#########################################################################################

## ----------------------------- Finite Differenciation ----------------------------- 
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

## ----------------------------- Automatic Differenciation -----------------------------
## ForwardDiff.jl: scalars and gradients
using ForwardDiff
f(x) = x^2;
ForwardDiff.derivative(f, 2.0)

g(x) = x[1]^2 + 2*x[1] + x[2]^2 - 2*x[2] + 4*x[1]*x[2];
ForwardDiff.gradient(g, [1.0, 1.0])

## ForwardDiff.jl: Jacobians
# Note the square brackets that make array_g return an array
array_g(x) = [x[1]^2 + 2*x[1] + x[2]^2 - 2*x[2] + 4*x[1]*x[2]];
array_g([1.0, 1.0])

ForwardDiff.jacobian(array_g, [1.0, 1.0])

function G(x)
    G = similar(x)
    G[1] = x[1]^2 + 2.0*x[1] + x[2]^2 - 2.0*x[2] + 4.0*x[1]*x[2]
    G[2] = x[1]^2 - 2.0*x[1] + x[2]^2 + 2.0*x[2] - 4.0*x[1]*x[2]
    return G
end;
ForwardDiff.jacobian(G, [1.0 1.0])

## ForwardDiff.jl: Hessians
ForwardDiff.hessian(g, [1.0, 1.0])