##############################################################################
######################## Lecture 3.1 - Linear Systems ########################
##############################################################################

## Solving linear equations in Julia
A = [-3 2 3; -3 2 1; 3 0 0]; b = [10; 8; -3];
x = A\b # This is an optimized division, faster than inverting A (more on that later)

## Example: LU vs Cramer

using LinearAlgebra
function solve_cramer(A, b)

    dets = Vector(undef, length(b))

    for index in eachindex(b)
        B = copy(A)
        B[:, index] = b
        dets[index] = det(B)
    end

    return dets ./ det(A)

end

n = 1000;
A = rand(n, n);
b = rand(n);

using BenchmarkTools
cramer_time = @elapsed solve_cramer(A, b);
cramer_allocation = @allocated solve_cramer(A, b);
lu_time = @elapsed A\b;
lu_allocation = @allocated A\b;

println(
"Cramer's rule solved in $cramer_time seconds and used $cramer_allocation kilobytes of memory.
LU solved in $(lu_time) seconds and used $(lu_allocation) kilobytes of memory.
LU is $(round(cramer_time/lu_time, digits = 0)) times faster 
 and uses $(round(lu_allocation/cramer_allocation*100, digits = 2))%  of the memory.")

## Example: LU vs matrix inversion
 using BenchmarkTools
invers_time = @elapsed ((A^-1)*b);
invers_allocation = @allocated ((A^-1)*b);

println(
"Matrix inversion solved in $invers_time seconds and used $invers_allocation kilobytes of memory.
LU solved in $(lu_time) seconds and used $(lu_allocation) kilobytes of memory.
LU is $(round(invers_time/lu_time, digits = 2)) times faster
 and uses $(round(lu_allocation/invers_allocation*100, digits = 2))%  of the memory.")

## Numerical error blow-up: Julia example

function solve_lu(M)
    b = [1, 2]
    U = [-M^-1 1; 0 M+1]
    L = [1. 0; -M 1.]
    y = L\b
    # Round element-wise to 3 digits
    x = round.(U\y, digits = 5)
end;

true_solution(M) = round.([M/(M+1), (M+2)/(M+1)], digits = 5);
println("True solution for M=10   is approx. $(true_solution(10)), computed solution is $(solve_lu(10))");
println("True solution for M=1e10 is approx. $(true_solution(1e10)), computed solution is $(solve_lu(1e10))");
println("True solution for M=1e15 is approx. $(true_solution(1e15)), computed solution is $(solve_lu(1e15))");
println("True solution for M=1e20 is approx. $(true_solution(1e20)), computed solution is $(solve_lu(1e20))");

println("True solution for M=10 is approximately $(true_solution(10)), computed solution is $(solve_lu(10))")
println("True solution for M=1e10 is approximately $(true_solution(1e10)), computed solution is $(solve_lu(1e10))")
println("True solution for M=1e15 is approximately $(true_solution(1e15)), computed solution is $(solve_lu(1e15))")
println("True solution for M=1e20 is approximately $(true_solution(1e20)), computed solution is $(solve_lu(1e20))")


M = 1e20;
A = [-M^-1 1; 1 1];
b = [1., 2.];
julia_solution = A\b;
println("Julia's division operator is actually pretty smart though,
        true solution for M=1e20 is $(julia_solution)")

## Ill-conditioning

using LinearAlgebra;
cond([1. 1.; 1. 1.0001])
cond([1. 1.; 1. 1.00000001])
cond([1. 1.; 1. 1.000000000001])

## Convergence

dx = Inf # Start with a very large number
tol = 1e-6 # One example
iteration = 0 # Initialize value
max_iterations = 1000 # Set max of 1000 iterations
while (dx >= tol && iteration <= max_iterations)
  iteration = iteration + 1
  # Here you iterate your solutions and recalculate dx
end

