## ================= Setting Up ================= ##
# import Pkg; Pkg.add("JuMP") ;Pkg.add("Ipopt")
using JuMP, Ipopt;
# objective function
f(x_1,x_2) = -exp(-(x_1*x_2 - 3/2)^2 - (x_2-3/2)^2);

## ================= Unconstrained Problem ================= ##
my_first_model = Model(Ipopt.Optimizer) # Declare the problem

set_optimizer_attribute(my_first_model, "tol", 1e-9) # This is relative tol. Default is 1e-8
@variable(my_first_model, x_1 >=0)
@variable(my_first_model, x_2 >=0)
@objective(my_first_model, Min, f(x_1,x_2))
print(my_first_model)
set_silent(my_first_model);
optimize!(my_first_model);
unc_x_1 = value(x_1)
unc_x_2 = value(x_2)
unc_obj = objective_value(my_first_model)

## ================= Equality Constraints ================= ##
my_model_eqcon = Model(Ipopt.Optimizer);
@variable(my_model_eqcon, x_1 >=0, start = 1.0);
@variable(my_model_eqcon, x_2 >=0, start = 1.0);
@objective(my_model_eqcon, Min, f(x_1, x_2));
@constraint(my_model_eqcon, -x_1 + x_2^2 == 0);

print(my_model_eqcon)
optimize!(my_model_eqcon)


eqcon_x_1 = value(x_1)
eqcon_x_2 = value(x_2)

value(-x_1 + x_2^2) # We can evaluate expressions too
eqcon_obj = objective_value(my_model_eqcon)



my_model_ineqcon = Model(Ipopt.Optimizer);
@variable(my_model_ineqcon, x_1 >=0);
@variable(my_model_ineqcon, x_2 >=0);
@objective(my_model_ineqcon, Min, f(x_1, x_2));
@constraint(my_model_ineqcon, -x_1 + x_2^2 <= 0);
optimize!(my_model_ineqcon);

ineqcon_x_1 = value(x_1)
ineqcon_x_2 = value(x_2)

ineqcon_obj = objective_value(my_model_ineqcon)


# relaxing the constraint it's behaving like unconstraint problem 
my_model_ineqcon_2 = Model(Ipopt.Optimizer);
@variable(my_model_ineqcon_2, x_1 >=0);
@variable(my_model_ineqcon_2, x_2 >=0);
@objective(my_model_ineqcon_2, Min, f(x_1, x_2));
@constraint(my_model_ineqcon_2, c1, -x_1 + x_2^2 <= 1.5);
optimize!(my_model_ineqcon_2);

ineqcon_obj = objective_value(my_model_ineqcon_2)

ineqcon2_x_1 = value(x_1)
ineqcon2_x_2 = value(x_2)


##
using JuMP
using PATHSolver
using Ipopt
using Plots, LaTeXStrings
using DataFrames

## 
# Time horizon and discounting
T  = 20            # quarters (5 years)
β  = 0.85^(1/4)   # quarterly discount factor (annual rate ≈ 0.85)

# Sanctions parameters
d  = 15.0          # shadow fleet discount (USD/barrel)

# Initial conditions
K₁    = 2.0 * 90  # initial shadow fleet capacity (mb/quarter)
K_max = 12.0 * 90  # max capacity (mb/quarter)

# Demand parameters
p_0   = 80.0       # baseline price (USD/barrel)
ϵ_D   = 0.125      # short-run demand elasticity
S_ROW = 92.0 * 90  # non-Rogue supply (mb/quarter), treated as fixed
Q_W_0 = 5.4 * 90   # baseline Rogue traditional-channel exports (mb/quarter)
Q_NW_0 = 2.0 * 90  # baseline Rogue non-traditional exports (mb/quarter)
R_0   = Q_W_0 + Q_NW_0  # baseline total Rogue exports
D_0   = S_ROW + R_0      # baseline global demand

# Production cost parameters
c_0 = 17.0                    # marginal cost intercept (USD/barrel)
c̄   = (p_0 - c_0) / R_0      # marginal cost slope

# Investment cost parameter (calibrated in Cardoso, Salant, and Daubanes)
f̄ = 4.102

# Inverse demand
p_B(X) = p_0 * ((S_ROW + X) / D_0)^(-1/ϵ_D)

# Marginal production cost
C_prime(X) = c_0 + (p_0 - c_0) * (X / R_0)

# Total production cost
C_total(X) = c_0 * X + (p_0 - c_0) / (2 * R_0) * X^2

# Investment cost and marginal
F_cost(I) = f̄ / 2 * I^2
F_prime(I) = f̄ * I

# Integral of inverse demand: ∫₀ˣ p_B(q) dq
# Used in Approach 2 for the surplus objective
demand_integral(X) = p_0 * D_0 / (1 - 1/ϵ_D) * (((S_ROW + X) / D_0)^(1 - 1/ϵ_D) - (S_ROW / D_0)^(1 - 1/ϵ_D))

##
println("MC at baseline exports: $(C_prime(R_0))")
println("Baseline price: $p_0")

##

mod_mcp = Model(PATHSolver.Optimizer) #To solve MCP

# Choice variables with non-negativity bounds
@variable(mod_mcp, X[1:T] >= 0, start = R_0/2)
@variable(mod_mcp, I[1:(T-1)] >= 0, start = 0.0)
@variable(mod_mcp, alpha[1:T] >= 0, start = 0.0)

# State variable: capacity as a function of initial condition + cumulative investment
K = Vector{Any}(undef, T)
K[1] = K₁
for t in 2:T
    K[t] = K₁ + sum(I[s] for s in 1:(t-1))
end

# FOC for X (flipped sign for PATH convention):
# C'(X_t) + alpha_t - p_B(X_t) + d >= 0  ⊥  X_t >= 0
@constraint(mod_mcp, foc_X[t in 1:T],
    C_prime(X[t]) + alpha[t] - p_B(X[t]) + d ⟂ X[t]) #  \perp: ⟂

# FOC for I:
# F'(I_t) - sum_{s=t+1}^{T} β^{s-t} α_s >= 0  ⊥  I_t >= 0
@constraint(mod_mcp, foc_I[t in 1:(T-1)],
    F_prime(I[t]) - sum(β^(s-t) * alpha[s] for s in (t+1):T) ⟂ I[t])

# Complementarity on capacity constraint:
# K_t - X_t >= 0  ⊥  alpha_t >= 0
@constraint(mod_mcp, cap_constraint[t in 1:T],
    K[t] - X[t] ⟂ alpha[t])

optimize!(mod_mcp)
println(solution_summary(mod_mcp))

##
X_mcp     = value.(X)
I_mcp     = value.(I)
alpha_mcp = value.(alpha)
K_mcp = Vector{Float64}(undef, T)
K_mcp[1] = K₁
for t in 2:T
    K_mcp[t] = K₁ + sum(I_mcp[s] for s in 1:(t-1))
end

println("Shadow fleet capacity (K): ", round.(K_mcp, digits=1))
println("Sales (X):                 ", round.(X_mcp, digits=1))
println("Investment (I):            ", round.(I_mcp, digits=1))
println("Shadow price (α):          ", round.(alpha_mcp, digits=2))