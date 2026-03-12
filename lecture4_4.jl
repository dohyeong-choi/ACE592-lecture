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


