x = (1e-20 + 1) - 1;
y = (1e-20) + (1 - 1);
x == y # they are different

println("Machine epsilon ϵ is $(eps(Float64))")
println("Is 1 + ϵ/2 > 1? $(1.0 + eps(Float64)/2 > 1.0)")

println("The smallest representable number larger than 1.0 is $(nextfloat(1.0))")
println("The largest representable number smaller than 1.0 is $(prevfloat(1.0))")


