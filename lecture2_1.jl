####################################################################################
######################## Lecture 2.1 - Numerical arithmetic ########################
####################################################################################


function hello()
  print("hello, world\n")
end;

hello()

## Simple arithmetic

x = (1e-20 + 1) - 1;
y = (1e-20) + (1 - 1);
x == y

## Welcome to the world of finite precision
x
y

println(typeof(5.0))
println(typeof(5))


## The limits of computers

## Fact 1: There exists a machine epsilon
println("Machine epsilon ϵ is $(eps(Float64))")
println("Is 1 + ϵ/2 > 1? $(1.0 + eps(Float64)/2 > 1.0)")

println("The smallest representable number larger than 1.0 is $(nextfloat(1.0))")
println("The largest representable number smaller than 1.0 is $(prevfloat(1.0))")

println("32 bit machine epsilon is $(eps(Float32))")
println("Is 1 + ϵ/2 > 1? $(Float32(1) + eps(Float32)/2 > 1)")

## Fact 2: There is a smallest representable number
println("64 bit smallest float is $(floatmin(Float64))")
println("32 bit smallest float is $(floatmin(Float32))")
println("16 bit smallest float is $(floatmin(Float16))")

## Fact 3: There is a largest representable number

println("64 bit largest float is $(floatmax(Float64))")
println("32 bit largest float is $(floatmax(Float32))")
println("16 bit largest float is $(floatmax(Float16))")

## Epsilon is about differences in number magnitudes
println("Is 1/100 + ϵ/100 > 1/100? $(1.0/100 + eps(Float64)/100 > 1.0/100)")

println("Because the next representable number from 1/100 is $(nextfloat(1.0/100.0)),
 so the machine epsilon at that scale is $(eps(1.0/100.0))")

 println("The mininum float is $(floatmin(Float64)), and the next float is $(nextfloat(floatmin(Float64))),
 so their difference $(nextfloat(floatmin(Float64)) - floatmin(Float64))
 is equal to epsilon at that scale $(eps(floatmin(Float64)))")

 println("Is floatmin + ϵ(1.0) = ϵ(1.0)? $(floatmin(Float64) + eps(1.0) == eps(1.0))")

## The limits of computers: time is a flat circle
 println("The largest 64 bit integer is $(typemax(Int64))")
println("Add one to it and we get: $(typemax(Int64)+1)")
println("It loops us around the number line: $(typemin(Int64))")

## Rounding

## Error example 1
println("Half of π is: $(π/2)")

x = (1e-20 + 1) - 1   
y = 1e-20 + (1 - 1)   

println("The difference is: $(x-y).")

## Error example 2
println("64 bit: 13 - √168 = $(13-sqrt(168))")
println("32 bit: 13 - √168 = $(convert(Float32,13-sqrt(168)))")
println("16 bit: 13 - √168 = $(convert(Float16,13-sqrt(168)))")

x64 = 13-sqrt(168); x32 = convert(Float32,13-sqrt(168)); x16 = convert(Float16,13-sqrt(168));
f(x) = x^2 - 26x + 1;
println("64 bit: $(f(x64))")
println("32 bit: $(f(x32))")
println("16 bit: $(f(x16))")

## Error example 3
println("100000.2 - 100000.1 is: $(100000.2 - 100000.1)")
if (100000.2 - 100000.1) == 0.1
    println("and it is equal to 0.1")
else
    println("and it is not equal to 0.1")
end

(100000.2 - 100000.1) ≈ 0.1 # You type \approx then hit TAB

isapprox(100000.2 - 100000.1, 0.1)