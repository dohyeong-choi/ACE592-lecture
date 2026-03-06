##############################################################################
######################## Lecture 1.3 - Intro to Julia ########################
##############################################################################

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
