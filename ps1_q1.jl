## Cost Function
C(q) = 10*q + 2*q^2 
C(1)

## A. Conditional Profit Function
π(p, q) = p*q-C(q)
π(1, 1)

## Ploting
q = collect(range(0, stop = 20, length = 100))
π.(50, q)

plot(q, [π.(p, q) for p = 10:10:50],
                    linewidth = 1.5, linecolor = [:red :orange :yellow :green :blue],
                    tickfontsize = 12, grid = :no,
                    xlabel = "q", ylabel = "π(q)" ,
                    label = ["p = 10" "p = 20" "p = 30" "p = 40" "p = 50"])

## B. Supply Fucntion
qstar(p) = (p-10)/4
qstar(11)

p = collect(range(0, stop = 50, length = 100))
plot(qstar.(p), p, 
    linewidth = 1.5, linecolor = [:black],
    tickfontsize = 12, grid = :no,
    xlabel = "q", ylabel = "p" ,
    label = "qstar(p)")

## C. Profit Function
Π(p) = ( (p-10)^2 ) / 8
Π(12)

using FiniteDifferences; # Import package
my_central_diff  = central_fdm(3, 1) # Forward difference operator with two points for 1st order derivative
my_central_diff(Π, 10)

## D. Analysis and discussion
p = collect(range(0, stop = 50, length = 100))
plot(p, my_central_diff.(Π, p),
    linewidth = 1.5, linecolor = [:black],
    tickfontsize = 12, grid = :no,
    xlabel = "p", ylabel = "dΠ/dp",
    label = "derivative")
plot!(p, qstar.(p), label="supply function", lw=1.5)

#======================================== Discussion ========================================
The derivative of the profit function is perfectly overlapped by the firm's supply function.
This implies one of the famous lemmas in microeconomics, Hotelling's lemma.
=============================================================================================#
##

plot(ns_trap, errs_trap, label="Trapezoid", lw=2, marker=:circle, ms=3, yscale=:log10,
     xlabel="Number of nodes", ylabel="Absolute error (log scale)",
     title="Convergence: Expected utility of Cobb-Douglas",
     legend=:bottomright, size=(700, 400), ylims=(1e-16, 1e0))
plot!(ns_lege, errs_lege, label="Gauss-Legendre", lw=2, marker=:square, ms=3)
plot!(ns_herm, errs_herm, label="Gauss-Hermite", lw=2, marker=:diamond, ms=3)
hline!([eps()], label="Machine epsilon", ls=:dash, color=:gray)
