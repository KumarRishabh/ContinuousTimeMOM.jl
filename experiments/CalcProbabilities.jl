using PrettyTables
using Revise
using Plots
using Random
using ProgressMeter

include("../src/ContactProcess.jl")
using .ContactProcess

function sum_edges(matrix)
    return sum(matrix[2, :]) + sum(matrix[end - 1, :]) + sum(matrix[:, 2]) + sum(matrix[:, end - 1])
end