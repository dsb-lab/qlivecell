using DataFrames
using CSV
using Statistics
using LinearAlgebra
using StatsBase
using Random
using PyPlot
using PyCall
animation = pyimport("matplotlib.animation")
patch = pyimport("matplotlib.patches")

const pathh = "/home/pablo/Desktop/"
pathh_save = "/home/pablo/Desktop/PhD/EmbryonicDev/data_analysis/figures/2h_data/"
include(join([pathh, "PhD/projects/EmbryonicDev/data_analysis/code/types.jl"]))
include(join([pathh, "PhD/projects/EmbryonicDev/data_analysis/code/utils.jl"]))

data = DataFrame(CSV.File(join([pathh,"PhD/projects/Data/blastocysts/CSVs/Claire_2h/Lineage_2hr.csv"])))

_Embs = extract_data(data)
Embs = extract_data_cells.(_Embs)

ac = 1
apo_embs = [10,13,14,16,18,19,25]
apo_cells = [[1,2],[1,2],[1],[1],[1],[1],[1]]
tissues=["ICM", "Polar"]

emb = apo_embs[1]
apo_cells, apo_starts, apo_ends = select_apo(Embs[emb], ["ICM", "Polar"])

Emb = Embs[emb]
cid = 1
cell = Emb[apo_cells[cid]]
t = cell.TimeM[apo_starts[cid]]
x = cell.X[apo_starts[cid]]
y = cell.Y[apo_starts[cid]]
z = cell.Z[apo_starts[cid]]

println(Emb[cid].E_ID)
println(x)
println(y)
println(z) 
println(t)