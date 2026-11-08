## Cell 1: load packages

using Pkg; Pkg.add(["PowerModels", "JuMP", "Gurobi", "DataFrames", "Memento", "HiGHS", "Plots"])
using PowerModels, JuMP, Gurobi, DataFrames, LinearAlgebra, Memento, Logging, HiGHS, Plots
Memento.config!("error")
Logging.disable_logging(Logging.Warn)


## Cell 2: Three-bus setup

baseMVA = 100.0 # multiplier for the numbers shown, for example, 1.00 p.u. = 100 MW
ref_bus = 1          # slack / reference bus (make_basic_network picks bus with bus_type == 3)

# ---- bus -------------------------------------------------------------------
# load_opf_data only keeps va/vm; the models really just use size(bus,1).
bus = DataFrame(
    va = [0.0, 0.0, 0.0],
    vm = [1.0, 1.0, 1.0],
)

# ---- branch ----------------------------------------------------------------
# directly specify susceptances
branch = DataFrame(
    fr_bus = [1, 1, 2],
    to_bus = [2, 3, 3],
    b = [-20.0, -10.0, -12.5], # susceptances
    s = [0.40, 0.40, 0.300], # thermal flow limits
)

# ---- gen -------------------------------------------------------------------
# here, we assume that the capacity expansion part of the model has been run and the two new generators 
# at buses 1 and 2 are assumed to have cleared capacity expansion
gen = DataFrame(
    bus  = [1, 2, 3],
    pmin = [0.00, 0.00, 0.00],
    pmax = [1.20, 0.80, 0.60],
)

# ---- load ------------------------------------------------------------------
# One row per load, NOT one row per bus (bus 1 simply has no load row).
load = DataFrame(
    bus = [3],
    pd  = [1.00],
)

# ---- gen cost --------------------------------------------------------------
# a 1D vector, one entry per generator, similar to in HW 1 setup
gen_cost = [0.01, 0.02, 0.03]

println("branch susceptances b (negative, as in load_opf_data): ", round.(branch.b, digits = 3))
println("total load = ", sum(load.pd), " p.u. = ", sum(load.pd) * baseMVA, " MW")
println("total capacity = ", sum(gen.pmax), " p.u.")
show(branch, summary = false); println()

## Cell 3: PTDF function


function build_ptdf(bus, branch; ref_bus = 1)
    """
    PTDF matrix for the DC network.
    Rows are lines, columns are buses, and `f = PTDF * inj` for any net-injection vector `inj`.
    """
    nbus, nline = size(bus, 1), size(branch, 1)

    # incidence: +1 at the from-bus, -1 at the to-bus
    A = zeros(nline, nbus)
    for l in 1:nline
        A[l, Int(branch[l, "fr_bus"])] += 1.0
        A[l, Int(branch[l, "to_bus"])] -= 1.0
    end

    beta = -branch[:, "b"]          # positive series susceptance; = 1/x when r = 0
    Bd   = Diagonal(beta)
    Bbus = A' * Bd * A              # singular by construction (rows sum to zero)

    noref = setdiff(1:nbus, [ref_bus])
    PTDF  = zeros(nline, nbus)
    PTDF[:, noref] = (Bd * A[:, noref]) * inv(Bbus[noref, noref])
    return PTDF
end

PTDF = build_ptdf(bus, branch; ref_bus = ref_bus)


