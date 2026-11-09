## Cell 1: loaing packages

using Pkg; Pkg.add(["PowerModels", "JuMP", "Gurobi", "DataFrames", "Memento", "HiGHS", "Plots"])
using PowerModels, JuMP, Gurobi, DataFrames, LinearAlgebra, Memento, Logging, HiGHS, Plots
Memento.config!("error")
Logging.disable_logging(Logging.Warn)


## Cell 2: Three-bus setup

baseMVA = 100.0 # multiplier for the numbers shown, for example, 1.00 p.u. = 100 MW

# ---- bus -------------------------------------------------------------------
bus = DataFrame(:bus => [1; 2; 3])

# ---- branch ----------------------------------------------------------------
# directly specify susceptances
# s_max_add / c_up are the transmission-expansion parameters:
#   s_max_add = most capacity that may be added to the line  [p.u.]   (dF_max in the formulation)
#   c_up      = capital cost per p.u. of added capacity       [$/p.u.] (C_up in the formulation)
# An upgrade raises the line's thermal rating s but leaves b unchanged, so the
# PTDF matrix is a constant and the problem stays an LP.
branch = DataFrame(
    fr_bus = [1, 1, 2],
    to_bus = [2, 3, 3],
    b = [-20.0, -10.0, -12.5], # susceptances
    s = [0.40, 0.40, 0.300], # thermal flow limits
    s_max_add = [0.30, 0.30, 0.30], # max capacity addition per line
    c_up = [0.030, 0.030, 0.020], # capital cost per p.u. of added capacity
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

function build_ptdf(bus, branch, ref_bus)
    """
    PTDF matrix for the DC network (shift factor matrix).
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

    noref = setdiff(1:nbus, [ref_bus]) # setdiff - returns set of buses (in order) that are not the reference bus
    PTDF  = zeros(nline, nbus)
    PTDF[:, noref] = (Bd * A[:, noref]) * inv(Bbus[noref, noref]) # remove the reference bus from the bus susceptance matrix
    return PTDF
end

## Cell 4: Build PTDF matrix
ref_bus = 3

PTDF = build_ptdf(bus, branch, ref_bus)

## Cell 5: transmission expansion (line upgrades) and DCOPF optimization model with constraints!

function solve_texp_DCOPF_ptdf(bus, gen, load, branch, gen_cost, ref_bus, solver;
                               hours = 1.0, allow_upgrades = true)
    #=
    PTDF (shift-factor) DCOPF with line-capacity upgrades as decision variables.

        min  sum_l c_up[l] * dF[l]  +  hours * sum_g cg[g] * PG[g]
        s.t. sum_g PG[g] == sum_n D[n]                                (Balance)
             -(s[l] + dF[l]) <= psi[l,:] * inj <= s[l] + dF[l]        (FBs_min / FBs_max)
             0 <= dF[l] <= s_max_add[l]
             pmin[g] <= PG[g] <= pmax[g]

    `hours` weights the single operating period so the $/MWh energy savings are
    comparable to the one-off $/MW capital cost. With hours = 1 you are pricing
    one hour of congestion relief against the full capital cost of the upgrade.
    `allow_upgrades = false` pins dF = 0, which recovers the baseline DCOPF.
    =#

    tstart = time()

    #Get number of elements
    nbus = size(bus,1)
    ngen = size(gen,1)
    nload = size(load,1)
    nline = size(branch,1)

    cg = gen_cost

    #Create model
    texp_DCOPF = Model(optimizer_with_attributes(
        solver,
        MOI.Silent() => true
    ))

    ##Set tolerances for HiGHS to Gurobi defaults
    if solver == HiGHS.Optimizer
        set_optimizer_attribute(texp_DCOPF, "dual_feasibility_tolerance", 1e-6)
        set_optimizer_attribute(texp_DCOPF, "primal_feasibility_tolerance", 1e-6)
    end

    # getting shift factor matrix
    psi = build_ptdf(bus, branch, ref_bus) # returns the entire shift factor matrix

    # Variables
    @variable(texp_DCOPF, PG[1:ngen] >= 0)   # generator power generation
    @variable(texp_DCOPF, DF[1:nline] >= 0)  # capacity ADDED to line l (dF in the formulation)

    # Generator min and max (the same)
    @constraint(texp_DCOPF, [i = 1:ngen], PG[i] >= gen[i,:].pmin)
    @constraint(texp_DCOPF, [i = 1:ngen], PG[i] <= gen[i,:].pmax)

    # Upgrade bound: 0 <= DF[l] <= s_max_add[l] (upper half; lower half is the variable bound)
    upgrade_cap = allow_upgrades ? branch.s_max_add : zeros(nline)
    @constraint(texp_DCOPF, UpBound[l = 1:nline], DF[l] <= upgrade_cap[l])

    # systemwide supply demand balance
    Balance = @constraint(texp_DCOPF, sum(PG[i] for i in 1:ngen) == sum(load[i,:].pd for i in 1:nload))

    # net injection at each bus. init = 0.0 so buses with no gen row or no load row
    # (bus 1 has no load) sum to zero instead of erroring on an empty collection.
    @expression(texp_DCOPF, inj[b = 1:nbus],
        sum(PG[i] for i in findall(gen.bus .== b); init = 0.0) -
        sum(load.pd[i] for i in findall(load.bus .== b); init = 0.0))

    # flow on each line = row of the shift factor matrix times the injection vector
    @expression(texp_DCOPF, flow[l = 1:nline],
        sum(psi[l,b] * inj[b] for b in 1:nbus))

    # thermal limits, now relaxable by the upgrade variable DF.
    # DF is moved to the LHS so the constant RHS stays s[l]: the dual of each
    # constraint is then d(objective)/d(rating), i.e. the congestion rent.
    FBs_max = Array{Any}(nothing, nline)
    FBs_min = Array{Any}(nothing, nline)
    for l = 1:nline
        fs = branch[l,:].s # existing flow limit of branch
        FBs_max[l] = @constraint(texp_DCOPF, flow[l] - DF[l] <=  fs)
        FBs_min[l] = @constraint(texp_DCOPF, flow[l] + DF[l] >= -fs)
    end

    @objective(texp_DCOPF, Min,
        sum(branch[l,:].c_up * DF[l] for l = 1:nline) +          # capital cost of upgrades
        hours * sum(cg[i] * PG[i] for i = 1:ngen))               # operating cost

    tend_build = time()

    #Solve model
    optimize!(texp_DCOPF)
    tend = time()

    build_time = tend_build - tstart
    solve_time = tend - tend_build
    total_time = tend - tstart

    # LMPs. Raising load at bus b by 1 raises the Balance RHS by 1 and lowers inj[b]
    # by 1, which slackens each line constraint exactly as raising its RHS by psi[l,b]
    # would. So the congestion term ADDS the duals of both limits, and dual(FBs_max)
    # is already <= 0 under JuMP's convention for a Min problem (no sign flip needed).
    # Divide by `hours` to report a price per MWh rather than per operating period.
    LMPs = zeros(nbus)
    for b in 1:nbus
        LMPs[b] = (dual(Balance) +
                   sum(psi[l,b] * (dual(FBs_max[l]) + dual(FBs_min[l])) for l in 1:nline)) / hours
    end

    # Congestion rent earned by each line's rating = |dual(max)| + |dual(min)|.
    # Complementary slackness: a line with 0 < DF < s_max_add is upgraded until this
    # equals its capital cost c_up; below c_up it is not upgraded, above it is maxed out.
    rent = [-dual(FBs_max[l]) + dual(FBs_min[l]) for l in 1:nline]

    results = (
        objective    = objective_value(texp_DCOPF),
        capital_cost = sum(branch[l,:].c_up * value(DF[l]) for l = 1:nline),
        op_cost      = hours * sum(cg[i] * value(PG[i]) for i = 1:ngen),
        PG           = value.(PG),
        DF           = value.(DF),
        flow         = [value(flow[l]) for l in 1:nline],
        new_rating   = branch.s .+ value.(DF),
        LMPs         = LMPs,
        rent         = rent,
        upgrade_dual = [dual(UpBound[l]) for l in 1:nline],
    )

    return results, build_time, solve_time, total_time
end

## Cell 6: solve the baseline and the expansion case, and compare

hours = 1.0

base, _, _, _ = solve_texp_DCOPF_ptdf(bus, gen, load, branch, gen_cost, ref_bus,
                                      HiGHS.Optimizer; hours = hours, allow_upgrades = false)
texp, _, _, _ = solve_texp_DCOPF_ptdf(bus, gen, load, branch, gen_cost, ref_bus,
                                      HiGHS.Optimizer; hours = hours, allow_upgrades = true)

println("\n==== baseline (no upgrades allowed) ====")
println("total cost      = ", round(base.objective, digits = 6))
println("dispatch PG     = ", round.(base.PG, digits = 4))
println("flows           = ", round.(base.flow, digits = 4), "  (limits ", branch.s, ")")
println("LMPs [\$/p.u.]   = ", round.(base.LMPs, digits = 5))
println("rating rent     = ", round.(base.rent, digits = 5))

println("\n==== with line upgrades ====")
println("total cost      = ", round(texp.objective, digits = 6),
        "  (capital ", round(texp.capital_cost, digits = 6),
        " + operating ", round(texp.op_cost, digits = 6), ")")
println("dispatch PG     = ", round.(texp.PG, digits = 4))
println("capacity added  = ", round.(texp.DF, digits = 4), " p.u. = ",
        round.(texp.DF .* baseMVA, digits = 2), " MW")
println("new ratings     = ", round.(texp.new_rating, digits = 4))
println("flows           = ", round.(texp.flow, digits = 4))
println("LMPs [\$/p.u.]   = ", round.(texp.LMPs, digits = 5))
println("rating rent     = ", round.(texp.rent, digits = 5), "  vs c_up ", branch.c_up)
println("net saving vs baseline = ", round(base.objective - texp.objective, digits = 6))

# complementary-slackness check on every upgraded-but-not-maxed line
for l in 1:size(branch, 1)
    if 1e-7 < texp.DF[l] < branch[l,:].s_max_add - 1e-7
        println("line ", l, ": interior upgrade, rent ", round(texp.rent[l], digits = 6),
                " should equal c_up ", branch[l,:].c_up)
    end
end
