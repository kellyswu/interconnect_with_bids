## Cell 1: loading packages

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
branch = DataFrame(
    fr_bus = [1, 1, 2],
    to_bus = [2, 3, 3],
    b = [-20.0, -10.0, -12.5], # susceptances
    s = [0.40, 0.40, 0.30], # thermal flow limits
    s_max_add = [10.0, 10.0, 10.0], # set very high so not binding. In case
    # we want to represent the range in which the constant c_up is assumed valid.
    c_up = [0.010, 0.010, 0.005], # capital cost per p.u. of added capacity
)

# ---- gen -------------------------------------------------------------------
gen = DataFrame(
    bus  = [1, 2, 3],
    pmin = [0.00, 0.00, 0.00],
    pmax = [1.20, 0.80, 0.60],
    is_new = [true, true, false], # G^new: the new generators bidding for transmission
)

# ---- load ------------------------------------------------------------------
# One row per load, NOT one row per bus (bus 1 simply has no load row).
load = DataFrame(
    bus = [3],
    pd  = [1.00]
)

# ---- gen cost --------------------------------------------------------------
# a 1D vector, one entry per generator, similar to in HW 1 setup
# assumed constant marginal cost per p.u. for now (can put back in cost curves calcs later)
gen_cost = [0.01, 0.02, 0.03]

# ---- gen transmission bid --------------------------------------------------
transmission_bids = [0.0023, 0.0002, 0.0] # transmission bid for each generator
# MAX cost per p.u. of transmission each generator g is willing to pay
# TOY case. Used Claude to help come up with toy case. We have a low c_up
# to make transmission be built in the base case. Gen 1 bid generous, gen 2 barely bids
# sumT means lines are able to build
# flowT means gen 2's bid binds, only a bit of upgrading is built

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

# A test build
# ref_bus = 3

# PTDF = build_ptdf(bus, branch, ref_bus)

## Cell 5: pre-calculations required for the transmission bid constraint when allocated by "share of flow" on line
# note: the share of an upgrade assigned to a generator is based on its share of flow on the line in the baseline DCOPF.
# if the transmission cost is too high, the total transmission upgrade in the grid
# is limited (which means downstream the project may be required to face curtailment). Alternatively,
#the rest of the cost is socialized. To be determined.

# calculate bus-average ptdf to get the direction of flow for each generator
# (NEW and existing) on the existing grid, without transmission expansion (line upgrades)
function build_dfax(bus, branch)
    """
    DFAX (distribution factor) matrix: the PTDF averaged over every possible choice
    of reference bus, i.e. a uniformly distributed slack.
    nline x nbus matrix.
    """
    nbus  = size(bus, 1)
    nline = size(branch, 1)

    DFAX = zeros(nline, nbus)
    for r in 1:nbus
        DFAX .+= build_ptdf(bus, branch, r)
    end
    return DFAX ./ nbus
end

function flow_direction(base_flow; tol = 1e-9)
    """sigma_l in {+1,-1}: direction of the constrained flow in the baseline DCOPF."""
    return [f < -tol ? -1.0 : 1.0 for f in base_flow]
end

# generator's share of flow on line l = max(0, sigma_l * PTDF_l(n(g) Pg nameplate))/total share
# these can be pre-calculated prior to the optimization model!
function fixed_allocation_shares(base_flow, DFAX, gen)
    """
    Option A fixed allocation shares, returned as S[g,l] (one row per generator,
    one column per line):

        s[g,l] = max(0, sigma_l * DFAX[l,n(g)] * Pbar_g) / sum_g' max(0, same)
    only new generators receive a share.
    Pbar_g is nameplate capacity.
    """
    ngen, nline = size(gen, 1), size(DFAX, 1)
    sigma = flow_direction(base_flow)

    contrib = zeros(ngen, nline)
    for g in 1:ngen, l in 1:nline
        gen[g,:].is_new || continue
        contrib[g, l] = max(0.0, sigma[l] * DFAX[l, Int(gen[g,:].bus)] * gen[g,:].pmax)
    end

    S = zeros(ngen, nline)
    for l in 1:nline
        tot = sum(contrib[:, l])
        tot > 0 && (S[:, l] .= contrib[:, l] ./ tot)
    end
    return S, sigma, contrib
end

# DFAX is a pure function of the network, so it can be built straight away.
# S needs the baseline flows, so it is built in Cell 7 once `base` has been solved.
DFAX = build_dfax(bus, branch)


## Cell 6: transmission expansion (line upgrades) and DCOPF optimization model with constraints!

# TODO review the model and add in the transmission bid constraint

function solve_texp_DCOPF_ptdf(bus, gen, load, branch, gen_cost, solver; ref_bus = 3,
                               hours = 1.0, allow_upgrades = true)
    #=
    PTDF (shift-factor) DCOPF with line-capacity upgrades as decision variables.

        min  sum_l c_up[l] * dF[l]  +  hours * sum_g cg[g] * PG[g]
        s.t. sum_g PG[g] == sum_n D[n]                                (Balance)
             -(s[l] + dF[l]) <= psi[l,:] * inj <= s[l] + dF[l]        (FBs_min / FBs_max)
             0 <= dF[l] <= s_max_add[l]
             pmin[g] <= PG[g] <= pmax[g]

    `hours` weights the single operating period so the per-hour energy savings are
    comparable to the one-off capital cost of an upgrade. With hours = 1 you are
    pricing one hour of congestion relief against the full capital cost.
    `allow_upgrades = false` pins dF = 0, which recovers the baseline DCOPF.
    =#

    tstart = time()

    #Get number of elements
    nbus = size(bus,1)
    ngen = size(gen,1)
    nload = size(load,1)
    nline = size(branch,1)

    cg = gen_cost # vector with variable costs of generation per generator

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
    @variable(texp_DCOPF, DF[1:nline] >= 0)  # capacity ADDED to line l

    # Generator min and max (the same)
    @constraint(texp_DCOPF, [i = 1:ngen], PG[i] >= gen[i,:].pmin)
    @constraint(texp_DCOPF, [i = 1:ngen], PG[i] <= gen[i,:].pmax)

    # Upgrade bound: 0 <= DF[l] <= s_max_add[l]
    upgrade_cap = allow_upgrades ? branch.s_max_add : zeros(nline)
    @constraint(texp_DCOPF, UpBound[l = 1:nline], DF[l] <= upgrade_cap[l]) # currently set really high upgrade cap to not constrain

    # systemwide supply demand balance
    Balance = @constraint(texp_DCOPF, sum(PG[i] for i in 1:ngen) == sum(load[i,:].pd for i in 1:nload))

    # net injection at each bus. init = 0.0 for buses with no gen row or no load row
    @expression(texp_DCOPF, inj[b = 1:nbus],
        sum(PG[i] for i in findall(gen.bus .== b); init = 0.0) -
        sum(load.pd[i] for i in findall(load.bus .== b); init = 0.0))

    # flow on each line = row of the shift factor matrix times the injection vector
    @expression(texp_DCOPF, flow[l = 1:nline],
        sum(psi[l,b] * inj[b] for b in 1:nbus))

    # thermal limits, now relaxable by the upgrade variable DF.
    # kept for duals later
    FBs_max = Array{Any}(nothing, nline)
    FBs_min = Array{Any}(nothing, nline)
    for l = 1:nline
        fs = branch[l,:].s # existing flow limit of branch
        FBs_max[l] = @constraint(texp_DCOPF, flow[l] - DF[l] <=  fs)
        FBs_min[l] = @constraint(texp_DCOPF, flow[l] + DF[l] >= -fs)
    end

    
    # Transmission bid constraint based on the selected allocation method.
    # Skipped when allow_upgrades = false: DF is pinned to zero there, so the
    # constraint is vacuous, and the baseline solve is what produces the shares S.
    if allow_upgrades
        if bid_constraint == :sumT # total transmission bids must cover total upgrade cost
            @constraint(texp_DCOPF, sum(branch[l,:].c_up * DF[l] for l = 1:nline) <= sum(transmission_bids))
        elseif bid_constraint == :flowT
            # Option A: each NEW generator's allocated share of the INCREMENTAL upgrade
            # cost (above whatever was already built in the baseline) must be covered by
            # its own bid:  sum_l s[g,l] * c_up[l] * (DF[l] - DF_base[l]) <= T_max[g]
            @constraint(texp_DCOPF, BidShare[g = 1:ngen; gen[g,:].is_new],
                sum(S[g,l] * branch[l,:].c_up * (DF[l]) for l = 1:nline) <= # (DF[l] - DF_base[l])later
                transmission_bids[g])
        end
    end


    # objective function

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

    # LMPs. Divide by `hours` to report a price per MWh rather than per operating period.
    # LMPs - we can see how they change in each instance - if the generators lose money.
    LMPs = zeros(nbus)
    for b in 1:nbus
        LMPs[b] = (dual(Balance) +
                   sum(psi[l,b] * (dual(FBs_max[l]) + dual(FBs_min[l])) for l in 1:nline)) / hours
    end


    results = (
        objective    = objective_value(texp_DCOPF),
        capital_cost = sum(branch[l,:].c_up * value(DF[l]) for l = 1:nline),
        op_cost      = hours * sum(cg[i] * value(PG[i]) for i = 1:ngen),
        PG           = value.(PG),
        DF           = value.(DF),
        flow         = [value(flow[l]) for l in 1:nline],
        new_rating   = branch.s .+ value.(DF),
        LMPs         = LMPs,
        upgrade_dual = [dual(UpBound[l]) for l in 1:nline],
    )

    return results, build_time, solve_time, total_time
end





## Cell 7: solve the baseline and the expansion case and compare

# transmission-bid allocation toggle: :sumT = Cell 5A, bids must cover the total
# upgrade cost in aggregate; :flowT = Cell 5B, cost split by share of line flow
bid_constraint = :sumT

hours = 10.0
ref_bus = 3 # only fixes the PTDF's angle datum; dispatch, flows and LMPs are invariant to it

# baseline optimization in which no bids are submitted
base, _, _, _ = solve_texp_DCOPF_ptdf(bus, gen, load, branch, gen_cost,
                                      HiGHS.Optimizer; ref_bus = ref_bus, hours = hours,
                                      allow_upgrades = false)

# calculating fixed allocation shares from baseline flows
# these variables are called in the run below
S, sigma, contrib = fixed_allocation_shares(base.flow, DFAX, gen)
# DF_base = base.DF # what the system would build absent the entrants (useful later)

texp, _, _, _ = solve_texp_DCOPF_ptdf(bus, gen, load, branch, gen_cost,
                                      HiGHS.Optimizer; ref_bus = ref_bus, hours = hours,
                                      allow_upgrades = true)

println("\n==== baseline (no upgrades allowed) ====")
println("total cost      = ", round(base.objective, digits = 6))
println("dispatch PG     = ", round.(base.PG, digits = 4))
println("flows           = ", round.(base.flow, digits = 4), "  (limits ", branch.s, ")")
println("LMPs [\$/p.u.]   = ", round.(base.LMPs, digits = 5))

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
println("net saving vs baseline = ", round(base.objective - texp.objective, digits = 6))


## Cell 8: IF TIME-do a waterfall-type cost assignment for constraint
# that allocates costs to generators

