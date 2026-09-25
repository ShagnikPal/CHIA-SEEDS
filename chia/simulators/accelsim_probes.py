"""chia.simulators.accelsim_probes — idealization probes and cost functions.

**Why probes instead of counter-reading.** The obvious step-2 design is "dump
the counters, let the model name the bottleneck, propose fixes". It does not
work reliably, and the reason is visible in the very first baseline measured
here: 526K L1 bank conflicts made the cache look like the obvious culprit, while
the kernel was actually parallelism-starved at 2.4 blocks/SM and 30% occupancy.
Counters are correlational. An agent reading them picks the biggest number,
which is usually a symptom.

A probe answers a causal question instead, by measuring: *if this resource were
free, how much faster would the kernel be?* Set the L1 to absurd capacity and
the delta against baseline is the headroom the cache is actually costing you --
not a correlation, an upper bound. Probes are the same shape as the fan-out
step 2 already performs (N configs, one trace, embarrassingly parallel), cost
the same ~3 minutes each, and turn step 3 from a guess into a ranking.

Probes are diagnostics, not candidate designs. ``ideal_l1`` is not buildable.
Its job is to tell step 3 which knob family is worth sweeping and step 4 which
kernel rewrite is worth writing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from chia.simulators.accelsim import AccelSimRunResult, sweep_configs


# Re-tracing the same source shifts the baseline by ~0.4%, so a headroom below
# that is indistinguishable from having changed nothing. HW_TARGET_PCT is the
# cut above which a knob family is worth a sweep slot; NOISE_PCT is the cut
# below which a number is not a measurement at all.
NOISE_PCT = 1.0
HW_TARGET_PCT = 5.0


@dataclass(frozen=True)
class Probe:
    """One idealization. ``knobs`` are deliberately unbuildable."""

    name: str
    knobs: dict[str, str]
    question: str
    if_large: str  # what a large speedup implies for hardware
    kernel_hint: str  # what it implies for a kernel rewrite (step 4)
    reference: str = ""
    """Name of the probe to measure headroom against, instead of baseline.

    A probe that must disable a confounding feature to make its real knob land
    is not measured against baseline -- the disabling itself moves cycles. The
    L1 probes have to set ``-gpgpu_adaptive_cache_config 0`` before any
    ``-gpgpu_cache:dl1`` edit takes effect, and that alone costs 0.16%. Naming
    ``adaptive_off`` here makes the report subtract the control rather than
    leaving the subtraction as prose the reader is trusted to perform.
    """

    direction: str = "up"
    """Whether this probe *adds* a resource ("up") or *removes* one ("down").

    Eleven probes are idealizations: they hand the kernel more cache, more
    bandwidth, less latency, and a positive headroom is the expected outcome.
    ``half_sm`` alone takes a resource away, so its large result is negative and
    its ``if_large`` text is written to be read that way.

    The distinction has to be declared rather than inferred from the sign,
    because an idealization *can* come out negative: at 512^3, -dram_latency 1
    is 6.85% SLOWER than baseline. That is not a speedup to explain away -- it
    means the resource is not the constraint and forcing it perturbs request
    timing. Gating on magnitude alone printed "DRAM latency is the constraint"
    next to that number.
    """

    hint_when: str = "large"
    """Whether ``kernel_hint`` applies when this probe is large or when it is flat.

    Eleven of the twelve hints are "the idealization bought cycles, here is the
    software way to buy the same cycles without the hardware change" -- they fire
    on a large result. ``2x_sm`` is the exception: its software advice applies
    precisely when adding SMs did *not* help. Emitting every hint on a large
    result printed "the kernel does not expose enough parallelism" next to a
    15% SM-scaling win, which is the same self-contradiction that ``if_large``
    text next to a 0.00% headroom used to produce.
    """

    trace_knobs: dict[str, str] = field(default_factory=dict)
    """Overrides for ``trace.config`` rather than ``gpgpusim.config``.

    Not a convenience. In trace-driven mode the SASS instruction latencies come
    from ``-trace_opcode_latency_initiation_*`` in ``trace.config``; the
    ``-ptx_opcode_latency_*`` knobs in ``gpgpusim.config`` are read, accepted,
    and never consulted. A probe that edits the PTX knobs is silently inert --
    measured: bit-identical 27921 cycles with fp/int latency set to 1.
    """


# Every probe here has been checked for the failure mode that makes this whole
# approach useless: a knob the parser accepts and the model ignores. See
# ``compare_runs(...)['inert']``. The two known traps are baked in:
#   * -gpgpu_adaptive_cache_config 1 recomputes L1D geometry at each launch and
#     discards -gpgpu_cache:dl1 edits, so every L1 probe must turn it off --
#     and ``adaptive_off`` is included as the control that separates "L1 got
#     bigger" from "adaptive allocation stopped happening".
#   * tensor-core knobs are no-ops in trace-driven mode (HMMA dispatches to a
#     specialized unit), so there is no tensor probe.
IDEALIZATION_PROBES: tuple[Probe, ...] = (
    Probe(
        "adaptive_off",
        {"-gpgpu_adaptive_cache_config": "0"},
        "Control: what does disabling adaptive L1/shmem partitioning cost by itself?",
        "Not a result. Subtract this from every ideal_l1-family probe.",
        "none",
        direction="control",
    ),
    # The second control, and the reason two starve probes were measuring
    # nothing. The tuned A100 config ships -gpgpu_perfect_inst_const_cache 1,
    # which makes every instruction and constant access a hit regardless of the
    # geometry those caches are configured with. ``starve_const`` and
    # ``starve_icache`` therefore could not bind: shrinking a cache that is
    # perfect by fiat is a no-op, and their 0.00% was an artefact of the flag
    # rather than evidence the structures are dead. Both now turn the flag off
    # and are measured against this control, so the cut is real and the cost of
    # the flag itself is subtracted rather than attributed to the cut. This is
    # also why the area ledger refused these two the credit it gives the
    # texture cache: TEX_ACC is genuinely zero, while the const/icache
    # counters were suppressed.
    Probe(
        "perfect_ic_off",
        {"-gpgpu_perfect_inst_const_cache": "0"},
        "Control: what does costing real instruction and constant fetches by "
        "itself?",
        "Not a result. Subtract this from every icache/const-family probe.",
        "none",
        direction="control",
    ),
    Probe(
        "ideal_l1",
        {
            "-gpgpu_adaptive_cache_config": "0",
            "-gpgpu_unified_l1d_size": "1536",
            "-gpgpu_cache:dl1": "S:192:128:64,L:T:m:L:L,A:4096:512,16:0,32",
        },
        "If the L1 never missed and never ran out of MSHRs, how much faster?",
        "L1 capacity/associativity/MSHR count is worth sweeping.",
        "Improve locality: tile into shared memory, or reorder so reuse lands "
        "within an L1 lifetime.",
        reference="adaptive_off",
    ),
    Probe(
        "ideal_l2",
        {
            "-gpgpu_cache:dl2": "S:512:128:32,L:B:m:L:X,A:1024:32,32:0,32",
            "-gpgpu_l2_rop_latency": "20",
        },
        "If the L2 absorbed the whole working set, how much faster?",
        "L2 capacity or banking is worth sweeping.",
        "Shrink the footprint per wave, or block the loop so the working set "
        "fits L2.",
    ),
    Probe(
        "fast_mem_lat",
        {"-gpgpu_l1_latency": "2", "-gpgpu_smem_latency": "2"},  # >1 is asserted
        "If L1 and shared-memory hits returned in 1 cycle, how much faster?",
        "The kernel is latency-bound, not bandwidth-bound. More MSHRs or a "
        "shorter pipeline helps; more capacity does not.",
        "Increase memory-level parallelism: unroll, issue independent loads "
        "earlier, raise ILP per thread.",
    ),
    Probe(
        "zero_dram_lat",
        {"-dram_latency": "1", "-gpgpu_l2_rop_latency": "1"},
        "If DRAM answered instantly, how much faster?",
        "DRAM latency, not bandwidth, is the constraint -- look at queueing "
        "and row-buffer policy before widening the bus.",
        "Prefetch, or raise occupancy so more warps hide the latency.",
    ),
    Probe(
        "wide_icnt",
        {
            "-icnt_flit_size": "160",
            "-icnt_in_buffer_limit": "4096",
            "-icnt_out_buffer_limit": "4096",
        },
        "If the on-chip network had no congestion, how much faster?",
        "NoC width/topology is worth sweeping.",
        "Reduce request count: coalesce accesses, use wider loads (float4).",
    ),
    Probe(
        "fast_alu",
        {},
        "If arithmetic had 1-cycle latency, how much faster?",
        "The kernel is dependency-bound on the ALU pipeline. Expect this to "
        "correlate with a high shader_cycle_distro_1 (scoreboard) fraction.",
        "Break the dependency chain: multiple accumulators, unroll-and-jam.",
        trace_knobs={
            "-trace_opcode_latency_initiation_sp": "1,1",
            "-trace_opcode_latency_initiation_int": "1,1",
        },
    ),
    # NOTE: there is deliberately no scheduler-count probe. Every ID_OC_* entry
    # of -gpgpu_pipeline_widths and every specialized unit's ID_OC_SPEC width
    # must equal -gpgpu_num_sched_per_core (shader.cc:130, :134, :147).
    # Satisfying all of them still fails on SM80_A100: 8 schedulers deadlocks at
    # ~cycle 5012 (unchanged by 2x operand collectors, so it is not collector
    # starvation) and 2 schedulers aborts at abstract_hardware_model.h:2001
    # 'reg_id < regs.size()' under -gpgpu_sub_core_model 1. Issue capability is
    # probed below via the OC_EX_* / EX_WB widths, which carry no such coupling.
    Probe(
        "wide_lsu",
        # index 9 is OC_EX_MEM, index 10 is EX_WB; the ID_OC_* entries are left
        # at the scheduler count because the simulator asserts that equality.
        {"-gpgpu_pipeline_widths": "4,4,4,4,4,4,4,4,4,8,16,4,4"},
        "If the load/store unit could accept and retire twice as many "
        "instructions per cycle, how much faster?",
        "The structural stall is LD/ST issue-to-execute width. A flat result "
        "means the structural stall is the fixed per-scheduler memory issue "
        "rate, which no width knob can raise -- a software finding, not a "
        "hardware one.",
        "Cut the number of memory instructions: tile, vectorize the loads, or "
        "stage through shared memory.",
    ),
    Probe(
        "many_mshr",
        # -gpgpu_adaptive_cache_config 1 recomputes L1D geometry per launch and
        # silently discards this line, so the probe must turn it off. Compare
        # against adaptive_off, not against baseline.
        {
            "-gpgpu_adaptive_cache_config": "0",
            "-gpgpu_cache:dl1": "S:4:128:64,L:T:m:L:L,A:2048:256,16:0,128",
        },
        "If the L1 could track 4x the outstanding misses and move 4x the bytes "
        "per cycle, how much faster?",
        "The kernel is limited by miss-handling capacity -- memory-level "
        "parallelism is being throttled at the L1, not by latency.",
        "Increase per-thread independent loads so more misses are in flight.",
        reference="adaptive_off",
    ),
    Probe(
        "2x_sm",
        {"-gpgpu_n_clusters": "216"},
        "If the GPU had twice the SMs, how much faster?",
        "Near-2x means the kernel is throughput-scalable and the chip is the "
        "limit. Near-1x means it is NOT parallelism-limited and adding SMs is "
        "wasted area -- the most expensive mistake this loop can make.",
        "The kernel does not expose enough parallelism: more blocks, smaller "
        "blocks, or split-K.",
        hint_when="flat",
    ),
    Probe(
        "half_sm",
        {"-gpgpu_n_clusters": "54"},
        "How much slower with half the SMs?",
        "Much less than 2x slower means the baseline SMs are already idle and "
        "the chip is oversized for this kernel. Read together with 2x_sm: this "
        "pair is the area-efficiency gradient, and it is the reason a "
        "cycles-only objective is degenerate.",
        "none",
        direction="down",
    ),
    # ------------------------------------------------------------------
    # Upward twins added 2026-09-23, so that every *sized* structure is
    # quantified in both directions and can enter an iso-area trade. Before
    # this, six structures had a starve probe and no idealization (the three
    # instruction-side caches, register banks, and the functional-unit/issue
    # width axis), and the L2 -- the largest actionable headroom in the whole
    # sweep -- had three upward probes and no downward one. A one-sided
    # measurement can say a resource is load-bearing; it cannot say what a
    # marginal unit of it is worth, which is the number a reallocation needs.
    # ------------------------------------------------------------------
    #
    # The clean middle rung of the L2 ladder. ``ideal_l2`` moves capacity,
    # associativity, MSHRs and ROP latency at once and ``l2_capacity`` still
    # moves capacity, associativity and MSHRs together (A:192 -> A:1024), so
    # neither can be divided by an area delta. This probe and its two
    # starvation twins move *sets only* -- 32 / 64 / 128 / 256 at stock
    # associativity, stock MSHR count and stock -gpgpu_l2_rop_latency -- so the
    # four points are one variable and the slope between them is the price of
    # L2 capacity in cycles per doubling.
    Probe(
        "l2_2x",
        {"-gpgpu_cache:dl2": "S:256:128:16,L:B:m:L:X,A:192:4,32:0,32"},
        "What does doubling L2 capacity (256 KB -> 512 KB per sub-partition) "
        "buy, at stock associativity, MSHRs and ROP latency?",
        "L2 capacity is worth buying and the ladder gives the exchange rate; "
        "read with starve_l2_half and starve_l2_quarter, not alone.",
        "Block the loop so the working set fits the L2 you already have.",
    ),
    # The three instruction-side caches, upward. Each is 4x on associativity
    # alone so set indexing is unchanged, and each needs the perfect-cache flag
    # off before the geometry means anything -- hence the shared reference.
    # A flat result here *plus* a flat starve twin is what licenses reclaiming
    # the structure; either one alone does not.
    Probe(
        "big_icache",
        {"-gpgpu_perfect_inst_const_cache": "0",
         "-gpgpu_cache:il1": "N:64:128:64,L:R:f:N:L,S:2:48,4"},
        "What does 4x the instruction cache (128 KB -> 512 KB) buy?",
        "The unrolled mainloop does not fit the I-cache -- a real risk here, "
        "because tile-size search is the one search that inflates code size.",
        "Shorten the mainloop: less unrolling, or a smaller tile.",
        reference="perfect_ic_off",
    ),
    Probe(
        "big_const",
        {"-gpgpu_perfect_inst_const_cache": "0",
         "-gpgpu_const_cache:l1": "N:128:64:32,L:R:f:N:L,S:2:64,4"},
        "What does 4x the constant cache (64 KB -> 256 KB) buy?",
        "Constant traffic is on the critical path.",
        "Move constants into registers or shared memory.",
        reference="perfect_ic_off",
    ),
    Probe(
        "big_tex",
        {"-gpgpu_tex_cache:l1": "N:4:128:1024,L:R:m:N:L,T:512:8,128:2"},
        "What does 4x the texture cache (128 KB -> 512 KB) buy?",
        "Something uses the texture path after all -- which would contradict "
        "TEX_ACC = 0 in all 161 power reports and should be disbelieved before "
        "it is acted on.",
        "none",
    ),
    Probe(
        "more_regbanks",
        {"-gpgpu_num_reg_banks": "64"},
        "What does doubling register-file banks (32 -> 64) buy?",
        "Operand delivery is bank-conflicted; this is cheap area next to the "
        "register file itself.",
        "Renumber the accumulator registers so the operands of one instruction "
        "land in different banks.",
    ),
    # The legal upward twin of ``starve_subcore_half``, and the only way to ask
    # for more functional-unit throughput that GPGPU-Sim will actually honour:
    # under -gpgpu_sub_core_model 1 each scheduler owns exactly one pipeline of
    # each class, so raising -gpgpu_num_sp_units above -gpgpu_num_sched_per_core
    # leaves the extra units unreachable (the cap that made fast_alu inert).
    # Doubling the sub-cores doubles issue width and the unit count together.
    # Every width that shader.cc asserts against the scheduler count moves with
    # it -- pipeline_widths (index 10 is EX_WB, 8 -> 16), the unit counts, and
    # all four -specialized_unit_N strings, whose field order is
    # <on>,<num_units>,<latency>,<id_oc_w>,<oc_ex_w>,<NAME> with the stock A100
    # latencies deliberately unchanged.
    #
    # MEASURED 2026-09-23, and it settles a question the notes had wrong. An
    # earlier 8-sub-core recipe carried -gpgpu_num_reg_banks 64 and
    # -gpgpu_operand_collector_num_units_gen 16, and I read that as *required*
    # -- shader.cc:5170 asserts banks % num_sched == 0 and 5171 asserts
    # num_sched <= n_collector_units. Both already hold at stock: 32 % 8 == 0
    # and 8 <= 8. They were sufficient, not necessary. Run both ways on the
    # same trace: stock banks/collectors gives **99415 cycles (+8.68%)** and
    # the 64-bank/16-collector version gives **99901 (+8.24%)** -- the added
    # register bandwidth buys nothing and the 0.44% between them is inside the
    # 1.00% noise floor. The stock-bank version is kept because it moves issue
    # width alone, with no register-bandwidth confound to disentangle from
    # ``more_regbanks``.
    Probe(
        "2x_subcore",
        {"-gpgpu_num_sched_per_core": "8",
         "-gpgpu_pipeline_widths": "8,8,8,8,8,8,8,8,8,8,16,8,8",
         "-gpgpu_num_sp_units": "8",
         "-gpgpu_num_int_units": "8",
         "-gpgpu_num_sfu_units": "8",
         "-gpgpu_num_dp_units": "8",
         "-gpgpu_num_tensor_core_units": "8"},
        "What does doubling the sub-cores (4 -> 8 schedulers, and one pipeline "
        "of every class each) buy?",
        "Issue width is the constraint, and the FP32/INT arrays are worth "
        "their area; read against starve_subcore_half for the exchange rate.",
        "Raise ILP inside each warp so the existing 4 schedulers stay fed.",
        trace_knobs={"-specialized_unit_1": "1,8,4,8,8,BRA",
                     "-specialized_unit_2": "1,8,200,8,8,TEX",
                     "-specialized_unit_3": "1,8,12,8,8,TENSOR",
                     "-specialized_unit_4": "1,8,4,8,8,UDP"},
    ),
    # ``no_launch_lat`` (-gpgpu_kernel_launch_latency 0, -gpgpu_TB_launch_latency
    # 0) was removed on 2026-09-23. It ranked 4th at 512^3 (+5.06%) and 1st at
    # 256^3 (+17.91%), which is exactly the problem: it measures a fixed cost
    # that shrinks as the benchmark grows, so it competes for the top of the
    # ranking against microarchitecture without being a structure anyone can
    # buy, sell, or price in mm^2. It has no starvation twin either -- there is
    # no "less launch latency" to surrender -- so it cannot enter the iso-area
    # trade this catalogue exists to serve. Its one finding (the benchmark must
    # be >= 512^3 before launch overhead stops dominating) is already recorded
    # and acted on; see testkernel/d6_size.py, which keeps the knobs inline.
)

# ======================================================================
# Starve probes -- the other half of the gradient
# ======================================================================
#
# The idealization probes above answer "does more of X help?". They do NOT
# answer "does less of X hurt?", and the two are not the same question.
# ``ideal_l1`` reads 0.00% against baseline at every shape measured here, which
# says only that baseline's L1 is already big enough -- at 512^3 the same probe
# is +33.98% against ``adaptive_off``, i.e. the resource is load-bearing and
# fully supplied. Concluding "L1 is dead, sell it" from the upward probe alone
# would have removed a resource the kernel depends on.
#
# So an iso-area reallocation needs a second wave that shrinks each resource
# and measures what it costs. A resource is free to surrender only if *both*
# probes are flat: enlarging it buys nothing AND starving it costs nothing.
#
# Each probe moves exactly one structure, and each one maps onto a component
# McPAT prices, so the cycles cost can be divided by the mm^2 freed.
STARVE_PROBES: tuple[Probe, ...] = (
    # 4 KB and 8 KB are rejected by CACTI ("no valid data array organizations")
    # before any area comes out, so 16 KB is the floor the area model will
    # price. Capacity is in any case the wrong lever here: 128 KB -> 16 KB only
    # moves McPAT's texture-cache area 13.94 -> 12.32 mm^2, because the area is
    # dominated by the port and bank structure, not the array. The real
    # reallocation is to delete the unit, and this probe exists to establish
    # that deleting it is safe -- the counters already say TC_H = TC_M = 0.
    #
    # MEASURED 2026-09-18 at the *previous* 4 KB setting, which CACTI rejects
    # for area but the simulator runs fine: 0.00% on both tiles. The 4 KB
    # result is the stronger one and it is the number to quote; the 16 KB value
    # here only exists so the same probe can also be priced. Corroborated
    # across all 161 power reports collected so far -- TEX_ACC is zero in every
    # single one, for every tile and every hardware configuration, so the
    # texture path is dead for this workload class rather than for one kernel.
    Probe(
        "starve_tex",
        {"-gpgpu_tex_cache:l1": "N:4:128:32,L:R:m:N:L,T:512:8,128:2"},
        "What does cutting the texture cache from 128 KB to 16 KB cost?",
        "The texture path is load-bearing after all -- do not reclaim it.",
        "none",
        direction="down",
        hint_when="large",
    ),
    # Both of these carried their own disclaimer until 2026-09-23 -- "can only
    # bind once -gpgpu_perfect_inst_const_cache is turned off" -- and the flag
    # was never turned off, so both were measuring the flag and not the cut.
    # They now clear it themselves and subtract the ``perfect_ic_off`` control,
    # which is what makes a flat result here mean "the structure is dead"
    # rather than "the structure was bypassed".
    Probe(
        "starve_const",
        {"-gpgpu_perfect_inst_const_cache": "0",
         "-gpgpu_const_cache:l1": "N:16:64:8,L:R:f:N:L,S:2:64,4"},
        "What does cutting the constant cache from 64 KB to 8 KB cost?",
        "Constant traffic matters and the constant cache is load-bearing.",
        "none",
        reference="perfect_ic_off",
        direction="down",
    ),
    Probe(
        "starve_icache",
        {"-gpgpu_perfect_inst_const_cache": "0",
         "-gpgpu_cache:il1": "N:8:128:16,L:R:f:N:L,S:2:48,4"},
        "What does cutting the instruction cache from 128 KB to 16 KB cost?",
        "The instruction working set does not fit -- an unrolled mainloop is "
        "large and this is the one structure a tile-size search inflates.",
        "none",
        reference="perfect_ic_off",
        direction="down",
    ),
    # The downward half of the L2 ladder, and the gap that mattered most:
    # l2_capacity is the largest *actionable* headroom in the sweep (+9.72% at
    # 512^3, against +18.19% for 2x_sm, which is not actionable at constant
    # area), and until now nothing measured what giving L2 back costs. Without
    # these two rungs the biggest lever in the catalogue could be pointed at
    # but not priced: an iso-area design has to pay for a bigger L2 by
    # shrinking something, and "something" may well be the L2 of a different
    # partition. Sets only, at stock associativity, MSHRs and ROP latency, so
    # these sit on the same single-variable axis as ``l2_2x``.
    #
    # Expect asymmetry rather than a straight line. The counters say the L1D
    # write path is 100% MISS_QUEUE_FULL with 0% MSHR, i.e. L2 pressure is
    # write-side backpressure, and a structure that is already overcommitted
    # usually degrades faster on the way down than it improves on the way up.
    # If it does, the honest reading is that L2 is the wrong currency to pay
    # with, not that it is the wrong thing to buy.
    Probe(
        "starve_l2_half",
        {"-gpgpu_cache:dl2": "S:64:128:16,L:B:m:L:X,A:192:4,32:0,32"},
        "What does halving the L2 (256 KB -> 128 KB per sub-partition) cost?",
        "L2 capacity is load-bearing on the way down as well as the way up, "
        "so it cannot be the resource that funds anything else.",
        "none",
        direction="down",
    ),
    Probe(
        "starve_l2_quarter",
        {"-gpgpu_cache:dl2": "S:32:128:16,L:B:m:L:X,A:192:4,32:0,32"},
        "What does quartering the L2 (256 KB -> 64 KB per sub-partition) cost?",
        "L2 capacity is load-bearing well below the shipped size; the fourth "
        "rung of the ladder, and the one that shows whether the cost is linear "
        "in capacity or falls off a cliff.",
        "none",
        direction="down",
    ),
    Probe(
        "starve_l1_half",
        {"-gpgpu_adaptive_cache_config": "0",
         "-gpgpu_cache:dl1": "S:4:128:32,L:T:m:L:L,A:512:64,16:0,32"},
        "What does halving the L1 data cache (32 KB -> 16 KB) cost?",
        "L1 capacity is load-bearing; the upward probe was flat only because "
        "baseline already supplies enough.",
        "none",
        reference="adaptive_off",
        direction="down",
    ),
    Probe(
        "starve_l1_quarter",
        {"-gpgpu_adaptive_cache_config": "0",
         "-gpgpu_cache:dl1": "S:4:128:16,L:T:m:L:L,A:512:64,16:0,32"},
        "What does quartering the L1 data cache (32 KB -> 8 KB) cost?",
        "L1 capacity is load-bearing below 16 KB.",
        "none",
        reference="adaptive_off",
        direction="down",
    ),
    Probe(
        "starve_mshr",
        {"-gpgpu_adaptive_cache_config": "0",
         "-gpgpu_cache:dl1": "S:4:128:64,L:T:m:L:L,A:128:16,16:0,32"},
        "What does cutting L1 miss-handling capacity 4x cost?",
        "Memory-level parallelism is throttled at the L1 -- MSHR entries are "
        "cheap area but they are not free.",
        "none",
        reference="adaptive_off",
        direction="down",
    ),
    # 100 KB was the first try and it SIGABRTs before cycle 0: the trace header
    # says the stock CUTLASS tile asks for -shmem = 147456 per CTA, so any pool
    # below 144 KB makes the kernel unlaunchable rather than slow. The floor is
    # therefore set by the tile, which is exactly the coupling this study is
    # looking for -- shared memory is not a free resource to sell, it is a
    # resource the software has already spent.
    Probe(
        "starve_shmem",
        {"-gpgpu_shmem_size": "147456", "-gpgpu_shmem_sizeDefault": "147456",
         "-gpgpu_shmem_option": "0,8,16,32,64,144"},
        "What does cutting shared memory from 164 KB to the 144 KB the tile "
        "actually asks for cost?",
        "Shared memory caps CTAs per SM; this is the knob most coupled to the "
        "software tile choice, so read it per tile, not once.",
        "none",
        direction="down",
    ),
    # Same story as shared memory, tighter: 256 threads x 244 registers = 62464
    # of the 65536 available, so the stock tile occupies 95.3% of the register
    # file and a halved RF aborts at launch. 63488 is the largest cut that
    # still fits, and the point of running it is to put a number on how little
    # there is to reclaim rather than to assert it.
    Probe(
        "starve_regfile",
        {"-gpgpu_shader_registers": "63488",
         "-gpgpu_registers_per_block": "63488"},
        "What does cutting the register file to the 62464 registers the tile "
        "actually allocates cost?",
        "Occupancy is register-limited; the tile sizes itself to the RF, so "
        "the RF cannot be sold without resizing the tile.",
        "none",
        direction="down",
    ),
    Probe(
        "starve_regbanks",
        {"-gpgpu_num_reg_banks": "8"},
        "What does cutting register-file banks 32 -> 8 cost?",
        "Operand delivery, not register capacity, is the constraint.",
        "none",
        direction="down",
    ),
    # MEASURED, 2026-09-18: this probe DEADLOCKS on both tiles (last writeback
    # ~86k cycles before the detector fired). It is not a bug in the config, it
    # is the sub-core model. ``-gpgpu_sub_core_model 1`` with
    # ``-gpgpu_num_sched_per_core 4`` gives each scheduler exclusive use of one
    # pipeline of each type, so a unit count below 4 leaves schedulers 1..3 with
    # no port for that op class -- forever. A count < num_sched is therefore
    # unsimulatable, not slow.
    #
    # That makes the crash a result rather than a failure. ``starve_sfu``,
    # ``starve_dp`` and ``starve_tensor`` apply the same illegal cut and run to
    # completion, because a scheduler that never issues that op class never
    # waits on the missing port. Deadlock here == every sub-core needs FP32 and
    # INT; clean completion there == the workload does not touch SFU or DP at
    # all. The only legal way to reclaim functional-unit area is to remove a
    # whole sub-core: see ``starve_subcore_half``.
    #
    # Kept at the illegal value deliberately so the deadlock is reproducible.
    Probe(
        "starve_fpu",
        {"-gpgpu_num_sp_units": "1"},
        "What does cutting the FP32 pipelines 4 -> 1 cost? (expect DEADLOCK)",
        "The FP32 array is load-bearing. A flat result means this GEMM runs in "
        "the tensor cores and the SP array is mostly dark silicon.",
        "none",
        direction="down",
    ),
    Probe(
        "starve_int",
        {"-gpgpu_num_int_units": "1"},
        "What does cutting the INT pipelines 4 -> 1 cost? (expect DEADLOCK)",
        "Address arithmetic and loop bookkeeping are on the critical path -- "
        "expect this, INT_ACC dominates every other accumulator in the report.",
        "none",
        direction="down",
    ),
    Probe(
        "starve_dp",
        {"-gpgpu_num_dp_units": "0"},
        "What does removing the FP64 pipelines entirely cost?",
        "Something issues DP; check DP_ACC before believing it.",
        "none",
        direction="down",
    ),
    Probe(
        "starve_sfu",
        {"-gpgpu_num_sfu_units": "1"},
        "What does cutting the SFU pipelines 4 -> 1 cost?",
        "Transcendentals are on the critical path.",
        "none",
        direction="down",
    ),
    # Halved by *rate*, not by unit count. -gpgpu_num_tensor_core_units 2 --
    # what this probe asked for until 2026-09-22 -- cuts a pipeline HMMA never
    # enters, so it was guaranteed to read 0.00% and be filed as "tensor issue
    # is not the constraint". It is the constraint: 17.8% of baseline cycles,
    # fully serial. Doubling II from the stock 8 to 16 halves the aggregate
    # rate the way cutting real units would, and costs 16.9% on the base tile.
    Probe(
        "starve_tensor",
        {},
        "What does halving the tensor issue rate (II 8 -> 16) cost?",
        "Tensor issue rate is the constraint -- the expected result for a "
        "TF32 GEMM, and the control that says the starve wave can detect a "
        "resource that actually matters.",
        "none",
        direction="down",
        trace_knobs={"-trace_opcode_latency_initiation_spec_op_3": "12,16"},
    ),
    # OC_EX_MEM cannot go below the scheduler count: -gpgpu_sub_core_model 1
    # divides each width by -gpgpu_num_sched_per_core, and a width of 2 across
    # 4 schedulers gives 0 lanes each, which trips
    # `reg_id < regs.size()` in abstract_hardware_model.h. Only EX_WB (index
    # 10, 8 -> 4) has slack, so this probe narrows writeback alone.
    Probe(
        "starve_lsu",
        {"-gpgpu_pipeline_widths": "4,4,4,4,4,4,4,4,4,4,4,4,4"},
        "What does halving writeback width (EX_WB 8 -> 4) cost?",
        "Memory issue width is structural and load-bearing.",
        "none",
        direction="down",
    ),
    # The legal counterpart to the illegal ``starve_fpu``/``starve_int`` cuts:
    # halve the sub-core count itself, so every scheduler still owns a full set
    # of pipelines. This is the only knob path to functional-unit area under
    # ``-gpgpu_sub_core_model 1``. It halves issue width along with the units,
    # so a flat result would say the SM is not issue-limited at all.
    #
    # Changing -gpgpu_num_sched_per_core alone ABORTS at shader.cc:130: the
    # ID_OC_* register-set sizes come from -gpgpu_pipeline_widths, not from the
    # unit counts, and the sub-core model asserts each equals the scheduler
    # count. The assert fires for every *enabled specialized unit* too
    # (shader.cc:145), which is why all four -specialized_unit_N strings move
    # here and not just the tensor one. Field order is
    # <on>,<num_units>,<latency>,<id_oc_w>,<oc_ex_w>,<NAME>; latencies are the
    # stock A100 values and are deliberately unchanged, so this probe cuts
    # width alone. pipeline_widths index 10 is EX_WB (8 -> 4); 11 and 12 are
    # ID_OC/OC_EX_TENSOR_CORE. Verified as the mirror of a working 8-sub-core
    # build. Register banks (32) and collector units (8) still satisfy
    # shader.cc:5170-5172 at 2 schedulers and are left alone so this measures
    # issue width rather than register bandwidth.
    Probe(
        "starve_subcore_half",
        {"-gpgpu_num_sched_per_core": "2",
         "-gpgpu_pipeline_widths": "2,2,2,2,2,2,2,2,2,2,4,2,2",
         "-gpgpu_num_sp_units": "2",
         "-gpgpu_num_int_units": "2",
         "-gpgpu_num_sfu_units": "2",
         "-gpgpu_num_dp_units": "2",
         "-gpgpu_num_tensor_core_units": "2"},
        "What does halving the sub-cores (4 -> 2 schedulers and one pipeline "
        "of every class each) cost?",
        "Issue width is on the critical path and this is expensive; it is the "
        "price of the only legal FP32/INT area reclamation.",
        "none",
        direction="down",
        trace_knobs={"-specialized_unit_1": "1,2,4,2,2,BRA",
                     "-specialized_unit_2": "1,2,200,2,2,TEX",
                     "-specialized_unit_3": "1,2,12,2,2,TENSOR",
                     "-specialized_unit_4": "1,2,4,2,2,UDP"},
    ),
    Probe(
        "starve_icnt",
        {"-icnt_flit_size": "20"},
        "What does halving the on-chip network flit size cost?",
        "NoC width is load-bearing -- read with wide_icnt, which was +1.63%.",
        "none",
        direction="down",
    ),
)

PROBE_BY_NAME = {p.name: p for p in IDEALIZATION_PROBES + STARVE_PROBES}


def run_probes(
    node,
    kernelslist: str,
    *,
    probes: tuple[Probe, ...] = IDEALIZATION_PROBES,
    arch: str = "SM80_A100",
    outroot: str | None = None,
    gather: bool = True,
    **run_kwargs,
):
    """Fan out every probe plus a baseline against one trace.

    All of them are independent CPU simulations on a shared trace, so this is
    one wave of ~11 concurrent 3-minute runs: the whole diagnostic costs about
    as much wall-clock as a single simulation on a box with 11 free cores.
    """
    candidates = {"baseline": {}}
    candidates.update({p.name: dict(p.knobs) for p in probes})
    trace_candidates = {p.name: dict(p.trace_knobs) for p in probes if p.trace_knobs}
    return sweep_configs(
        node,
        kernelslist=kernelslist,
        candidates=candidates,
        arch=arch,
        outroot=outroot,
        trace_candidates=trace_candidates,
        gather=gather,
        **run_kwargs,
    )


@dataclass
class Headroom:
    """One probe's answer."""

    name: str
    cycles: float | None
    speedup: float | None  # baseline_cycles / probe_cycles; >1 means headroom
    headroom_pct: float | None
    inert: bool = False  # knob did not land -- experiment did not run
    no_headroom: bool = False  # knob landed, cycles unchanged -- a real finding
    reference: str = ""  # the control this probe's control_pct is measured against
    control_pct: float | None = None
    """Headroom against the probe's control instead of against baseline.

    These answer different questions and the report needs both. ``headroom_pct``
    (vs baseline) is "how much could we win over the config we would actually
    ship" -- that is what earns a sweep slot. ``control_pct`` (vs the control)
    is "is this resource on the critical path at all" -- that is diagnosis.

    Measured at 512^3: ideal_l1 is 0.00% vs baseline and +33.98% vs
    adaptive_off. L1 capacity matters enormously, *and* there is nothing to win,
    because baseline's adaptive sizing already delivers it. Reporting only the
    control number put ideal_l1 in hw_targets and would have spent a sweep slot
    on a knob worth zero.
    """
    unapplied: tuple[str, ...] = ()  # knobs whose effective value != requested
    ok: bool = True
    error: str = ""  # first line of stderr when the simulator aborted
    question: str = ""
    reads_as: str = ""
    kernel_hint: str = ""
    direction: str = "up"  # "up" = idealization (expect a speedup), "down" = removal
    hint_when: str = "large"  # whether kernel_hint fires on a large or a flat result


def wave_quantization(stats: dict, n_sm: int) -> dict:
    """Decide whether the SM-count probes are measuring scalability or a tail.

    ``2x_sm`` cannot distinguish "the kernel is throughput-scalable" from "the
    grid is slightly larger than the machine, so there is a nearly-empty final
    round". Both show a large speedup; only the first is an argument for buying
    silicon, and acting on the second is the mistake ``2x_sm``'s own text calls
    the most expensive one this loop can make.

    Measured at 512^3 SGEMM: grid (32,32,1) = 1024 CTAs, CTA/core = 8, 108 SMs
    = 864 resident slots. 1024 is 1.19x capacity, so the run is two rounds at
    59% average utilisation and 2x_sm reports 55.49%. Doubling the SMs makes
    the whole grid resident at once (1024 <= 1728) -- and occupancy *falls*
    from 89.06% to 59.23%, which is 1024/1728 to three figures. The extra area
    is idle; it only absorbed the tail.
    """
    ctas = stats.get("issued_ctas")
    per_core = stats.get("cta_per_core")
    if not ctas or not per_core or not n_sm:
        return {}
    capacity = per_core * n_sm
    rounds = math.ceil(ctas / capacity)
    util = ctas / (rounds * capacity)
    confounded = rounds <= 2 and util < 0.8
    return {
        "ctas": int(ctas),
        "cta_per_core": int(per_core),
        "n_sm": int(n_sm),
        "capacity": int(capacity),
        "rounds": rounds,
        "utilization": util,
        "confounded": confounded,
        "note": (
            f"{int(ctas)} CTAs against {int(capacity)} resident slots "
            f"({int(n_sm)} SM x {int(per_core)}) = {rounds} rounds at "
            f"{util:.0%} utilisation. The SM-count probes are measuring this "
            "tail, not throughput scalability: more SMs absorb the partial "
            "round. Reshape the grid first -- it is free -- and re-probe "
            "before reading 2x_sm as an argument for more silicon."
            if confounded
            else f"{int(ctas)} CTAs / {int(capacity)} slots = {rounds} rounds "
            f"at {util:.0%} utilisation; SM probes are not tail-confounded."
        ),
    }


# --------------------------------------------------------------------------
# Paired gradient: what each structure is worth in BOTH directions.
# --------------------------------------------------------------------------
# An upward-only sweep answers "does more of X help?" and never "does less of
# X hurt?", and the second question is the one an area-constrained design has
# to answer -- you cannot fund a structure you want to buy without pricing the
# structure you intend to sell. Worse, the two are not mirror images. Measured
# 2026-09-23 at 4096x4096x64: doubling L2 buys +9.19%, halving it costs
# -13.69%, quartering it costs -22.29%. The curve is convex, so the upward
# number systematically *understates* what the structure is worth, and a loop
# reading only the upward half will happily propose paying for its top lever
# by selling the same structure somewhere else.
#
# The pairing is declared here rather than pattern-matched out of probe names
# afterwards, because the two halves of a pair are not always the same knob at
# different multipliers (``ideal_l1`` is an idealization, ``starve_l1_half`` is
# a capacity cut) and no naming rule recovers that.
PROBE_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("L2 capacity", "l2_2x", "starve_l2_half"),
    ("L2 capacity, 4x span", "l2_2x", "starve_l2_quarter"),
    ("L1D capacity", "ideal_l1", "starve_l1_half"),
    ("L1D capacity, 4x span", "big_l1", "starve_l1_quarter"),
    ("L1 MSHRs", "many_mshr", "starve_mshr"),
    ("Shared memory", "big_shmem", "starve_shmem"),
    ("Register file", "big_regfile", "starve_regfile"),
    ("Register banks", "more_regbanks", "starve_regbanks"),
    ("Instruction cache", "big_icache", "starve_icache"),
    ("Constant cache", "big_const", "starve_const"),
    ("Texture cache", "big_tex", "starve_tex"),
    ("Sub-cores / issue width", "2x_subcore", "starve_subcore_half"),
    ("Tensor issue rate", "fast_tensor", "starve_tensor"),
    ("Interconnect width", "wide_icnt", "starve_icnt"),
    ("Writeback width", "wide_lsu", "starve_lsu"),
    ("SM count", "2x_sm", "half_sm"),
)


def _probe_status(r: "Headroom | None") -> str:
    """Why a probe produced no number, in one word the table can print."""
    if r is None:
        return "not run"
    if r.inert:
        return "INERT"
    if not r.ok:
        hay = (r.error or "").lower()
        return ("DEADLOCK" if "deadlock" in hay
                else "ABORT" if "assertion" in hay else "FAILED")
    return ""


def _signed_pct(r: "Headroom | None") -> float | None:
    """The probe's own effect: against its control when it has one."""
    if r is None or not r.ok or r.inert or r.headroom_pct is None:
        return None
    return r.control_pct if r.control_pct is not None else r.headroom_pct


def pair_gradient(rows: list, *, pairs: tuple = PROBE_PAIRS) -> dict:
    """Join up-probes to down-probes and rank both sides of the exchange.

    ``gain_pct`` is cycles saved by enlarging. ``cost_pct`` is cycles LOST by
    shrinking, **printed positive**, so the two columns are directly
    comparable without the reader flipping a sign in their head.

    Three orderings come out, and they answer three different questions:

    * ``buy_ranked``   -- largest gain first: where to spend effort and area.
    * ``sell_ranked``  -- smallest cost first: what to shrink to pay for it.
    * ``free``         -- both directions inside the noise floor: structures
      that are neither a lever nor a loss, i.e. area available for nothing.

    ``ratio`` is cost/gain. Above 1.0 the structure is convex -- it costs more
    to give up than it pays to acquire -- which disqualifies it as currency
    even when its upward number is large enough to look like a target.
    """
    by = {r.name: r for r in rows}
    out = []
    for label, up, dn in pairs:
        u, d = by.get(up), by.get(dn)
        gain = _signed_pct(u)
        raw = _signed_pct(d)
        # A starve probe that loses cycles reads as negative headroom; the
        # cost column is its negation.
        cost = -raw if raw is not None else None
        ratio = (cost / gain) if (gain and cost is not None and abs(gain) > 1e-9) else None
        if gain is None or cost is None:
            verdict = "incomplete"
        elif abs(gain) < NOISE_PCT and abs(cost) < NOISE_PCT:
            verdict = "free"
        elif cost >= HW_TARGET_PCT and gain >= HW_TARGET_PCT:
            verdict = "lever, not currency" if (ratio or 0) > 1.0 else "lever"
        elif cost >= HW_TARGET_PCT:
            verdict = "load-bearing only downward"
        elif gain >= HW_TARGET_PCT:
            verdict = "buy-side lever, cheap to give back"
        else:
            verdict = "minor"
        out.append({
            "structure": label, "up": up, "down": dn,
            "gain_pct": gain, "cost_pct": cost, "ratio": ratio,
            "up_status": _probe_status(u), "down_status": _probe_status(d),
            "verdict": verdict,
        })
    done = [r for r in out if r["verdict"] != "incomplete"]
    return {
        "rows": out,
        "buy_ranked": sorted(done, key=lambda r: -(r["gain_pct"] or 0.0)),
        "sell_ranked": sorted(done, key=lambda r: (r["cost_pct"] or 0.0)),
        "free": [r["structure"] for r in done if r["verdict"] == "free"],
    }


def probe_report(
    results: dict[str, AccelSimRunResult],
    *,
    baseline: str = "baseline",
    probes: tuple[Probe, ...] = IDEALIZATION_PROBES,
    inert_threshold: float = 0.0,
) -> dict:
    """Turn a probe sweep into a ranked headroom table.

    ``headroom_pct`` is the percentage of baseline cycles that the idealization
    removes. It is an *upper bound* on what any real change to that resource can
    buy, which is the property that makes it usable as a budget: a probe showing
    3% headroom means no amount of sweeping that knob family is worth a slot.

    ``inert`` is reported separately from "no headroom". They look identical in
    a cycle count and mean opposite things -- one says the resource is not the
    bottleneck, the other says the experiment did not run. Conflating them is
    how a loop ends up re-proposing dead knobs forever.
    """
    base = results.get(baseline)
    if base is None or not base.ok:
        return {"error": f"baseline '{baseline}' missing or failed"}
    base_cycles = base.stats["cycles"]

    rows: list[Headroom] = []
    for p in probes:
        r = results.get(p.name)
        if r is None:
            continue
        if not r.ok:
            # Accel-Sim aborts on an illegal config with the assertion on
            # stderr. Naming the probe without the assertion makes a fixable
            # config error look like an infrastructure flake.
            hay = (r.stderr_tail or "") + "\n" + (r.stdout_tail or "")
            err = next(
                (
                    ln.strip()
                    for ln in hay.splitlines()
                    if "Assertion" in ln or "deadlock detected" in ln
                ),
                "",
            ) or next(
                (ln.strip() for ln in hay.splitlines()
                 if ln.strip() and not ln.startswith("/") and "===" not in ln),
                "",
            )
            rows.append(Headroom(
                p.name, None, None, None, ok=False, question=p.question,
                error=err,
            ))
            continue
        c = r.stats["cycles"]
        # A probe that had to disable a confounding feature is measured against
        # that control, not against baseline, so the reported headroom is the
        # knob's own effect rather than the knob plus the disabling.
        ref_cycles = base_cycles
        if p.reference:
            ref = results.get(p.reference)
            if ref is not None and ref.ok and ref.stats.get("cycles"):
                ref_cycles = ref.stats["cycles"]
        flat = abs(c - ref_cycles) <= inert_threshold
        # A zero delta has two opposite causes. Read the knobs back off the
        # config the simulator actually consumed: if the requested value is not
        # there, the experiment never ran (inert); if it is, the resource simply
        # is not on the critical path (no_headroom), which is a real answer and
        # belongs in the report rather than in a list of broken probes.
        unapplied = tuple(
            k for k, v in {**p.knobs, **p.trace_knobs}.items()
            if str((r.trace_knobs if k in p.trace_knobs else r.knobs).get(k, "")).strip()
            != str(v).strip()
        )
        rows.append(
            Headroom(
                name=p.name,
                cycles=c,
                speedup=base_cycles / c if c else None,
                headroom_pct=100.0 * (base_cycles - c) / base_cycles,
                control_pct=(
                    100.0 * (ref_cycles - c) / ref_cycles
                    if ref_cycles != base_cycles
                    else None
                ),
                reference=p.reference if ref_cycles != base_cycles else "",
                inert=bool(unapplied),
                no_headroom=flat and not unapplied,
                unapplied=unapplied,
                question=p.question,
                reads_as=p.if_large,
                kernel_hint=p.kernel_hint,
                direction=p.direction,
                hint_when=p.hint_when,
            )
        )

    # The SM count the baseline actually ran with, read back off the effective
    # config rather than assumed, so an arch override cannot silently break it.
    try:
        n_sm = int(float(base.knobs.get("-gpgpu_n_clusters", 0))) * int(
            float(base.knobs.get("-gpgpu_n_cores_per_cluster", 1))
        )
    except (TypeError, ValueError):
        n_sm = 0
    quant = wave_quantization(base.stats, n_sm)

    ranked = sorted(
        [r for r in rows if r.headroom_pct is not None and not r.inert],
        key=lambda r: -(r.headroom_pct or 0.0),
    )
    return {
        "baseline_cycles": base_cycles,
        "stall_mix": base.stall_mix,
        "ranked": ranked,
        # Both directions of every structure that has a probe on each side.
        # Empty-ish when the caller passed an upward-only probe set: every
        # row then reads "incomplete", which is the honest rendering of a
        # one-sided sweep rather than a silent omission.
        "pairs": pair_gradient(rows),
        "inert": [r.name for r in rows if r.inert],
        "unapplied": {r.name: list(r.unapplied) for r in rows if r.inert},
        "no_headroom": [r.name for r in rows if r.no_headroom],
        "failed": [r.name for r in rows if not r.ok],
        "failures": {r.name: r.error for r in rows if not r.ok},
        # A backwards result is excluded: it is evidence the resource is *not*
        # the constraint, so spending a sweep slot on it is exactly wrong.
        "quantization": quant,
        "hw_targets": [
            r.name for r in ranked
            if (r.headroom_pct or 0) >= HW_TARGET_PCT and r.direction == "up"
            # A tail-confounded SM probe is not a hardware target: the speedup
            # comes from absorbing a partial round, which reshaping the grid
            # does for free.
            and not (quant.get("confounded") and r.name in ("2x_sm", "half_sm"))
        ],
        "kernel_hints": [
            {"probe": r.name, "headroom_pct": round(r.headroom_pct or 0, 2), "hint": r.kernel_hint}
            for r in ranked
            if r.kernel_hint != "none"
            and (
                (r.headroom_pct or 0) >= HW_TARGET_PCT
                if r.hint_when == "large"
                else abs(r.headroom_pct or 0) < HW_TARGET_PCT
            )
        ],
    }


def format_probe_report(rep: dict) -> str:
    """Render a probe report for an LLM prompt or a terminal."""
    if "error" in rep:
        return f"probe report unavailable: {rep['error']}"
    lines = [f"baseline = {rep['baseline_cycles']:.0f} cycles"]
    sm = rep.get("stall_mix") or {}
    if sm:
        lines.append(
            "scheduler cycles: "
            + ", ".join(f"{k} {100*v:.1f}%" for k, v in sm.items())
            + "   (idle=front-end, dependency=waiting on memory, structural=no free unit)"
        )
    q = rep.get("quantization") or {}
    if q:
        lines.append(
            ("!! CTA QUANTIZATION: " if q.get("confounded") else "cta supply: ")
            + q["note"]
        )
    lines.append("")
    lines.append(f"{'probe':<16}{'cycles':>10}{'headroom':>11}   reads as")
    for r in rep["ranked"]:
        pct = r.headroom_pct or 0.0
        # Gate on magnitude AND on sign against the probe's declared direction.
        # Magnitude alone is not enough: half_sm at -35% is a real result whose
        # text is written for the negative direction, but zero_dram_lat at -6.85%
        # is an idealization that came out backwards, and printing "DRAM latency
        # is the constraint" beside it is the same contradiction in a new guise.
        mag = abs(pct)
        expected = pct >= 0 if r.direction == "up" else pct <= 0
        if r.direction == "control":
            # A control is never a finding at any magnitude; it exists to be
            # subtracted. Labelling it BACKWARDS at -51% was nonsense.
            reads = r.reads_as
        elif mag >= HW_TARGET_PCT and expected:
            reads = r.reads_as
        elif mag >= HW_TARGET_PCT:
            reads = (
                "BACKWARDS: idealizing this resource made the kernel SLOWER. It "
                "is not the constraint -- forcing it changes request timing and "
                "arbitration. Not a hardware target."
                if r.direction == "up"
                else "BACKWARDS: removing this resource made the kernel FASTER. "
                "Suspect a scheduling artifact, not an area saving."
            )
        elif mag >= NOISE_PCT:
            reads = f"small but real; below the {HW_TARGET_PCT:.0f}% bar for a sweep slot"
        else:
            reads = (
                f"within run-to-run noise (~{NOISE_PCT:.0f}%) -- this resource is "
                "not on the critical path"
            )
        if r.control_pct is not None:
            # Both numbers, because they answer different questions: the first
            # is what a sweep could win, the second is whether the resource is
            # on the critical path at all.
            if abs(r.control_pct) < NOISE_PCT:
                reads = (
                    f"this knob does nothing: {r.control_pct:+.2f}% against its "
                    f"control {r.reference}. The {pct:+.2f}% against baseline is "
                    "the control's cost, not this knob's."
                )
            elif abs(pct) < NOISE_PCT and r.control_pct >= HW_TARGET_PCT:
                reads = (
                    f"on the critical path ({r.control_pct:+.2f}% vs {r.reference}) "
                    "but baseline already realises it -- nothing left to win, no "
                    "sweep slot"
                )
            else:
                reads = f"{reads}  [{r.control_pct:+.2f}% vs {r.reference}]"
        if q.get("confounded") and r.name in ("2x_sm", "half_sm"):
            reads = (
                f"TAIL-CONFOUNDED ({q['rounds']} rounds, {q['utilization']:.0%} "
                "utilisation) -- this is the partial round, not scalability. "
                "Reshape the grid and re-probe before reading it as silicon."
            )
        lines.append(f"{r.name:<16}{r.cycles:>10.0f}{pct:>10.2f}%   {reads}")
    if rep.get("no_headroom"):
        lines.append(
            "\nNO HEADROOM (knob verified applied, cycles unchanged -- this "
            f"resource is not on the critical path): {', '.join(rep['no_headroom'])}"
        )
    if rep["inert"]:
        lines.append(
            "\nINERT (requested knob is NOT in the effective config -- the "
            "experiment did not run, do not read these as results):"
        )
        for n, ks in rep.get("unapplied", {}).items():
            lines.append(f"  {n}: {', '.join(ks)}")
    if rep["failed"]:
        lines.append("\nFAILED (simulator aborted -- config is illegal, not a flake):")
        for n in rep["failed"]:
            lines.append(f"  {n}: {rep.get('failures', {}).get(n, '') or '(no stderr captured)'}")
    lines.append(format_pair_gradient(rep.get("pairs") or {}))
    return "\n".join(lines)


def format_pair_gradient(pg: dict) -> str:
    """Render the two-sided table: what to buy, and what to sell to pay for it."""
    rows = pg.get("rows") or []
    if not rows:
        return ""
    out = ["", "=== PAIRED GRADIENT (both directions) ===",
           "  gain% = cycles saved by enlarging.  cost% = cycles LOST by "
           "shrinking, printed",
           "  positive.  ratio = cost/gain; above 1.00 the structure is convex "
           "and cannot be",
           f"  used as currency.  Both inside +-{NOISE_PCT:.2f}% = free to "
           "surrender.", "",
           f"  {'structure':<26}{'gain%':>10}{'cost%':>10}{'ratio':>8}   verdict"]
    for r in rows:
        g = f"{r['gain_pct']:+.2f}" if r["gain_pct"] is not None else (r["up_status"] or "--")
        c = f"{r['cost_pct']:+.2f}" if r["cost_pct"] is not None else (r["down_status"] or "--")
        ra = f"{r['ratio']:.2f}" if r["ratio"] is not None else "--"
        out.append(f"  {r['structure']:<26}{g:>10}{c:>10}{ra:>8}   {r['verdict']}")
    buy = [r for r in pg.get("buy_ranked", []) if (r["gain_pct"] or 0) >= HW_TARGET_PCT]
    out.append("")
    out.append("  BUY SIDE (optimize these): " + (", ".join(
        f"{r['structure']} {r['gain_pct']:+.2f}%" for r in buy) or "nothing clears the bar"))
    sell = [r for r in pg.get("sell_ranked", []) if (r["cost_pct"] or 0) < HW_TARGET_PCT]
    out.append("  SELL SIDE (cheapest to shrink first; +cost = what you pay): "
               + (", ".join(f"{r['structure']} {r['cost_pct']:+.2f}%"
                            for r in sell[:8]) or "everything is load-bearing"))
    out.append("  FREE TO SURRENDER: " + (", ".join(pg.get("free") or []) or "none"))
    return "\n".join(out)
