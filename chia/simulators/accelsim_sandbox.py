"""chia.simulators.accelsim_sandbox — confinement for the kernel-writing agent.

CHIA's ``ClaudeCodeLLM`` defaults to ``dangerously_skip_permissions=True`` and
puts ``--dangerously-skip-permissions`` on the CLI (``chia/models/claude.py``).
That is fine for an agent that edits a config file. It is not fine for step 5,
where the agent's whole job is to author code that then gets compiled and
executed on the host: an unconfined agent authoring executables is the same
trust boundary as ``curl | sh``.

The confinement here is structural rather than advisory. The agent is given
**no filesystem or shell tools at all** -- only the MCP methods on
:class:`KernelForgeTool`. It submits CUDA *source text* as a tool argument; this
module decides the filename, the directory, the compiler flags, the harness it
is linked against, the inputs it runs on and whether the result is accepted. The
agent never names a path.

Three layers, in order of how much they are relied on:

1. **No tools.** ``sandboxed_kernel_llm`` sets ``dangerously_skip_permissions``
   to False and explicitly disallows every built-in tool, so ``--allowedTools``
   contains only the ``mcp__`` entries CHIA generated. This is the layer that
   matters.
2. **Fixed harness.** The candidate supplies ``launch()`` and nothing else. It
   cannot choose N, generate inputs, or write the output file, so it cannot
   make itself look fast by solving a smaller problem.
3. **Source screening.** :func:`screen_source` rejects source that has no
   business appearing in a GEMM kernel -- process spawning, file I/O, network,
   inline PTX/asm, preprocessor path escapes. This is a tripwire, not a
   security boundary: it catches an agent that misunderstood the task, and it
   would not stop one that was trying. Layer 1 is what stops that.

What this does *not* provide is OS-level isolation. nvcc runs arbitrary code at
compile time (templates, ``#pragma``), so a determined adversary with source
control has a compiler to work with. For an untrusted model, run the build and
run nodes in a container -- CHIA's cluster YAML already supports it
(``docker.run_options: "--gpus all"``) -- and treat this module as defense in
depth rather than the boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field

from chia.base.tools.ChiaTool import ChiaTool
from chia.simulators.accelsim import (
    AccelSimNode,
    opcode_histogram,
    work_preserved,
)

# Patterns with no legitimate place in a GEMM kernel or its launcher.
FORBIDDEN_SOURCE = (
    (r"\bsystem\s*\(", "process spawning"),
    (r"\bpopen\s*\(", "process spawning"),
    (r"\bexec[lv][pe]?\s*\(", "process spawning"),
    (r"\bfork\s*\(", "process spawning"),
    (r"\bfopen\s*\(|\bfreopen\s*\(|\bofstream\b|\bifstream\b", "file I/O"),
    (r"\bsocket\s*\(|\bcurl\b|<netinet/", "network"),
    (r"\bgetenv\s*\(|\bputenv\s*\(|\bsetenv\s*\(", "environment access"),
    (r"#\s*include\s*<\s*cublas|#\s*include\s*<\s*cutlass|\bcublas[A-Z]", "vendor BLAS"),
    (r"\basm\s+volatile|\b__asm__", "inline PTX"),
    (r"\bmain\s*\(", "defines main (the harness owns main)"),
    (r"\bcudaMemcpy\b|\bcudaMalloc\b|\bcudaFree\b", "allocates or copies (the harness owns memory)"),
)

REQUIRED_SYMBOL = re.compile(r"\bvoid\s+launch\s*\(")

# Includes are allowlisted, not denylisted. A denylist cannot win here: the
# earlier rule only rejected ".." traversal, so `#include "/etc/passwd"` sailed
# through to nvcc, which quoted the file's first line back in its diagnostic --
# and submit_kernel returns that diagnostic to the model. That is an arbitrary
# file read, one line per submission, through compiler error messages.
# Measured, not hypothesised: it is what the esc_include case actually did.
INCLUDE_RE = re.compile(r"#\s*include\s*([\"<])([^\">]*)[\">]")
ALLOWED_INCLUDES = frozenset(
    {
        "cuda_runtime.h", "cuda_runtime_api.h", "device_launch_parameters.h",
        "cuda_fp16.h", "cuda_bf16.h", "mma.h", "cooperative_groups.h",
        "vector_types.h",
        "math.h", "cmath", "stdint.h", "cstdint", "stddef.h", "cstddef",
        "float.h", "cfloat", "algorithm", "type_traits",
    }
)


@dataclass
class ScreenResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def screen_source(src: str) -> ScreenResult:
    """Reject CUDA source that does things a kernel candidate should not.

    A tripwire for a confused agent, not a security boundary -- see the module
    docstring. It runs before nvcc, so a rejection costs nothing.
    """
    reasons = []
    stripped = re.sub(r"//[^\n]*|/\*.*?\*/", "", src, flags=re.S)
    for pat, why in FORBIDDEN_SOURCE:
        if re.search(pat, stripped):
            reasons.append(f"{why} (matched /{pat}/)")
    for quote, target in INCLUDE_RE.findall(stripped):
        name = target.strip()
        if name not in ALLOWED_INCLUDES:
            reasons.append(
                f"include of {name!r} is not on the allowlist -- a kernel "
                "candidate may only include standard CUDA and C++ headers, "
                f"one of: {', '.join(sorted(ALLOWED_INCLUDES))}"
            )
    if not REQUIRED_SYMBOL.search(stripped):
        reasons.append("does not define `void launch(const float*, const float*, float*, int)`")
    return ScreenResult(ok=not reasons, reasons=reasons)


def sandboxed_llm(
    tools,
    *,
    model: str = "claude-sonnet-4-6",
    system_message: str = "",
    timeout_seconds: int = 900,
    **kwargs,
):
    """A ``ClaudeCodeLLM`` that can call *tools* and nothing else.

    ``dangerously_skip_permissions=False`` means CHIA does not pass
    ``--dangerously-skip-permissions``, so the CLI's ``--allowedTools`` list --
    which CHIA populates with exactly the ``mcp__<server>__<tool>`` entries for
    the tools handed in -- becomes the complete set of things the agent can do.
    ``--disallowedTools`` names the built-ins explicitly as well, because
    relying on an allowlist's implicit complement is a bad way to be confident
    about a permission model.
    """
    from chia.models.claude import ClaudeCodeLLM

    disallowed = ",".join(
        ["Bash", "Read", "Write", "Edit", "MultiEdit", "NotebookEdit",
         "Glob", "Grep", "WebFetch", "WebSearch", "Task", "Agent"]
    )
    extra = list(kwargs.pop("extra_cli_args", None) or [])
    extra += ["--disallowedTools", disallowed]
    return ClaudeCodeLLM(
        model=model,
        system_message=system_message,
        timeout_seconds=timeout_seconds,
        dangerously_skip_permissions=False,
        extra_cli_args=extra,
        **kwargs,
    )


#: The kernel forge was the first caller and named the helper; the co-design
#: architect in ``accelsim_agent`` needs the identical sandbox, so the general
#: name is the real one and this stays for the drivers that already import it.
sandboxed_kernel_llm = sandboxed_llm


def _mnk_tool_description(m: int, n: int, k: int, sm: int,
                          hw_note: str = "") -> str:
    """Tool description for rectangular mode.

    ``submit_kernel``'s docstring is written for the square problem and names
    ``int N``, an ``N x N`` output and an ``N=512`` worked example.  Handing
    that text to an agent working a 4096x4096x64 problem would misdescribe the
    signature it must write, so rectangular mode overrides the description
    rather than editing the docstring the square path still depends on.
    """
    mtiles, ntiles = (m + 127) // 128, (n + 127) // 128
    return f"""Compile, correctness-check and trace a candidate SGEMM kernel.

Provide the complete contents of one .cu file. It must define exactly:

    void launch(const float *A, const float *B, float *C, int M, int N, int K)

which enqueues your kernel(s) on the default stream. A is {m}x{k}, B is
{k}x{n} and C is {m}x{n}, all row-major, already allocated on the device; C is
zeroed. Do not allocate, copy, free, or define main() -- the harness owns all
of that, and source that tries will be rejected before it reaches the compiler.

Your kernel must compute the full C = A x B. It is checked element-by-element
against the reference within a relative tolerance (reassociating the sum is
fine; skipping terms is not), and its floating-point operation count is
compared against the reference's. A kernel that produces the right numbers
with substantially fewer arithmetic operations is rejected, not rewarded.

Args:
    label: short identifier for this candidate, e.g. "tiled_128x128".
    cuda_source: the full text of the .cu file.

Returns a dict with `accepted` plus, when something failed, the compiler
diagnostic or the numerical error that caused it. When it succeeded you get:

- static: `global_loads`, `shared_loads`, `total_warp_insts`
- measured: `cycles`, `power_w`, `edp`, `occupancy`, `cta_per_core`

**`cycles` is the objective. Rank on it and nothing else.** The static counts
are diagnostics that explain a cycle count; they are not a proxy for it, and
on this problem they are close to anti-correlated with it.

This shape is the thing to think about before you tile. K is only {k} -- the
reduction is short, so there is little arithmetic to hide memory behind, and
the kernel is bandwidth- and issue-bound rather than compute-bound. M and N
are large, so the output is big: a 128x128 threadblock tile gives a
{mtiles}x{ntiles} grid = {mtiles * ntiles} threadblocks, which is ample for a
{sm}-SM part. Bigger tiles cut loads but cut block count too; with K this
short, deep double-buffering has little to overlap.

`cta_per_core` and `occupancy` in the return value tell you whether you are
using the machine. Spend your simulations on designs that differ in shape,
not on micro-variants of one you have already measured; `sims_left` says how
many you have.
{hw_note}"""


class KernelForgeTool(ChiaTool):
    """The only surface a kernel-writing agent gets.

    Everything operational -- the Accel-Sim install, the harness, the problem
    size, the output directory, the reference dump -- is bound here at
    construction and never exposed as a model-facing argument. The agent's whole
    vocabulary is "here is some CUDA, tell me if it is correct and what it
    costs".
    """

    def __init__(
        self,
        name: str,
        node: AccelSimNode,
        *,
        harness_cu: str,
        work_dir: str,
        reference_dump: str,
        reference_opcodes: dict[str, int],
        n: int = 256,
        mnk: tuple[int, int, int] | None = None,
        seed: int = 12345,
        arch_sm: str = "sm_80",
        rtol: float = 1e-4,
        atol: float = 1e-5,
        min_work_ratio: float = 0.999,
        max_est_sim_seconds: float = 900.0,
        simulate: bool = False,
        sim_sm: int = 108,
        sim_knobs: dict | None = None,
        sim_trace_knobs: dict | None = None,
        hw_note: str = "",
        max_sims: int = 12,
        accelsim_root: str = "",
        cuda_path: str = "",
        task_options=None,
    ):
        self._node = node
        self._harness_cu = os.path.abspath(harness_cu)
        self._work_dir = os.path.abspath(work_dir)
        self._reference_dump = os.path.abspath(reference_dump)
        self._reference_opcodes = dict(reference_opcodes)
        self._n = n
        # Rectangular mode.  ``n`` stays the square default so every existing
        # caller is byte-for-byte unaffected; when ``mnk`` is given the harness
        # is driven with three dimensions instead of one, which is what lets a
        # written kernel be traced at the *same* problem shape as the CUTLASS
        # tiles and therefore share a grid with them.
        self._mnk = tuple(int(x) for x in mnk) if mnk else None
        self._hargs = (tuple(str(x) for x in self._mnk) if self._mnk
                       else (str(n),))
        self._seed = seed
        self._arch_sm = arch_sm
        self._rtol = rtol
        self._atol = atol
        self._min_work_ratio = min_work_ratio
        self._max_est_sim_seconds = max_est_sim_seconds
        # Simulating every accepted kernel is what turns this tool from a
        # correctness gate into an optimisation loop.  It is off by default
        # because it costs ~200 s per accepted candidate; when it is on, the
        # cycle count -- not the instruction count -- is what comes back.
        self._simulate = bool(simulate)
        self._sim_sm = int(sim_sm)
        # The machine the candidate is graded on.  Empty means stock, which is
        # what the standalone forge uses.  The co-design loop passes the
        # hardware point it just proposed, because a kernel whose whole value
        # is that it keeps a widened queue busy scores no better than one that
        # does not when the queue is not widened -- the effect being searched
        # for is invisible on stock hardware.
        self._sim_knobs = dict(sim_knobs or {})
        self._sim_trace_knobs = dict(sim_trace_knobs or {})
        self._hw_note = hw_note
        self._max_sims = int(max_sims)
        self._n_sims = 0
        self._accelsim_root = accelsim_root or getattr(node, "accelsim_root", "")
        self._cuda_path = cuda_path or getattr(node, "cuda_path", "")
        self._submissions: dict[str, dict] = {}
        os.makedirs(self._work_dir, exist_ok=True)
        super().__init__(name, task_options)
        self.mcp.add_tool(
            self.submit_kernel, name=f"{self.name}_submit_kernel",
            description=(_mnk_tool_description(*self._mnk, self._sim_sm,
                                               self._hw_note)
                         if self._mnk else None))
        self.mcp.add_tool(self.list_submissions, name=f"{self.name}_list_submissions")
        super().__post_init__()

    # -- model-facing ---------------------------------------------------
    async def submit_kernel(self, label: str, cuda_source: str) -> dict:
        """Compile, correctness-check and trace a candidate SGEMM kernel.

        Provide the complete contents of one .cu file. It must define exactly:

            void launch(const float *A, const float *B, float *C, int N)

        which enqueues your kernel(s) on the default stream. A, B and C are
        already-allocated N x N row-major device buffers; C is zeroed. Do not
        allocate, copy, free, or define main() -- the harness owns all of that,
        and source that tries will be rejected before it reaches the compiler.

        Your kernel must compute C = A x B for the full N x N output. It is
        checked element-by-element against the reference within a relative
        tolerance (reassociating the sum is fine; skipping terms is not), and
        its floating-point operation count is compared against the reference's.
        A kernel that produces the right numbers with substantially fewer
        arithmetic operations is rejected, not rewarded. Accepted kernels are
        then simulated, and the cycle count that comes back is the only
        ranking metric that counts.

        Args:
            label: short identifier for this candidate, e.g. "tiled_32".
            cuda_source: the full text of the .cu file.

        Returns a dict with `accepted` plus, when something failed, the compiler
        diagnostic or the numerical error that caused it.

        When it succeeded you get both kinds of number, and they do not agree:

        - static: `global_loads`, `shared_loads`, `total_warp_insts`
        - measured: `cycles`, `power_w`, `edp`, `occupancy`, `cta_per_core`,
          from a full simulation of the part

        **`cycles` is the objective. Rank on it and nothing else.** The static
        counts are diagnostics that tell you *why* a kernel is fast or slow;
        they are not a proxy for speed, and on this problem they are close to
        anti-correlated with it. A previous agent minimised `total_warp_insts`
        and chose a kernel that is 69.8% slower than the one it ranked last.

        The reason is worth internalising before you tile anything. A large
        threadblock tile minimises loads and instructions by giving each
        thread more accumulators -- and it minimises the *number of
        threadblocks*. At N=512 a 256x128 tile is a (4,2) grid: **8
        threadblocks on a 108-SM part, 100 SMs idle**. A 128x128 tile is
        (4,4) = 16, still only 16. `cta_per_core` and `occupancy` in the
        return value are what tell you whether you are using the machine;
        a decomposition that raises the block count (split-K, for instance)
        costs more loads and may still win by a lot.

        Simulation is budgeted. `sims_left` tells you how many accepted
        kernels can still be timed; spend them on designs that differ in
        shape, not on micro-variants of one you have already measured.
        """
        import asyncio
        return await asyncio.to_thread(self._submit_kernel_sync,
                                       label, cuda_source)

    def _submit_kernel_sync(self, label: str, cuda_source: str) -> dict:
        """Body of :meth:`submit_kernel`, run off the event loop.

        Same defect, same fix as ``CoDesignTool.simulate``: FastMCP calls a
        synchronous tool function directly on uvicorn's event loop, so a
        body that blocks wedges the whole MCP server and not merely its own
        call.  This one blocks on four ``chia_remote_blocking`` hops --
        build, verify, trace and, since cycles became the objective, a full
        GPGPU-Sim run -- which together take tens of minutes.  Without the
        hand-off the kernel agent would lose its transport on its first
        submission, exactly as the architect agent did.
        """
        label = re.sub(r"[^A-Za-z0-9_.-]", "_", label)[:64] or "candidate"
        rec: dict = {"label": label, "accepted": False, "stage": "screen"}

        screen = screen_source(cuda_source)
        if not screen.ok:
            rec["error"] = "; ".join(screen.reasons)
            self._record(label, rec)
            return rec

        d = os.path.join(self._work_dir, label)
        os.makedirs(d, exist_ok=True)
        src = os.path.join(d, "candidate.cu")
        with open(src, "w") as fh:
            fh.write(cuda_source)

        rec["stage"] = "build"
        build = AccelSimNode.build_cuda.chia_remote_blocking(
            self._node,
            sources=[self._harness_cu, src],
            binary=os.path.join(d, "harness"),
            arch_sm=self._arch_sm,
        )
        if not build.ok:
            rec["error"] = build.stderr_tail
            self._record(label, rec)
            return rec

        rec["stage"] = "verify"
        ver = AccelSimNode.verify_kernel.chia_remote_blocking(
            self._node,
            binary=build.binary,
            dump=os.path.join(d, "out.bin"),
            args=self._hargs + (str(self._seed),),
            reference_dump=self._reference_dump,
            rtol=self._rtol,
            atol=self._atol,
            cwd=d,
        )
        rec["verify"] = {
            k: v for k, v in asdict(ver).items()
            if k in ("ok", "reason", "n_elems", "n_mismatch", "max_abs_err", "max_rel_err")
        }
        if not ver.ok:
            rec["error"] = f"incorrect output: {ver.reason}"
            self._record(label, rec)
            return rec

        rec["stage"] = "trace"
        tr = AccelSimNode.trace_app.chia_remote_blocking(
            self._node,
            app_cmd=([build.binary] + list(self._hargs)
                     + [str(self._seed), os.path.join(d, "trace_out.bin")]),
            trace_dir=os.path.join(d, "traces"),
            max_est_sim_seconds=self._max_est_sim_seconds,
            cwd=d,
        )
        if not tr.ok:
            rec["error"] = tr.aborted_reason or f"tracing failed: {tr.stderr_tail}"
            self._record(label, rec)
            return rec

        rec["stage"] = "work_check"
        wp = work_preserved(
            tr.opcodes, self._reference_opcodes, min_ratio=self._min_work_ratio
        )
        rec["work_check"] = wp
        if not wp["ok"]:
            rec["error"] = (
                f"work not preserved: {wp['work_ops_candidate']} arithmetic ops vs "
                f"{wp['work_ops_reference']} in the reference (ratio {wp['ratio']:.4f}). "
                "The output matched but the kernel did not do the arithmetic."
            )
            self._record(label, rec)
            return rec

        rec.update(
            accepted=True,
            stage="accepted",
            kernelslist=tr.kernelslist,
            n_kernels=tr.n_kernels,
            est_sim_seconds=round(tr.est_sim_seconds, 1),
            global_loads=sum(v for k, v in tr.opcodes.items() if k.startswith("LDG")),
            shared_loads=sum(v for k, v in tr.opcodes.items() if k.startswith("LDS")),
            total_warp_insts=sum(tr.opcodes.values()),
        )
        if self._simulate:
            rec.update(self._simulate_candidate(label, d, tr.kernelslist))
        self._record(label, rec)
        return rec

    def _simulate_candidate(self, label: str, d: str, kernelslist: str) -> dict:
        """Measure the accepted kernel on the simulated part.

        This exists because the first run of this tool returned only static
        counts, and the agent -- correctly optimising the number it was given
        -- picked a kernel that is **+69.8% slower in cycles** than the one it
        ranked last.  A 256x128 tile at N=512 issues the fewest instructions
        and launches 8 CTAs onto a 108-SM part.  Instructions and occupancy
        pull in opposite directions here and only the simulator arbitrates, so
        the simulator is now inside the loop rather than after it.
        """
        from chia.simulators.accelsim_codesign import (
            ADAPTIVE_OFF, evaluate_point, sm_knobs,
        )
        knobs = {**ADAPTIVE_OFF, **sm_knobs(self._sim_sm), **self._sim_knobs}
        # The simulation cache is keyed on the output path, so two runs of the
        # same kernel that differ only in their knobs must not share one: the
        # second would be handed the first's cycles and nothing would say so.
        tag = (hashlib.sha1(
            repr(sorted({**knobs, **self._sim_trace_knobs}.items())
                 ).encode()).hexdigest()[:8]
            if (self._sim_knobs or self._sim_trace_knobs) else "stock")
        if self._n_sims >= self._max_sims:
            return {"sim": {"ok": False,
                            "note": f"simulation budget spent "
                                    f"({self._max_sims}); the kernel is still "
                                    f"accepted, it just was not timed"}}
        self._n_sims += 1
        v = evaluate_point.chia_remote_blocking(
            self._accelsim_root, self._cuda_path, label,
            f"forge/{label}_sm{self._sim_sm}_{tag}", kernelslist,
            os.path.join(d, f"sim_{tag}"),
            knobs, True, self._sim_trace_knobs or None, False, 10800)
        if not v.get("ok"):
            # A kernel can be correct and still be unsimulatable -- the naive
            # reference deadlocks GPGPU-Sim at sm108.  That is information, not
            # a reason to un-accept it.
            return {"sim": {"ok": False, "note": str(v.get("note", ""))[:160]},
                    "sims_left": self._max_sims - self._n_sims}
        return {"cycles": v.get("cycles"), "power_w": v.get("power_w"),
                "edp": v.get("edp"), "occupancy": v.get("occupancy"),
                "cta_per_core": v.get("cta_per_core"),
                "sim": {"ok": True, "sm": self._sim_sm,
                        "hw": self._hw_note.strip()[:80] or "stock",
                        # A knob the simulator accepted but did not act on
                        # would otherwise be indistinguishable from one that
                        # did nothing, and the kernel would be graded on a
                        # machine that was never built.
                        "unapplied": list(v.get("unapplied") or ()),
                        "cached": bool(v.get("cached"))},
                "sims_left": self._max_sims - self._n_sims}

    def _record(self, label: str, rec: dict) -> dict:
        """Keep the record in-process *and* on disk.

        ``submit_kernel`` runs inside the tool's Ray actor, so ``_submissions``
        only ever exists there; the driver holds a different object and its
        ``list_submissions()`` comes back empty however many kernels were
        graded. A run that produced five accepted kernels reported zero. The
        per-candidate JSON below is the durable copy -- it sits next to the
        source and the trace it describes, so a run can be reconstructed from
        the work dir alone.
        """
        self._submissions[label] = rec
        d = os.path.join(self._work_dir, label)
        try:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "submission.json"), "w") as fh:
                json.dump(rec, fh, indent=1, default=str)
        except OSError:
            pass
        return rec

    def list_submissions(self) -> dict:
        """Return every kernel submitted so far and how each one fared.

        Use this before submitting to avoid re-proposing something already
        tried; a rejected candidate is rejected for the same reason twice.
        """
        return {
            k: {kk: vv for kk, vv in v.items() if kk != "work_check"}
            for k, v in self._submissions.items()
        }

    # -- driver-facing (never exposed to the model) ----------------------
    def _from_disk(self) -> dict[str, dict]:
        """Rebuild the submission table from the durable per-candidate JSON.

        ``_record`` already writes it; nothing read it back until 2026-09-23,
        which is why a run that graded and accepted eight kernels wrote
        ``"submissions": {}, "accepted": []`` to its own result file. The
        evidence was never lost -- it was on disk the whole time, one
        directory per candidate -- but the summary that a reader would quote
        said the run produced nothing.
        """
        out: dict[str, dict] = {}
        try:
            labels = sorted(os.listdir(self._work_dir))
        except OSError:
            return out
        for label in labels:
            f = os.path.join(self._work_dir, label, "submission.json")
            if not os.path.isfile(f):
                continue
            try:
                with open(f) as fh:
                    rec = json.load(fh)
            except (OSError, ValueError):
                continue
            if isinstance(rec, dict):
                out[rec.get("label", label)] = rec
        return out

    def all_submissions(self) -> dict[str, dict]:
        """Every submission, from the actor if reachable and from disk always.

        The driver holds a different object than the one ``submit_kernel``
        mutates, so ``self._submissions`` is empty here in the normal Ray
        case. Disk is therefore the source of truth and the in-process copy
        is the overlay, not the other way round.
        """
        recs = self._from_disk()
        recs.update(self._submissions)
        return recs

    def accepted(self) -> dict[str, dict]:
        return {k: v for k, v in self.all_submissions().items()
                if v.get("accepted")}


# ---------------------------------------------------------------------------
# Library entry point.  The same three things -- build a reference on real
# hardware, run the agent against it, publish what survives -- were written
# out three times (``forge_mnk.py``, ``forge_reference.py``, stage 0b of
# ``run_dse.py``) before the co-design loop needed them a fourth time.  They
# live here now so that a kernel entering the loop's grid is produced by
# exactly the code path a standalone forge run uses, and a fix to the gate
# order cannot apply to one caller and not the others.
# ---------------------------------------------------------------------------

#: The agent's standing brief.  It is deliberately about *method* rather than
#: about this problem: the problem is supplied per-call, and a system message
#: that also described the problem would have to be kept in step with it.
FORGE_SYSTEM = """You are a CUDA kernel engineer working through a single \
submission tool. You cannot read or write files, run commands, or search the \
web; the only thing you can do is submit complete .cu source text and read \
back what the build, the correctness check, the hardware trace and a \
cycle-accurate simulation say about it.

Work empirically. Submit, read the returned counters, and let them decide what \
to change next. Do not submit a batch of variants you have not reasoned about, \
and do not stop after the first kernel that is accepted -- an accepted kernel \
is a correct kernel, not a fast one.

Rank on the measured cycle count and nothing else. Instruction counts, load \
counts and occupancy are diagnostics that explain a cycle count; they are not \
substitutes for it, and on this problem they disagree with it."""


def prepare_reference(node, forge_dir: str, out_dir: str, m: int, n: int,
                      k: int, seed: int, log=print) -> dict | None:
    """Build the harness and capture the output reference on a real GPU.

    Returns ``{"dump", "opcodes", "n_elems"}`` or ``None`` if the harness will
    not build or the reference will not run -- both dull failures, and both
    worth finding out about *before* an agent has been briefed, because a
    briefed agent that cannot be graded is a wasted agent.

    The work gate's reference histogram is analytic rather than traced.  The
    arithmetic a GEMM requires is fixed by its shape: one FFMA per (m, n, k)
    triple, 32 threads to a warp-instruction.  Tracing a naive reference at
    4096x4096x64 to rediscover ``M*N*K/32`` would cost ~100M warp-instructions
    of trace for a number the shape already states exactly.  The *output*
    reference is still really executed on a real GPU; correctness is measured,
    never asserted.
    """
    d = os.path.join(out_dir, "reference")
    os.makedirs(d, exist_ok=True)
    build = AccelSimNode.build_cuda.chia_remote_blocking(
        node, sources=[os.path.join(forge_dir, "harness_mnk.cu"),
                       os.path.join(forge_dir, "reference_mnk.cu")],
        binary=os.path.join(d, "harness"), arch_sm="sm_80")
    if not build.ok:
        log(f"  reference build FAILED\n{build.stderr_tail[:400]}")
        return None
    ver = AccelSimNode.verify_kernel.chia_remote_blocking(
        node, binary=build.binary, dump=os.path.join(d, "reference.bin"),
        args=(str(m), str(n), str(k), str(seed)), reference_dump=None, cwd=d)
    if not ver.ok or ver.n_elems != m * n:
        log(f"  reference dump FAILED: {ver.reason}")
        return None
    return {"dump": os.path.join(d, "reference.bin"),
            "opcodes": {"FFMA": m * n * k // 32},
            "n_elems": ver.n_elems}


def publish_tile(label: str, kernelslist: str, trace_root: str,
                 m: int, n: int, k: int, log=print) -> str | None:
    """Link an accepted candidate's trace in under a name the loop can ask for.

    Additive by construction: a directory that already exists is reported and
    left alone, never replaced, because the loop may already have simulated
    what is in it.  The shape is written alongside, so a directory can state
    what problem it holds and ``trace_shape_ok`` can refuse it later rather
    than replaying one problem size against another and producing numbers that
    look fine and mean nothing.
    """
    from chia.simulators.accelsim_codesign import trace_dir_name
    tname = f"forge_{label}"
    dst = os.path.join(trace_root, trace_dir_name(tname, m, n, k))
    if os.path.exists(dst):
        log(f"  publish {tname}: already exists, left untouched")
        return tname
    src_traces = os.path.dirname(os.path.abspath(kernelslist or ""))
    if not os.path.exists(os.path.join(src_traces, "kernelslist.g")):
        log(f"  publish {tname}: no kernelslist at {src_traces}; skipped")
        return None
    os.makedirs(dst, exist_ok=True)
    os.symlink(src_traces, os.path.join(dst, "traces"))
    with open(os.path.join(dst, "shape.json"), "w") as fh:
        json.dump({"m": m, "n": n, "k": k}, fh)
    log(f"  publish {tname} -> {src_traces}")
    return tname


def forge_kernels(node, *, forge_dir: str, out_dir: str, trace_root: str,
                  brief: str, mnk: tuple, seed: int = 12345,
                  budget: int = 8, model: str = "claude-sonnet-4-6",
                  timeout_s: int = 3600, accelsim_root: str = "",
                  cuda_path: str = "", sim_sm: int = 108,
                  sim_knobs: dict | None = None,
                  sim_trace_knobs: dict | None = None,
                  hw_note: str = "", name: str = "forge",
                  log=print) -> dict:
    """Run the sandboxed kernel agent once and publish what it gets past the gate.

    ``sim_knobs`` is the reason this is callable from inside the co-design
    loop rather than only ahead of it.  A kernel whose whole value is that it
    keeps a widened structure busy measures no better than one that does not
    when that structure has not been widened -- so grading candidates on stock
    hardware while searching for kernels that suit a *proposed* machine looks
    for an effect on a part where it cannot appear.  Passing the loop's own
    hardware point through means the agent is answering the question the loop
    is actually asking.

    Returns ``{"tiles", "submissions", "agent", "accepted"}``.  ``tiles`` is
    what the caller feeds back into the grid; it is empty, not an error, when
    the agent produces nothing that passes the gate.
    """
    m, n, k = mnk
    os.makedirs(out_dir, exist_ok=True)
    ref = prepare_reference(node, forge_dir, out_dir, m, n, k, seed, log=log)
    if ref is None:
        return {"tiles": [], "submissions": {}, "agent": {"ok": False},
                "accepted": 0, "note": "reference unavailable"}
    log(f"  reference ok: {ref['n_elems']:,} elements, "
        f"{ref['opcodes']['FFMA']:,} FFMA warp-insts required")

    tool = KernelForgeTool(
        name, node, harness_cu=os.path.join(forge_dir, "harness_mnk.cu"),
        work_dir=os.path.join(out_dir, "candidates"),
        reference_dump=ref["dump"], reference_opcodes=ref["opcodes"],
        mnk=(m, n, k), seed=seed,
        simulate=True, sim_sm=sim_sm, max_sims=budget,
        max_est_sim_seconds=3600.0,
        sim_knobs=sim_knobs, sim_trace_knobs=sim_trace_knobs,
        hw_note=hw_note,
        accelsim_root=accelsim_root, cuda_path=cuda_path)

    sessions = os.path.join(out_dir, "claude_sessions")
    os.makedirs(sessions, exist_ok=True)
    llm = sandboxed_llm([tool], model=model, system_message=FORGE_SYSTEM,
                        timeout_seconds=timeout_s, projects_cwd=sessions,
                        log_dir=sessions)
    log(f"  tool up at {tool.hostname}:{tool.port}; briefing {model} "
        f"(budget {budget} simulations)")
    r = llm.prompt(brief, tools=[tool])
    log(f"  agent finished: success={r.success} rc={r.returncode}")

    # Not list_submissions(): submit_kernel runs inside the Ray actor, so this
    # object -- the driver's copy -- never sees a single candidate.
    subs = tool.all_submissions() or {}
    acc = {kk: v for kk, v in subs.items() if v.get("accepted")}
    tiles = []
    for label, rec in sorted(acc.items(),
                             key=lambda kv: kv[1].get("cycles") or 9e18):
        log(f"  {label:<28} {(rec.get('cycles') or 0):>12,.0f} cyc  "
            f"cta/sm={rec.get('cta_per_core', '-')}")
        t = publish_tile(label, rec.get("kernelslist") or "", trace_root,
                         m, n, k, log=log)
        if t:
            tiles.append(t)
    out = {"tiles": tiles, "submissions": subs, "accepted": len(acc),
           "agent": {"success": bool(r.success), "returncode": r.returncode}}
    try:
        with open(os.path.join(out_dir, "forge.json"), "w") as fh:
            json.dump({"shape": [m, n, k], "budget": budget, "model": model,
                       "hw_note": hw_note, "sim_knobs": dict(sim_knobs or {}),
                       **out}, fh, indent=1, default=str)
    except OSError:
        pass
    return out
