# Accel-Sim in CHIA — GPU hardware/software co-design

A GPU simulation backend for CHIA, and a co-design loop that explores hardware
knobs and GEMM tiles together.

---

## Requirements

| | |
|---|---|
| Accel-Sim 2.0 / GPGPU-Sim 4.2.0 | built with AccelWattch power |
| McPAT | on `$PATH`, or set `MCPAT_BIN` |
| CUDA 12.8 | plus one NVIDIA GPU, for tracing |
| CUTLASS | headers |
| CPU cores | one per concurrent simulation; 100 slots used by default |

## Install

The backend is an addition to CHIA, not a replacement for it: it imports
`chia.base.ChiaFunction`, `chia.base.tools.ChiaTool` and `chia.models.claude`.
Copy the modules into a CHIA checkout and point `CHIA_ROOT` at that checkout's
root — the directory containing the `chia` package.

```bash
CHIA=/path/to/chia            # the checkout root: holds chia/base, chia/models
cp chia/simulators/*.py chia/simulators/*.json  $CHIA/chia/simulators/
export CHIA_ROOT=$CHIA        # every driver does sys.path.insert(0, CHIA_ROOT)
```

## Setup

```bash
export ACCELSIM_ROOT=/path/to/accel-sim-framework
export CUDA_PATH=/usr/local/cuda-12.8
export CUTLASS_ROOT=/path/to/cutlass
export MCPAT_BIN=/path/to/mcpat
export CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT=2400000      # milliseconds
```

All other paths derive from each script's location. Optional overrides:
`DSE_ROOT`, `DSE_TRACES`, `DSE_OUT`, `DSE_SLOTS`, `PYTHON`.

## Run the whole exploration

One script does every stage and prints the numbers at each one.

```bash
# the built-in CUTLASS tile
python3 testkernel/dse/run_dse.py

# your own kernel, gated first and then explored
python3 testkernel/dse/run_dse.py mykernel.cu

# a different reduction depth
python3 testkernel/dse/run_dse.py mykernel.cu --k 256

# re-print a finished run's tables without measuring anything again
python3 testkernel/dse/run_dse.py --report testkernel/dse/out_dse_baseline_k64
```

Allow 6–8 hours on 100 slots for a cold run. Results are cached by config
fingerprint, so re-runs are cheaper.

Stages, in the order they print:

| | |
|---|---|
| 0 | capture the baseline trace |
| 0b | gate the supplied kernel: compile, correctness, work, trace, simulate |
| 1 | baseline cycles, power, energy, occupancy |
| 2 | probe each resource at its limit and rank the headroom |
| 3 | attribute stalls to a limiter |
| 4 | read the raw counters |
| 4c | write kernels from this run's bottleneck report (`DSE_FORGE=1`) |
| 5 | propose hardware points and tiles |
| 6 | build and trace the proposed tiles |
| 7 | the full grid: every tile against every hardware point |
| 7b | price every hardware point and drop anything over the area budget |
| 8 | check whether two wins still win together |
| 9 | re-measure across SM 104–112 |
| 10 | area ledger |

Switches:

| Variable | Effect |
|---|---|
| `DSE_AGENT=0` | run with no LLM in the loop |
| `DSE_AGENT_BUDGET=N` | cap simulations (0 = unlimited) |
| `DSE_FORGE_TILES=a,b` | add already-published tiles; `-` adds none |
| `DSE_FORGE=1` | write new kernels inside stage 4c |
| `DSE_FORGE_BUDGET=N` | simulations the kernel agent may spend; default 8 |
| `DSE_FORGE_GRADE=stock` | time candidates on a stock A100 instead of the proposed point |
| `DSE_SLOTS=N` | concurrent simulations |
| `DSE_AREA_BUDGET=mm2` | net area a design may add; default 0 (iso-area) |

## Testing a new kernel

Write one function. Nothing else.

```c
void launch(const float *A, const float *B, float *C, int M, int N, int K);
```

A is M x K, B is K x N, C is M x N, all row-major float32. The harness owns
allocation, initialisation and checking.

```bash
python3 testkernel/dse/run_dse.py mykernel.cu
```

The kernel must compile against the harness, match the reference to 1e-4
relative / 1e-5 absolute, and issue the arithmetic the shape requires. If it
fails, the run stops and names the stage it failed at. If it passes, it is
published as a tile and carried through every later stage.

To gate a kernel without exploring afterwards:

```bash
FORGE_KERNEL=mykernel.cu FORGE_LABEL=mykernel \
python3 testkernel/dse/forge_mnk.py submit
```

Set `FORGE_M`, `FORGE_N`, `FORGE_K` for another shape, and pass the same K to
`run_dse.py`. Traces are tagged with their shape; one shape is not replayed
against another.

Run kernel source you did not write inside a container. `nvcc` executes
arbitrary code at compile time, and the source screen does not stop that.

## Output

Written to `DSE_OUT` (default `testkernel/dse/out_dse_<kernel>_k<K>/`):

| file | contents |
|---|---|
| `datapoints.csv` | every grid point: cycles, watts, energy, EDP, CTA/SM |
| `datapoints.json` | the same, plus the selected design |
| `journal.json` | every step, every measurement, every decision rule |

Plot the grid — hardware points across, tiles up, the selected point and the
baseline ringed.  One figure per metric: colour is EDP, cycles or energy; the
number in each cell is the change against the baseline:

```bash
python3 testkernel/dse/plot_grid.py <run_dir>            # all three
python3 testkernel/dse/plot_grid.py <run_dir> perf       # one only
# writes grid_edp.png, grid_perf.png, grid_energy.png
```

## Running stages separately

```bash
python3 testkernel/dse/forge_selftest.py      # gate check; expect all_as_expected=True
python3 testkernel/dse/chia_dse.py trace 64   # baseline trace only
python3 testkernel/dse/chia_dse.py smoke      # dispatch check
python3 testkernel/dse/chia_dse.py run 64     # the loop, without the reporter
python3 testkernel/dse/probes_both.py         # knob sweep
python3 testkernel/dse/counter_agent.py       # counter study
FORGE_BUDGET=6 python3 testkernel/dse/forge_mnk.py   # write kernels
```

## Layout

```
chia/simulators/        the backend (library; takes every path as an argument)
testkernel/forge/       kernel harness + reference
testkernel/dse/         drivers, and the tile template in src/
testkernel/dse/exploratory/   development scripts; not portable, not required
```

`chia/simulators/` is the backend. Reproducing the numbers needs `testkernel/`
as well.

## Troubleshooting

Two files in your CHIA checkout may need editing. Neither is shipped here.

| Symptom | File | Fix |
|---|---|---|
| `ImportError: cannot import name 'AccelSimNode' from 'chia.simulators'` | `chia/simulators/__init__.py` | Import from the submodule instead — `from chia.simulators.accelsim import AccelSimNode` — or re-export the new names from `__init__.py`. Nothing shipped here needs the re-export. |
| An agent run dies with `AuthenticationError` while your credentials are fine | `chia/models/claude.py` | The CLI prints `API Error: 502 ...` to stdout, but `ClaudeCodeLLM` classifies errors by keyword and reads only stderr, so the word `login` in the banner matches first. In its error handling, read an HTTP status from both `cli.stderr` and `cli.result` and branch on the status code before any keyword match. |
