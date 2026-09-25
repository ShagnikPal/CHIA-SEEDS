"""Build + trace the CUTLASS GEMM at a SECOND shape, to scope the findings.

Everything measured so far is one shape: 4096x4096x64 TF32.  That is a *thin
K* GEMM -- only 4 K-iterations at TB_K=16 -- so each output element is written
after very little accumulation and the kernel is unusually write-heavy.  The
whole miss-queue story is a write-path story (100% of L2 reservation failures
are MISS_QUEUE_FULL), which makes "is this just a thin-K artifact?" the
obvious and so far unanswered objection.

K=256 is the control: same M, same N, so the same 1024 CTAs and the same
output traffic, but 4x the accumulation per output element and therefore ~4x
less write pressure per unit of work.  If the occ2+mq16 sign flip (a stable
+4.14% on stock software, a stable -3.69% on tuned software) survives at K=256
it is a property of the co-design, not of the shape.  If it vanishes, the
finding is real but scoped to thin-K GEMMs -- which is worth saying plainly,
since thin-K is exactly the shape that shows up in attention projections.

Usage:  shape2.py [K]            (default 256)
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")
import ray

ROOT = "/raid/sp56429/chia_hackathon/testkernel/dse"
ACCELSIM = "/raid/sp56429/chia_hackathon/accel-sim-framework"
CUDA = "/usr/local/cuda-12.8"
CUTLASS = "/raid/sp56429/Accel-sim/sp_accelsim/final_workloads/cutlass_gemms/cutlass"
M = N = 4096


TILES = {
    "base": ["-DTB_M=128", "-DTB_N=256", "-DTB_K=32",
             "-DWP_M=64", "-DWP_N=64", "-DWP_K=32", "-DSTAGES=3"],
    "t128x128x16_s4": ["-DTB_M=128", "-DTB_N=128", "-DTB_K=16",
                       "-DWP_M=64", "-DWP_N=64", "-DWP_K=16", "-DSTAGES=4"],
}


@ray.remote(num_cpus=8, num_gpus=1)
def build_and_trace(name: str, defines: list, k: int, tdir: str) -> dict:
    sys.path.insert(0, "/raid/sp56429/chia_hackathon/chia")
    from chia.simulators.accelsim import AccelSimNode
    t0 = time.time()
    out = f"{ROOT}/build_k{k}/{name}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = ["/usr/local/cuda-12.8/bin/nvcc", "-arch=sm_80",
           f"-I{CUTLASS}/include", f"-I{CUTLASS}/tools/util/include",
           "-O3", "--expt-relaxed-constexpr", "--use_fast_math",
           *defines, "-DSPLIT_K=1",
           f"-DGEMM_M={M}", f"-DGEMM_N={N}", f"-DGEMM_K={k}",
           f"{ROOT}/src/gemm_tpl.cu", "-o", out]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        return {"tile": name, "ok": False, "stage": "nvcc",
                "note": p.stderr[-500:]}


    r = subprocess.run([out], capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or not r.stdout.startswith("OK "):
        return {"tile": name, "ok": False, "stage": "correctness",
                "note": (r.stdout + r.stderr)[:300]}
    node = AccelSimNode(ACCELSIM, CUDA)
    tr = AccelSimNode.trace_app(node, app_cmd=[out],
                                trace_dir=f"{tdir}/{name}", cwd=ROOT)
    return {"tile": name, "ok": bool(tr.ok), "stage": "trace",
            "kernelslist": tr.kernelslist, "ref": r.stdout.strip(),
            "wall_s": time.time() - t0}


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    tdir = f"{ROOT}/traces_k{k}"
    os.makedirs(tdir, exist_ok=True)
    ray.init(ignore_reinit_error=True)
    print(f"[shape2] building + tracing {M}x{N}x{k} for {list(TILES)}")
    futs = [build_and_trace.remote(n, d, k, tdir) for n, d in TILES.items()]
    res = ray.get(futs)
    for r in res:
        print(f"  {r['tile']:<20} ok={r['ok']} stage={r['stage']} "
              f"{r.get('wall_s', 0):.0f}s  {r.get('ref', '')[:60]}")
        if not r["ok"]:
            print(f"     note: {r.get('note', '')[:400]}")
    json.dump(res, open(f"{tdir}/build.json", "w"), indent=1)
    print(f"[shape2] traces in {tdir}")
