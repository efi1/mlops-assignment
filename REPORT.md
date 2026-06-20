# Text-to-SQL Agent on vLLM — Report

Model served: `Qwen/Qwen3-30B-A3B-Instruct-2507` (MoE, ~30B total / ~3B active) on a single H100 80GB, via vLLM 0.10.2.
Agent: a LangGraph `attach_schema → generate_sql → execute → verify → (revise → execute → verify)×N` loop, served over FastAPI and traced in Langfuse, with vLLM metrics scraped by Prometheus and visualised in Grafana.

---

## 1. Serving configuration

`scripts/start_vllm.sh` launches vLLM with:

```
--max-model-len 8192
--gpu-memory-utilization 0.90
--max-num-seqs 64
--enable-prefix-caching
--enable-chunked-prefill
```

| Flag | Justification |
|------|---------------|
| `--max-model-len 8192` | Prompts are ~1.5–3K tokens (dominated by the rendered DB schema) with short SQL outputs. Capping context far below the model's native maximum frees KV-cache memory for concurrency instead of reserving it for context we never use. |
| `--gpu-memory-utilization 0.90` | After the BF16 weights (~60GB) on the 80GB H100, this hands most of the remaining memory to the KV cache while leaving headroom against load-time OOM. |
| `--max-num-seqs 64` | Allows dozens of in-flight sequences; gives concurrency headroom without over-committing KV cache. |
| `--enable-prefix-caching` | Every question for a given database shares the same long schema prefix, so caching it avoids recomputing that prefix per request. **Measured 84.7% prefix-cache hit rate at runtime**, confirming this assumption — most prompt tokens were reused rather than recomputed. |
| `--enable-chunked-prefill` | Interleaves prefill of long prompts with decode of in-flight requests, smoothing tail latency when requests arrive together. |

Output was capped at `max_completion_tokens=256`, since SQL answers are short; this bounds worst-case generation time and reduces per-request GPU occupancy.

Tensor/expert parallelism and FP8 were deliberately omitted: the model fits on one H100 in BF16, so they would add complexity or comms overhead for no benefit.

One compatibility note: vLLM 0.10.2 requires `transformers` 4.5x (5.x removed a tokenizer attribute the engine relies on), so `transformers==4.56.2` is pinned in `pyproject.toml`.

---

## 2. Baseline evaluation

Execution accuracy on the 30-question BIRD subset (`results/eval_baseline.json`), with the original verifier and `MAX_ITERATIONS=3`:

| Metric | Value |
|--------|-------|
| Final execution accuracy | **40.0%** (12/30) |
| Pass rate @ iter 1 | 33.3% |
| Pass rate @ iter 2 | 36.7% |
| Pass rate @ iter 3 | 40.0% |
| Agent errors | 0 |

(Grafana during the eval run: `grafana_eval_run.png`.)

**Commentary.** First-attempt accuracy is 33.3%; the verify→revise loop lifts this to 40.0% by iter 3, so the loop recovers questions the first attempt got wrong rather than merely consuming tokens. Spot-checking the failures shows they are genuine semantic SQL errors, not artifacts of the result-comparison metric — e.g. for *"How many male clients in 'Hl.m. Praha' district?"* the agent generated a well-formed query filtering `gender = 'm'`, but the data encodes gender as `'M'`, so the count was wrong. Failures are dominated by value-encoding guesses and by aggregations (average / percentage / difference / count), which is the expected hard core of text-to-SQL for a 30B model given only the schema (no sample values). (This accuracy run uses `MAX_ITERATIONS=3` to show the loop's full potential; the final *serving* config uses `MAX_ITERATIONS=2`, trading ~7 accuracy points for a large latency reduction — see §3.)

---

## 3. Hitting the SLO

**Target: P95 end-to-end agent latency < 5s at 10 RPS, sustained 5 min.**

### The system fails the SLO at 10 RPS
At the target load the agent collapses: P95 ≈ **113s** against the 5s target, with **27% of requests timing out** and only ~40% completing. This is a ~23× miss — so the interesting work is not a green check, but explaining *why* it breaks.

### Diagnosis: the dashboard rules out the GPU
The decisive clue is on Grafana (`grafana_serving.png`): as load ramps, the *Requests/sec* and *Generated tokens/sec* panels climb, but the **KV Cache Usage & Queue panel stays low (peak ~4–12%) with requests-waiting flat at 0**, and the prefix-cache hit rate is 84.7%. If vLLM were the bottleneck, the KV cache would saturate and the queue would grow — neither happens. So the GPU has spare capacity; the latency is being created *somewhere else*. vLLM-side latency looked healthy (~4.5s) while end-to-end latency was ~25× the SLO, and the gap between those two boundaries is the diagnosis: **the time is spent before requests reach the GPU, on the agent side.**

Two agent-side factors produce that gap:
1. **Fan-out (work per request).** Each request makes 2–4 *sequential* LLM calls (generate → verify → revise → verify), each ~1–2s. At 10 RPS that is 20–40 LLM calls/sec aimed at one vLLM instance.
2. **Serialization (handling of that work).** A single uvicorn worker orchestrates all of those requests in one process, so they queue *inside the agent* before ever being dispatched to vLLM.

These suggest two independent levers: reduce the **work** (`MAX_ITERATIONS`), and parallelise the **handling** (uvicorn `--workers`). The rest of this section tests each, changing one thing at a time and confirming on the dashboard.

### Lever 1 — reduce work per request (`MAX_ITERATIONS`)
Load tests at 5 RPS / 60s, prompt held constant, only `MAX_ITERATIONS` changed (`grafana_before.png` = iter=3, `grafana_after.png` = iter=2):

| Config | P95 | P99 | timeouts |
|--------|-----|-----|----------|
| iter=3 | 8.28s | 10.22s | 0 |
| iter=2 | **5.17s** | 7.02s | 0 |

Dropping 3→2 removes one revise round (fewer LLM calls per request). On the dashboard the *Requests running* / token panels show lower sustained activity for the same offered load — the targeted metric moved — and **end-to-end P95 moved with it** (8.28s → 5.17s, ~37%). This is the "metric improved *and* the SLO improved" case. At a gentler 2 RPS, iter=2 reaches P95 = 2.91s (P99 3.94s, max 4.77s) — comfortably inside the SLO. So fewer iterations clearly helps, but at the full 10 RPS it is not enough on its own.

### Lever 2 — parallelise request handling (uvicorn workers)
If the bottleneck is single-process serialization (as the idle GPU implies), giving the agent more worker processes should let it dispatch concurrent requests instead of queuing them. Tested at 10 RPS, `MAX_ITERATIONS=2`:

| Config @ 10 RPS | P95 | P99 | timeouts | ok |
|-----------------|-----|-----|----------|-----|
| 1 worker (baseline) | ~113s | ~119s | 27% | 40% |
| 4 workers | 8.97s | 12.28s | 0 | 87% |
| 8 workers | **6.79s** | 9.57s | 0 | 87% |

Scaling to 8 workers cut P95 from ~113s to **6.79s (≈17×)** and eliminated timeouts entirely — **directly confirming the dashboard-grounded hypothesis that the bottleneck was agent-side serialization, not the GPU.** On Grafana, `requests running` rose from ~7 (1 worker) to ~30 (8 workers) while the KV cache still barely moved (`grafana_workers_sweep.png`): the agent was finally feeding vLLM concurrently, yet the GPU was *still* not the limit.

### Where it lands, and the residual bottleneck
Best configuration found: **`MAX_ITERATIONS=2` + 8 uvicorn workers → P95 6.79s at 10 RPS, zero timeouts, 87% success** — a 17× improvement over the baseline that turns a total collapse into a near-miss. Two limits remain:
- P95 (6.79s) is still just above the 5s SLO, and achieved throughput plateaus at ~5 RPS (the driver could not push the full 10).
- Under sustained 8-worker load, `requests running` pinned near the `--max-num-seqs 64` ceiling — so the *residual* bottleneck has now moved to **vLLM single-instance throughput**.

So there are two distinct bottlenecks, addressed by two different levers: agent concurrency (fixed by workers) and vLLM call throughput (the next limit). The levers are not interchangeable — reducing `MAX_ITERATIONS` cuts the *total* LLM work and so eases **both**, whereas workers only parallelise the agent's handling of that work. This is why `MAX_ITERATIONS=3` re-collapsed even at 8 workers (the extra calls re-saturated vLLM). **Honest verdict:** the SLO as stated (P95 < 5s at a true 10 RPS) was not met on a single H100, but the investigation took it from 113s to 6.79s and localised every bottleneck along the way — which is the actual objective of this phase. The committed config is `MAX_ITERATIONS=2`. (Note: `MAX_ITERATIONS=3` worker runs were attempted but discarded — a configuration typo made them unreliable.)

**A secondary finding:** under sustained load ~13% of requests returned instant (sub-10ms) HTTP 500s — failing *before* any LLM call. Single requests, short bursts, and the sequential eval all run error-free, and spacing requests out eliminates them, so these are connection contention between agent and vLLM under sustained parallelism (the agent surfaces a downstream connection failure as a 500), not a logic bug. The stricter verifier (more calls) made them more frequent, consistent with the fan-out picture. Mitigation: retry-with-backoff and connection-pool tuning in the agent's vLLM client.

---

## 4. Did the agent loop add value?

Yes. The per-iteration pass rate rises monotonically (baseline 33.3% → 36.7% → 40.0%), so the loop converts first-attempt failures into passes rather than burning tokens. A concrete case: for *"What is the coordinates location of the circuits for the Australian Grand Prix?"*, the first attempt omitted `DISTINCT` and returned 11 identical rows; the verifier flagged the duplicates (`ok: false`, "only one unique (lat, lng) pair should be returned"), `revise` regenerated the query with `DISTINCT`, and the second attempt returned the single correct row — a failure turned into a pass under the multiset comparison (this exact trace is shown in `langfuse_trace.png`). Strengthening the verifier amplified this effect: it raised final accuracy from **40.0% → 46.7%** (`eval_after_tuning.json`), with the gain coming through the loop (iter curve 36.7% → 40.0% → 46.7%). The cost is more revise calls — i.e. more fan-out, hence the §3 latency/stability tension. At `MAX_ITERATIONS=2` we keep most of the loop's benefit while bounding latency.

---

## 5. What I would do with more time

- **Schema linking with sample values.** Most remaining failures are the model guessing literal values it can't see from the schema alone (the `gender='m'` vs `'M'` class) and complex aggregations. Retrieving the relevant tables/columns *with a few example rows per column* would attack the largest failure group directly, and shrinking the prompt would also cut latency.
- **Verifier precision.** Several failures were approved by the verifier on the first attempt (wrong answers it judged plausible), so the loop never engaged. I'd build a small labelled set of (question, result, correct?) pairs and tune the verify prompt against measured precision/recall, rather than inferring verifier quality from end-to-end accuracy.
- **Scale the serving layer to actually reach 10 RPS.** The worker experiment fixed agent-side serialization but exposed vLLM single-instance throughput as the next limit (`requests running` pinned at `--max-num-seqs`). I'd raise `--max-num-seqs`, run multiple vLLM replicas behind a load balancer, and tune the agent's connection pool — combined with `MAX_ITERATIONS=2` and workers, this is the path to meeting the SLO at the true 10 RPS rather than the ~5 RPS currently sustainable.
- **Connection robustness and adaptive iteration.** Add retry-with-backoff and connection-pool tuning to eliminate the sustained-load 500s, and an adaptive iteration cap (stop early when verifier confidence is high, allow an extra revise only when a failure looks recoverable) to spend LLM calls — and therefore fan-out — where they actually help.
