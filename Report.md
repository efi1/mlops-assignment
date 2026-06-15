# Text-to-SQL Agent on vLLM — Report

> Model served: `Qwen/Qwen3-30B-A3B-Instruct-2507` on a single H100 (80GB), via vLLM.
> Agent: LangGraph `generate → execute → verify → (revise → execute)×N` loop, traced in Langfuse.

---

## 1. vLLM configuration rationale

Serving command (`start_vllm.sh`):

```
--max-model-len 8192
--gpu-memory-utilization 0.90
--max-num-seqs 64
--enable-prefix-caching
--enable-chunked-prefill
```

| Flag | Why |
|------|-----|
| `--max-model-len 8192` | Prompts are 1.5–3K tokens (mostly the rendered DB schema) with short SQL outputs. Capping context well below the model's native maximum frees KV-cache memory for concurrency instead of reserving it for context we never use. |
| `--gpu-memory-utilization 0.90` | After ~60GB of BF16 weights on the 80GB H100, this hands most of the remaining ~20GB to the KV cache. Dropped to 0.85 only if load-time OOM occurred (see §3). |
| `--max-num-seqs 64` | At 10 RPS with ~2–3s/request, in-flight concurrency stays in the low dozens; 64 gives headroom without over-committing KV cache. Primary tuning knob in §3. |
| `--enable-prefix-caching` | Every question for a given database shares the same long schema prefix. Prefix caching reuses that prefix's KV cache across requests instead of recomputing it — the highest-leverage flag for this workload. |
| `--enable-chunked-prefill` | Interleaves prefill of long prompts with decode of in-flight requests, smoothing tail latency when requests arrive together. |

Deliberately omitted: tensor parallelism (model fits on one H100; TP would add comms overhead for no gain), expert parallelism (single GPU), and FP8 (BF16 fits in 80GB; not needed).

<!-- TODO after H100: note the exact vLLM version used, and confirm metric names matched the dashboard queries -->

---

## 2. Evaluation results

Execution accuracy on the 30-question BIRD subset. Both agent SQL and gold SQL are executed against the same SQLite DB; result sets compared via the provided multiset canonicalization (sorted rows, stringified cells, NULL→"").

Baseline (`results/eval_baseline.json`):

| Metric | Value |
|--------|-------|
| Final execution accuracy | `<TODO %>` |
| Pass rate @ iter 1 | `<TODO %>` |
| Pass rate @ iter 2 | `<TODO %>` |
| Pass rate @ iter 3 | `<TODO %>` |
| Agent errors | `<TODO>` |

**Commentary:** `<TODO — 1–2 sentences: what was the first-attempt accuracy, how much did the loop add by iter 2/3, and what does that say about generation quality vs. the loop's contribution>`

<!-- TODO after H100: paste the summary block from results/eval_baseline.json -->

---

## 3. SLO diagnosis and tuning log

Target: **P95 end-to-end agent latency < 5s at 10 RPS.**

Load test: `load_test/driver.py --rps 10 --duration 300`.

### Baseline performance vs. SLO
- **P95 latency:** `<TODO>` s vs. 5s target → `<met / not met>`
- **Throughput sustained:** `<TODO>` RPS vs. 10 RPS target
- **Supporting metrics at baseline:** KV-cache peak `<TODO>`, requests-waiting peak `<TODO>`

### Iteration log

**Iteration 1**
- **Saw:** `<TODO — e.g. KV cache saturated at 0.98, queue climbing>`
- **Hypothesized:** `<TODO — e.g. max-num-seqs too high, causing cache thrash>`
- **Changed:** `<TODO — e.g. max-num-seqs 64 → 32>`
- **Result:** `<TODO — P95 moved from X to Y>`

**Iteration 2**
- **Saw / Hypothesized / Changed / Result:** `<TODO>`

<!-- Keep at least one real diagnosis cycle even if iter 0 already met the SLO.
     Diagnosis quality is graded above hitting the number. -->

### Final numbers (post-tuning)
- **Final P95 latency:** `<TODO>` s → `<met / not met>`
- **Final throughput:** `<TODO>` RPS
- **Final config change(s) from baseline:** `<TODO — list the flags that differ from §1>`

### Quality regression check
Re-ran the eval after tuning (`results/eval_after_tuning.json`) to confirm latency changes did not degrade correctness:

| | Baseline | After tuning |
|--|----------|--------------|
| Final accuracy | `<TODO>` | `<TODO>` |

<!-- TODO after H100: confirm accuracy held; if it dropped, explain the tradeoff -->

---

## 4. Did the verify→revise loop add value?

Yes — the per-iteration pass rate rises from iter 1 to iter 2 (`<TODO baseline numbers>`), so the loop recovers questions the first attempt got wrong rather than just burning tokens.

A concrete example observed during development: for *"What is the coordinates location of the circuits for the Australian Grand Prix?"*, the first attempt omitted `DISTINCT` and returned 11 identical coordinate rows. The verifier flagged the duplicates (`ok: false`, issue noting repeated rows), `revise` regenerated the query with `DISTINCT`, and the second attempt returned the single correct row — turning a failure into a pass. Under the provided multiset comparison, the un-deduplicated first attempt would have failed the eval, so the loop directly converted a wrong answer into a right one.

The curve also shows where the loop stops paying off: gains `<plateau / continue>` after iter `<N>`, which argues for `MAX_ITERATIONS = <value>` as the cost/quality sweet spot.

<!-- TODO after H100: replace bracketed numbers with the real per-iteration curve -->

---

## 5. What I would do with more time

- **Verifier precision:** the verifier is lenient by design (to avoid revising correct answers), but this lets some wrong-but-plausible results through. I'd build a small labeled set of (question, result, correct?) pairs and tune the verify prompt against it, measuring verifier precision/recall directly rather than inferring it from end-to-end accuracy.
- **Schema-aware prompting:** the schema dominates the prompt token budget. I'd experiment with retrieving only the tables/columns relevant to each question (schema linking) to cut prompt size, which would lower latency and likely improve generation quality.
- **Adaptive iteration cap:** rather than a fixed `MAX_ITERATIONS`, stop early when the verifier's confidence is high and allow an extra revise when the failure looks recoverable, spending LLM calls where they actually help.

<!-- Keep these specific and tied to what was observed. Avoid generic infra answers. -->
