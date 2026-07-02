# Supporting `qwen35` (hybrid SSM + attention) in Prism

**Status:** analysis / not implemented
**Trigger model:** `model/Qwen3.5-4B-Q4_0.gguf` (Unsloth quant, arch `qwen35`)
**Date:** 2026-07-02

---

## 1. Summary

The `Qwen3.5-4B` GGUF cannot run on Prism's current inference engine. Two
independent issues were found:

1. **Unsupported quantization types** — the file mixes in `Q5_K` and `Q6_K`
   tensors. *This has been fixed*: Prism now dequantizes `Q4_K`, `Q5_K`, `Q6_K`
   (see `src/Prism.Vector.pas`). The model now loads and the server starts.
2. **Unsupported architecture** — `qwen35` is **not** a Llama-family transformer.
   It is a **hybrid Mamba2/SSM + gated-attention** model. Prism's engine
   (`src/Prism.Llama.pas`) only implements the standard Llama block, so the first
   inference request fails with `missing tensor: blk.0.ffn_norm.weight`.

This document scopes what a `qwen35` engine would require. It is a **large**
addition (multi-day to multi-week), and the core risk is matching the exact SSM
math against a reference implementation.

---

## 2. How we got here (symptoms)

| Symptom | Cause | Resolution |
|---|---|---|
| `EIntOverflow` at startup (`HalfToFloatCompute(1024)`) | Unsigned underflow `UInt32(Exp) - 15` with `{$Q+}` on | Fixed: reordered to `(Exp + 127) - 15` |
| `ERROR: TQTensor: unsupported GGML type` on load | Model contains `Q5_K`/`Q6_K` tensors | Fixed: added K-quant dequant (Way A) |
| `missing tensor: blk.0.ffn_norm.weight` on first request | `qwen35` layer layout ≠ Llama | **Open** — requires this engine |

---

## 3. Current Prism capabilities (reusable building blocks)

- RMSNorm, SwiGLU FFN (`ffn_gate`/`ffn_up`/`ffn_down`)
- Quantized `MatVec` incl. `F32, F16, Q4_0, Q4_1, Q8_0` and now `Q4_K, Q5_K, Q6_K`
- Embedding lookup, weight tying, sampling, tokenizer, REST server (OpenAI/Ollama)
- Standard GQA attention with RoPE (NeoX-style) + KV-cache
- Layer streaming (LRU of N layers)

---

## 4. The `qwen35` architecture (from GGUF inspection)

### 4.1 Global hyperparameters (metadata)

```
block_count               = 32
embedding_length          = 2560
feed_forward_length       = 9216
attention.head_count      = 16
attention.head_count_kv   = 4        (GQA)
attention.key_length      = 256
attention.value_length    = 256      (head_dim = 256)
attention.layer_norm_rms_epsilon = 1e-6
rope.freq_base            = 1e7
rope.dimension_count      = 64       (partial RoPE: 64 of 256 head dims)
rope.dimension_sections   = [11,11,10,0]   (M-RoPE / sectioned)
full_attention_interval   = 4        (every 4th layer is full attention)
ssm.conv_kernel           = 4
ssm.state_size            = 128
ssm.group_count           = 16
ssm.time_step_rank        = 32
ssm.inner_size            = 4096
context_length            = 262144
vocab                     = 248320
general.tags              = [unsloth, image-text-to-text]   (multimodal)
```

### 4.2 Two block types

Layers alternate by `full_attention_interval = 4`: 24 SSM blocks and 8 full
attention blocks. Both end in a SwiGLU FFN preceded by `post_attention_norm`.

**SSM / gated-linear-attention block** — 24 layers `[0,1,2,4,5,6, …, 28,29,30]`

| Tensor | Shape (in,out) | Type | Role |
|---|---|---|---|
| `attn_norm.weight` | 2560 | F32 | input RMSNorm |
| `attn_qkv.weight` | 2560 → 8192 | Q4_0 | in-projection (x / B / C / dt inputs) |
| `attn_gate.weight` | 2560 → 4096 | Q4_0 | output gate (SiLU) |
| `ssm_conv1d.weight` | 4 × 8192 | F32 | causal depthwise conv, kernel 4 |
| `ssm_a` | 32 | F32 | SSM `A` (decay) parameter |
| `ssm_alpha.weight` | 2560 → 32 | Q8_0 | projection (dt / selection) |
| `ssm_beta.weight` | 2560 → 32 | Q8_0 | projection (dt / selection) |
| `ssm_dt.bias` | 32 | F32 | `dt` bias |
| `ssm_norm.weight` | 128 | F32 | gated RMSNorm over state dim |
| `ssm_out.weight` | 4096 → 2560 | Q5_K | out-projection |

**Full attention block** — 8 layers `[3,7,11,15,19,23,27,31]`

| Tensor | Shape (in,out) | Type | Role |
|---|---|---|---|
| `attn_norm.weight` | 2560 | F32 | input RMSNorm |
| `attn_q.weight` | 2560 → 8192 | Q4_0 | query projection |
| `attn_k.weight` | 2560 → 1024 | Q4_0 | key (4 KV heads × 256) |
| `attn_v.weight` | 2560 → 1024 | Q4_0 | value |
| `attn_q_norm.weight` | 256 | F32 | **per-head QK RMSNorm** |
| `attn_k_norm.weight` | 256 | F32 | per-head QK RMSNorm |
| `attn_output.weight` | 4096 → 2560 | Q4_0 | output projection |

Top-level: `token_embd.weight` (Q6_K, tied output), `output_norm.weight` (F32).
No separate `output.weight` → **weight tying**.

---

## 5. Gap analysis

| Component | Status |
|---|---|
| K-quant dequant (`Q4_K/Q5_K/Q6_K`) | ✅ done |
| RMSNorm / SwiGLU FFN | ✅ reusable |
| Per-layer type dispatch | ❌ new |
| QK-RMSNorm (per head) | ❌ new |
| M-RoPE (sectioned, partial 64/256) | ❌ new |
| Causal depthwise Conv1D + conv state | ❌ new |
| Selective SSM / Mamba2 scan + recurrent state | ❌ new (core) |
| Stateful decode loop (SSM state cache per session) | ❌ new (invasive) |
| Vision encoder (multimodal) | ⛔ out of scope for text chat (separate mmproj) |

---

## 6. Implementation scope

### A. Config & layer dispatch — *small*
- Parse `qwen35.*` metadata (incl. `ssm.*`, `rope.dimension_sections`,
  `full_attention_interval`, explicit `key_length`/`value_length`).
- Two layer records (SSM vs attention); a layer `L` is full attention when
  `L mod full_attention_interval = full_attention_interval - 1` (i.e. `L mod 4 = 3`).
- Extend the (streaming) layer loader to load either tensor set.

### B. Attention variant (8 layers) — *medium*
- Separate `q/k/v` projections, head_dim 256, 16 q heads / 4 kv heads (GQA).
- **QK-RMSNorm**: RMSNorm applied to each 256-dim head vector of q and k before
  attention (new vs Prism's current attention).
- **M-RoPE**: rotate only the first 64 of 256 dims, split into sections
  `[11,11,10,0]`. For text-only decoding with a single position stream this
  collapses to a partial RoPE, but the partial/sectioned handling must be correct.
- Otherwise reuses the existing scaled-dot-product attention + KV-cache.

### C. SSM block (24 layers) — *large, the core risk*
1. Input RMSNorm (`attn_norm`).
2. In-projection (`attn_qkv`, 2560→8192) and gate projection (`attn_gate`,
   2560→4096). The 8192 vector splits into the SSM inputs (x + B/C/dt paths); the
   **exact split order must come from the reference**.
3. **Causal depthwise Conv1D** (kernel 4) over the projected channels, with a
   rolling **conv state** (last 3 inputs) carried across tokens. SiLU activation.
4. **Selective SSM scan** (Mamba2-style, grouped): from `ssm_a`, `ssm_alpha`,
   `ssm_beta`, `ssm_dt.bias` compute `dt = softplus(proj·x + dt_bias)`, then the
   recurrence per group (16 groups × state 128):
   `h_t = exp(dt·A)·h_{t-1} + dt·B·x_t`, `y_t = C·h_t`.
   Requires a **recurrent SSM state** `[groups × state_size × head_dim]` per
   sequence.
5. Gated output RMSNorm (`ssm_norm`, 128-dim) multiplied by `SiLU(gate)`.
6. Out-projection (`ssm_out`, 4096→2560).

> The precise parameterization (how `alpha`/`beta`/`a`/`dt` map to `A`, `B`, `C`,
> `dt`; the in-proj channel layout; the grouping) is **not derivable from tensor
> shapes alone** and must be taken from a reference implementation.

### D. Stateful decoding — *invasive*
- SSM layers are **not** stateless like KV-cache attention: each carries a
  **conv state + SSM state** that must persist across generated tokens and reset
  per request/session.
- Prefill must run the SSM recurrence **sequentially** over the prompt.
- Requires a parallel "SSM state cache" per layer alongside the KV-cache, and
  session lifecycle changes in `Prism.Inference` / `Prism.RestServer`.

### Out of scope
- **Multimodal / vision**: the `image-text-to-text` tag implies a vision encoder,
  which ships as a separate `mmproj` GGUF (not present here). Text-only chat does
  not need it.

---

## 7. Effort & risk

- **Effort:** multi-day to multi-week, dominated by section C and D.
- **Primary risk:** the SSM math must match the reference **exactly** — SSMs do
  not degrade gracefully; a wrong sign, order, or activation yields garbage, not
  "slightly worse" output.
- **Secondary:** M-RoPE sections and QK-norm placement must match; stateful decode
  touches the request lifecycle (regression risk for existing Llama models).

---

## 8. Critical dependency

`qwen35` is newer than reliable general knowledge here. The *shape* (hybrid
Mamba2 + gated attention) is clear from the tensors, but the **exact equations**
must come from a reference before writing code:

- llama.cpp's `qwen35` graph builder (`llama.cpp/src/`), and/or
- the Hugging Face `modeling_*.py` for Qwen3.5, and/or
- the model's `config.json` (confirms head counts, splits, activation, norm order).

Implementing from tensor shapes alone would be guesswork.

---

## 9. Recommendation & decision

- If the goal is simply "run a ~4B model on Prism," a **dense** Qwen3 / Qwen2.5 /
  Llama model in K-quant is **orders of magnitude less work** — it runs on the
  current engine today (K-quants now supported).
- Build the `qwen35` engine **only if this specific model is the goal**.

**Suggested next step:** obtain the reference (llama.cpp qwen35 builder + HF
modeling/config) and turn section C/D into exact equations + a step-by-step plan
*before* any implementation.

---

## Appendix A. Inspection method

The GGUF was inspected with a small PowerShell reader (no dependencies):
- header → tensor/metadata counts,
- metadata KVs (values skipped fast to avoid the 248 320-entry token array),
- per-tensor name/dims/type, grouped into per-layer "signatures".

Scripts used during analysis live under the session scratchpad
(`gguf_sig.ps1`, `gguf_full.ps1`).

## Appendix B. Changed files (K-quant support, already merged into working tree)

- `src/Prism.Vector.pas` — `TGgmlType` enum (+`Q4_K/Q5_K/Q6_K`), `QK_K=256`,
  `GetScaleMinK4`, `DequantSuperQ4_K/Q5_K/Q6_K`, `DotRowQ4_K/Q5_K/Q6_K`,
  `RowBytesOf`, `MatVec`, `DequantRow`.
- `src/Prism.Gguf.pas` — map GGUF types 12/13/14; clearer unsupported-type error.
- `src/Prism.Vector.pas` — `HalfToFloatCompute` overflow fix.
