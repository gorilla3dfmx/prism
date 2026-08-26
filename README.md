# Prism — LLM Framework in Pure Delphi (Object Pascal)

Prism is an LLM framework implemented entirely in **Delphi 13** — **without third-party libraries**, using only the units that ship with Delphi (RTL, Indy, FMX). It runs on **Windows, Linux, macOS, Android, and iOS** and is designed for local operation on mobile devices.

## What Prism can do

| Feature | Status |
|---|---|
| **Train** your own GPT-style transformer models (full backprop, AdamW) | ✅ |
| **Load existing trained LLMs**: GGUF binary format (llama.cpp ecosystem) | ✅ |
| Llama-architecture inference: RMSNorm, RoPE, GQA, SwiGLU | ✅ |
| Quantized inference: Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K, F16, F32 (fused kernels) | ✅ |
| **Clustering / layer streaming**: the model does not have to fit entirely in RAM | ✅ |
| **Mixture-of-Experts** ("thematic areas", top-1 routing) incl. training | ✅ |
| **Self-verification** of answers (perplexity, self-consistency, critic) | ✅ |
| REST API compatible with **OpenAI** and **Ollama** (incl. streaming) | ✅ |
| **Multimodal training** via byte-level tokenization (text, image, audio, video, 3D, binary) | ✅ |
| Online fine-tuning via REST (`POST /api/train`) | ✅ |
| **Law layer**: exact expression/formula evaluation, tool calling (`<<calc: ...>>`), law-grounded answer falsification | ✅ |
| Domain-guided expert training (corpora → thematic areas, `x_areas` routing report) | ✅ |
| **GPU backend via Vulkan** (dynamically loaded, no SDK required at runtime) | ✅ |
| GPU kernels for **quantized** GGUF weights (Q4_0/Q4_1/Q8_0/Q4_K/Q5_K/Q6_K/F16/F32), weights resident in VRAM | ✅ |
| GPU backend via OpenCL (F32 only, fallback where Vulkan is missing) | ⚠️ legacy |
| Billion-parameter models | ✅ via GGUF + quantization + streaming (64-bit targets) |

**Realistic expectation:** Prism models you train yourself are *small* models (millions of parameters) — useful for domain-specific assistants, autocomplete, classification, and for learning/experimenting. For "real" conversational quality, load a pre-trained GGUF model (e.g. TinyLlama 1.1B, Qwen2 1.5B, Mistral 7B) — thanks to quantized kernels and layer streaming this also runs on devices with little RAM. With `--gpu` the quantized weights live in VRAM and the MatVecs run as Vulkan compute shaders, which is the difference between "technically works" and "usable" — see [GPU](#gpu) for measured numbers.

---

## Directory structure

```
E:\delphi\projects\prism\
├── README.md
├── src\                        Framework (all units without external dependencies)
│   ├── Prism.Types.pas         Base types, config, parameter layout (Int64), RNG
│   ├── Prism.Vector.pas        Pointer-based vector kernels, quantization (Q4/Q8/F16)
│   ├── Prism.Tensor.pas        Training kernels: forward + backward (llm.c port)
│   ├── Prism.Tokenizer.pas     Custom byte-level BPE tokenizer (multimodal-capable)
│   ├── Prism.Model.pas         .prism checkpoint format, weight provider
│   ├── Prism.Streaming.pas     Layer/expert cluster streaming (LRU) for .prism
│   ├── Prism.Gguf.pas          GGUF reader + SPM/GPT2 tokenizer from metadata
│   ├── Prism.Llama.pas         Llama-architecture engine (GGUF), layer streaming
│   ├── Prism.Inference.pas     Custom engine (incl. MoE), generator, sampling
│   ├── Prism.Train.pas         Trainer (AdamW, MoE backprop), online training
│   ├── Prism.Verify.pas        Self-verification
│   ├── Prism.Multimodal.pas    Multimodal corpus pipeline
│   ├── Prism.Vulkan.Api.pas    Vulkan 1.0 bindings (runtime-loaded, no SDK)
│   ├── Prism.Vulkan.pas        Vulkan compute backend, VRAM residency, self-test
│   ├── Prism.Vulkan.Shaders.inc  GENERATED: SPIR-V for the 8 quant kernels
│   ├── Prism.Gpu.pas           Backend selection (Vulkan, then OpenCL, then CPU)
│   └── Prism.RestServer.pas    REST API (Indy), OpenAI + Ollama compatible
├── shaders\
│   └── matvec.comp             ONE GLSL compute shader, compiled per quant type
├── tools\
│   └── BuildShaders.ps1        glslc → SPIR-V → Prism.Vulkan.Shaders.inc
├── app\
│   ├── PrismTrain.dpr          Training CLI (console)
│   ├── PrismServer.dpr         Server CLI (console)
│   ├── PrismBench.dpr          GPU kernel self-test + CPU/GPU benchmark
│   └── mobile\
│       ├── PrismMobile.dpr     FMX app (Android/iOS/desktop)
│       ├── MainFormU.pas/.fmx
├── data\sample_corpus.txt      Sample training corpus (German)
└── model\                      Storage for models/tokenizers
```

## Compiling

All `.dpr` files reference their units via relative paths — just open them in the IDE:

1. Start **RAD Studio / Delphi 13** → open `app\PrismTrain.dpr` or `app\PrismServer.dpr` (Delphi generates the `.dproj` automatically) → select **Win64** as target → compile.
2. **Use the Release configuration!** The optimization difference is enormous for the compute kernels.
3. Mobile: open `app\mobile\PrismMobile.dpr`, add *Android 64-bit* or *iOS* as target platform, deploy the model files to the documents directory under *Project → Deployment*. Android requires the **INTERNET** permission.

The `.exe` is written next to the `.dpr`, i.e. `app\PrismServer.exe`. If a
change does not seem to take effect, check that timestamp first - a stale binary
there is easy to miss, and `--gpu` on a pre-Vulkan build even reports
`GPU backend active: OpenCL` while a quantized model runs entirely on the CPU.

Command line (example):

```bat
"C:\Program Files (x86)\Embarcadero\Studio\37.0\bin\dcc64.exe" -B -$O+ -U..\src app\PrismServer.dpr
```

> **The GPU shaders need no extra build step.** The SPIR-V for the compute
> kernels is checked in as `src\Prism.Vulkan.Shaders.inc`, so a normal build
> needs nothing but Delphi. Only editing `shaders\matvec.comp` requires the
> Vulkan SDK — see [GPU](#gpu).

> **Important for large models:** Always build 64-bit targets. All offsets/sizes in the code are `Int64` — files > 4 GB and billions of parameters are addressable, but only 64-bit processes can map them.

---

## Quick start A: Load an existing LLM (GGUF)

Get a pre-trained model in GGUF format (e.g. from Hugging Face: `tinyllama-1.1b-chat-v1.0.Q8_0.gguf`) and start:

```bat
PrismServer --model models\tinyllama-1.1b-chat.Q8_0.gguf --ctx 1024
```

Add `--gpu` if you have one — on a 3B model that is the difference
between roughly 1 and roughly 9 tokens/s, see [GPU](#gpu):

```bat
PrismServer --model models\qwen2.5-3b-instruct-q4_k_m.gguf --gpu
```

With limited RAM (e.g. on mobile), enable layer streaming — only N transformer layers are then kept in memory at a time:

```bat
PrismServer --model models\mistral-7b.Q4_0.gguf --ctx 512 --stream-layers 6
```

Supported: GGUF v2/v3, tensor types **F32, F16, Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K** — which covers the usual `Q4_K_M` / `Q5_K_M` downloads; convert anything else (Q2_K, Q3_K, IQ*, …) with `llama-quantize`. Architectures of the Llama family (llama, mistral, qwen2, …), tokenizers `llama` (SentencePiece) and `gpt2` (byte BPE).

## Quick start B: Train your own model

```bat
cd E:\delphi\projects\prism

:: 1. Learn the tokenizer on the corpus (byte BPE)
PrismTrain tokenizer --corpus data\sample_corpus.txt --vocab 512 --out model\tokenizer.json

:: 2. Tokenize the corpus
PrismTrain tokenize --corpus data\sample_corpus.txt --tokenizer model\tokenizer.json --out model\corpus.tokens

:: 3. Initialize the model (here: ~3M parameters; --experts 4 for MoE)
PrismTrain init --tokenizer model\tokenizer.json --dim 192 --layers 6 --heads 6 --seq 256 --experts 1 --out model\model.prism

:: 4. Train (loss should drop well below the starting value ~ln(vocabulary))
PrismTrain train --model model\model.prism --tokens model\corpus.tokens --steps 3000 --batch 4 --seq 128 --lr 0.0003

:: 5. Test
PrismTrain sample --model model\model.prism --tokenizer model\tokenizer.json --chat --prompt "Wer bist du?"

:: 6. Serve it (with self-verification and online training)
PrismServer --model model\model.prism --tokenizer model\tokenizer.json --verify --train
```

The sample corpus is deliberately tiny; it is sufficient for learning (the model memorizes the patterns). For usable results: build your own corpus in the same format (`<|user|>Question<|assistant|>Answer<|eos|>` per line, plus running text).

---

## REST API

The server is a drop-in replacement for OpenAI/Ollama endpoints — existing clients (SDKs, UIs) work without modification.

### OpenAI-compatible

```bash
curl http://localhost:11434/v1/chat/completions -d '{
  "model": "prism",
  "messages": [{"role": "user", "content": "Was ist die Hauptstadt von Deutschland?"}],
  "temperature": 0.7,
  "max_tokens": 128,
  "stream": false,
  "verify": true
}'
```

```cmd
curl http://127.0.0.1:11434/v1/chat/completions -d "{
  \"model\": \"prism\",
  \"messages\": [{\"role\": \"user\", \"content\": \"Was ist die Hauptstadt von Deutschland?\"}],
  \"temperature\": 0.7,
  \"max_tokens\": 128,
  \"stream\": false,
  \"verify\": true
}"
```

With `"verify": true`, the response additionally contains:

```json
"x_verification": {
  "perplexity": 3.412,
  "self_consistency": 0.71,
  "critic_score": 0.83,
  "verdict": "pass"
}
```

**CAUTION:** verification might take 4-5 times more time.

`stream: true` delivers server-sent events (`data: {...}`, terminated with `data: [DONE]`).

### Ollama-compatible

```bash
curl http://localhost:11434/api/chat     -d '{"model":"prism","messages":[{"role":"user","content":"Hallo"}]}'
curl http://localhost:11434/api/generate -d '{"model":"prism","prompt":"Es war einmal"}'
curl http://localhost:11434/api/tags
```

(Streaming via NDJSON, as is standard for Ollama.)

### Training via REST (`POST /api/train`)

Any kind of data can be fed in — text directly, everything else Base64-encoded with a modality tag:

```bash
# Chat pair
curl http://localhost:11434/api/train -d '{"user":"Was ist Prism?","assistant":"Ein LLM-Framework in Delphi."}'

# Running text
curl http://localhost:11434/api/train -d '{"text":"Delphi kompiliert nativ fuer fuenf Plattformen."}'

# Multimodal: image/audio/video/3D/binary + description
curl http://localhost:11434/api/train -d '{"data":"<base64>","modality":"image","description":"Ein roter Wuerfel"}'
```

With `--train` (only `.prism` models in full-memory mode), a background thread fine-tunes immediately and saves the checkpoint; without `--train`, the sample is collected in the corpus (`--corpus`) and trained offline later with `PrismTrain`.

---

## Concepts

### Clustering / memory streaming

Instead of loading the entire model, Prism keeps only a **resident portion** (embeddings, final norm) permanently in RAM. Transformer layers — and with MoE, individual **experts** — are read from disk as clusters on demand and held in **LRU caches** (`--stream-layers N`, `--experts-cache N`). This costs latency per cache miss (disk I/O) but reduces the memory footprint from "entire model" to "N layers + resident". Works for `.prism` and `.gguf`.

### "Thematic areas" = Mixture-of-Experts

`--experts N` at `init` creates N FFN experts per layer plus a router. The router selects **exactly one expert per token** (top-1) — so only a subgraph is ever computed instead of the whole network, and with streaming only the areas that are actually addressed need to be in memory. The specialization of the areas emerges on its own during training (router gradients via softmax backprop). Frequently used areas stay "warm" in the LRU cache — exactly the desired optimization: search in a subgraph instead of a full pass.

**Domain-guided areas:** pass several corpora to `train` — file index = domain = expert:

```bat
PrismTrain train --model model\areas.prism --tokens model\math.tokens,model\facts.tokens --router-aux 0.3 --steps 900
```

The auxiliary router loss (`--router-aux`) pulls each domain's tokens towards "its" expert, so the areas specialize on knowledge domains (math, facts, ...). At inference the router first recognizes *which* area applies, then computes only that subgraph — chat responses report the routing as `"x_areas": [76, 0]` (router decisions per expert for the request).

### The law layer (`Prism.Laws`)

Exact, symbolic knowledge next to the statistical model: an expression evaluator (arithmetic, functions, physical constants) plus a curated formula library (kinetic energy, Ohm's law, ideal gas, pendulum period, ...). Deterministic — the neural model proposes, the law layer computes.

```bash
curl http://localhost:11434/v1/tools/calc -d '{"expression":"0.5*m*v^2","variables":{"m":80,"v":3}}'
curl "http://localhost:11434/v1/laws?q=energy"
curl http://localhost:11434/v1/laws/eval -d '{"law":"kinetic_energy","variables":{"m":80,"v":3}}'
```

**Tool calling** (`"use_tools": true` in chat requests): when the model emits `<<calc: EXPRESSION>>`, the server evaluates it exactly and injects `<<result: VALUE>>` into both the output and the model context, then generation continues. GGUF instruct models get a system prompt teaching the protocol automatically; native Prism models learn it from their training corpus (see `data\tool_corpus.txt` for the sample format). Note: very small instruct models (0.5B) often ignore the protocol — the law-grounded verification below catches their arithmetic anyway.

### Self-verification

Four signals per answer: (1) **Perplexity** — how confident the model was in its own answer (rescoring), (2) **self-consistency** — similarity of alternative samples to the answer, (3) **critic pass** — the model rates its own answer (P("yes") vs. P("no")), (4) **law checks** — arithmetic claims in the answer ("6 mal 7 ergibt 42", "10 / 4 = 2.5") are extracted and re-computed by the law layer. A failed re-computation deterministically falsifies the answer (`verdict: fail`), and verified claims upgrade it — laws beat statistics. Thresholds are configurable in `TVerifier`; with small self-trained models the critic is naturally weak, so the law checks carry the verdict:

```json
"x_verification": {
  "law_checks": { "total": 1, "passed": 0, "failed": 1,
                  "details": ["12 * 3 = 130  [FAIL: expected 36]"] },
  "verdict": "fail"
}
```

### Multimodality (byte-level)

The Prism tokenizer works on **bytes** — so any kind of data can be tokenized. Modalities are framed by markers (`<|img|>…<|/img|>`, `<|aud|>`, `<|vid|>`, `<|3d|>`, `<|bin|>`), and large raw data is reduced via stride sampling. This is the honest, universal entry point; learned encoders (patch/mel embeddings) are the planned next step for serious image/audio quality.

### Inference efficiency

- **KV cache** (computes only against cached keys/values, never re-processes the prompt)
- **Prefill without logits**: while reading in the prompt, the expensive vocabulary projection is skipped entirely
- **Fused quant kernels**: the activation is quantized to int8 once, then pure integer MACs against Q4/Q8 weights — no F32 inflation in RAM
- Pointer-based, unrolled dot products; row-parallel MatVec across all CPU cores
- F16→F32 via lookup table

### GPU

`--gpu` brings up a **Vulkan compute backend**. The Vulkan loader
(`vulkan-1.dll` / `libvulkan.so.1` / MoltenVK) is opened at runtime and every
entry point is resolved dynamically — no third-party library, no SDK on the
user's machine, and if Vulkan is absent Prism simply keeps running on the CPU.

What actually runs on the GPU:

- **Quantized MatVec for all supported GGUF types** — Q4_0, Q4_1, Q8_0, Q4_K,
  Q5_K, Q6_K, F16, F32. The weights are uploaded to VRAM *in their quantized
  form* and dequantized inside the shader. Dequantizing on the host would throw
  away the factor-4 bandwidth advantage that made quantization worth having.
- **Weights stay resident.** Each tensor is uploaded once, on first use, and
  reused for every later token.
- **VRAM fills first-come.** Because the engine walks layers in order, this
  gives llama.cpp's `n_gpu_layers` behaviour for free: whatever fits runs on the
  GPU, the rest falls back to the CPU per tensor. `--gpu-budget MB` caps it; the
  default is derived from the device's VRAM.

  Partial residency is a safety net, not a performance mode. Measured on the
  same 1.83 GB model with `--gpu-budget 300` (60 of 253 tensors resident, 16% of
  the weights): **1.12 tok/s — slightly *worse* than the 1.26 tok/s of pure
  CPU.** The 84% still on the CPU dominates, and the GPU share does not overlap
  with it because the two run in sequence. It keeps a too-large model working
  instead of failing; it does not make it fast. Size the budget so that most of
  the model fits, or leave it at the default.

Options:

```
--gpu                 enable the GPU backend
--gpu-budget 2048     cap VRAM used for weights, in MB (0 = derive from device)
--gpu-device 1        pick a GPU by index or name substring
                      (equivalently: set PRISM_VK_DEVICE)
```

Device selection prefers discrete over integrated, VRAM breaks ties, and an
explicit `--gpu-device` always wins.

**Measured** — Qwen2.5-3B-Instruct Q4_K_M (1.83 GB of weights, fully resident),
same prompt and 140 generated tokens, RTX 3060 Ti vs. a 16-thread CPU. The GPU
row is a range over repeated runs at `--ctx 512` and at the model's default
32768 context; the spread is run-to-run variance, not a context effect:

| | tok/s | per generated token |
|---|---|---|
| CPU only | 1.26 | 791 ms |
| `--gpu` (fully resident) | 8.7 – 9.3 | 108 – 115 ms |
| `--gpu --gpu-budget 300` (16% resident) | 1.12 | 893 ms |

**About 7x end to end.** Isolated MatVec throughput (`PrismBench`, 4096x4096) is
higher — 13x for Q4_K, 27x for Q5_K, 14x for Q6_K — because a benchmark loop
re-reads one hot tensor while real inference streams 253 different tensors per
token.

**Where the latency of a short request goes.** Throughput is not the whole
story: a one-line answer still takes noticeable time, and it pays to know which
part of it. All three rows below are measured end to end over the HTTP API, best
of two warm runs, same model and GPU, at the model's default 32768 context:

| request | prompt / generated | wall clock |
|---|---|---|
| short prompt, 1 token out | 16 / 1 | 1.17 s |
| short prompt, 8 tokens out | 16 / 8 | 1.87 s |
| long prompt, 1 token out | 196 / 1 | 13.78 s |

Rows 1 and 3 differ only in prompt length, rows 1 and 2 only in generated
length, so the two costs separate cleanly:

- **~70 ms per prompt token**
- **~100 ms per generated token**
- fixed per-request cost: effectively zero (it was ~0.3 s until the shared
  generator below)

Cross-check: 16 prompt + 140 generated predicts 15.1 s, measured 15.4 s.

A generated token costs more than a prompt token because prefill skips the
vocabulary projection (`Prefill without logits` above) while generation pays it,
plus the sampling pass — a softmax and top-k/top-p scan over 151936 logits, on
the CPU.

The consequence for short answers: **the prompt dominates.** In the 8-token row
the 16 prompt tokens are 1.1 s of the 1.87 s, about 60% of the wait, and Prism
feeds them through the model one at a time. **Batch prefill** — a matmul over
the whole prompt instead of a token loop — would collapse those 16 passes into
roughly one and bring that request to about 0.8 s. It needs real MatMul kernels
(several columns at once) rather than the current MatVec shaders, so it is a
separate piece of work; it is the single biggest remaining win for short
requests and is on the roadmap.

Two things that used to make this worse and no longer do:

- The server built a **new generator per HTTP request**, and a generator builds
  an engine, and an engine allocates the whole KV cache — `2 x NLayers x CtxLen
  x KvDim` floats, which for a 3B model at the default 32768 context is 2.4 GB
  per request. It is now created once and reused (`SharedGenerator`); safe
  because generations are serialised by `FGenLock` anyway. Worth ~0.3 s per
  request. `TVerifier` had the same problem and the same fix.
- `--ctx N` still helps marginally (a smaller KV cache is cheaper to touch) but
  is no longer worth hundreds of milliseconds per request.

**`verify: true` is expensive by design.** It is not one extra pass over the
answer — it runs a perplexity rescoring, **two additional full generations** for
the self-consistency score, and a critic scoring pass. Measured on the 8-token
answer above: 1.87 s without, **13.4 s with**. That is the intended cost of the
feature, not a bug; leave it off unless you want the verdict.

Be clear about where the remaining gap to llama.cpp is. Of the ~100 ms per
generated token, roughly 40 ms is unavoidable weight reading and ~15 ms is
driver latency from the **one submit + one fence per MatVec** (253 of them per
token). The rest is everything still running on the CPU, each step with a host
round trip: RMSNorm, RoPE, attention/softmax, SwiGLU, and the sampling scan over
151936 logits. All of it is addressable and none of it needs new kernels:

1. Record a whole token as **one command buffer** instead of 253, and keep the
   KV cache in VRAM. Removes the fence latency and most host round trips.
2. Move the elementwise and attention steps into shaders so the activations
   never leave the GPU between MatVecs.
3. Sample on the GPU, so a 151936-element logit vector does not have to come
   back to the host every token.

**Correctness.** The GGUF bit layouts in `shaders/matvec.comp` were transcribed
by hand from `Prism.Vector.pas`, which is exactly the kind of code that can be
wrong in a way that merely looks plausible. So `VulkanInit` runs a **self-test**
before activating: synthetic tensors of all eight types are evaluated on the GPU
and compared against `TQTensor.DequantRow` plus a double-precision dot product.
If any layout disagrees, the backend refuses to activate and Prism stays on the
CPU. Current agreement is 3e-9 to 3e-8 relative. The comparison deliberately
does *not* use `TQTensor.MatVecCpu`: that kernel quantizes the activation to
int8 for integer MACs, so it carries ~2e-4 of its own error, and the GPU path —
which multiplies in float — is in fact the more accurate of the two.

Run it yourself:

```
PrismBench                  self-test + benchmark on the best GPU
PrismBench --verify-only    self-test only
PrismBench --device 1       pick a GPU
```

**Editing the shaders.** `shaders/matvec.comp` is one source compiled eight
times (`-DQTYPE=<ggml type id>`); `tools/BuildShaders.ps1` runs `glslc` and
writes the SPIR-V into `src/Prism.Vulkan.Shaders.inc` as Delphi const arrays.
That `.inc` is **committed on purpose** — building Prism needs nothing but
Delphi. Only someone changing a kernel needs the Vulkan SDK, and only then:

```
powershell -ExecutionPolicy Bypass -File tools\BuildShaders.ps1
```

One warning if you do touch it: the kernel is bound by its memory *access
pattern*. The lane decomposition is arranged so that adjacent lanes read
adjacent dwords; the obvious "lane t handles block t" version puts them
32/144/210 bytes apart and cost a measured 3-4x. The comment at the top of the
shader explains the layout before you change it.

**Online training vs. GPU.** `Prism.Train` updates the weights *in place*, and
`TFullWeights.Params` is a dynamic array shared by reference with the inference
path — the exact buffer whose contents the backend has cached in VRAM. The
optimizer step therefore ends with `Backend.InvalidateWeights(Pointer(FParams))`
so the next inference re-uploads. Without it, `--train --gpu` would keep serving
the pre-update weights from VRAM and produce silently stale output. (The same
was true of the older OpenCL weight cache; the fix covers both.)

**Streaming vs. GPU.** `--stream-layers N` and `--gpu` pull against each other:
an evicted layer must release its VRAM copy too, so a small `N` means
re-uploading over PCIe on every visit. If the model fits in VRAM, leave
`--stream-layers` at 0.

**OpenCL** remains as a fallback for machines with an OpenCL driver but no
Vulkan one. It only accelerates the F32 path of your own models; quantized GGUF
models stay on the CPU there, and the server logs a note saying so.

---

## Limits & roadmap

- **Don't expect miracles:** Training on a CPU will not reach GPT quality. The strength is the complete, understandable, compile-anywhere stack plus running pre-trained GGUF models.
- Roadmap, GPU: one command buffer per token instead of one per MatVec; KV cache in VRAM; elementwise/attention steps as shaders so activations stop round-tripping to the host; F16 accumulation and subgroup reductions where the device supports them.
- Roadmap, other: **batch prefill** (a matmul over the whole prompt instead of a token loop) - the biggest single win left for short requests, where the prompt accounts for roughly 60% of the wait; learned multimodal encoders, MoE load-balancing loss, GGUF export of your own models, speculative decoding.
- The GPT-2 BPE pretokenizer is simplified (no full regex) — tokenization can deviate minimally from the original in edge cases.
- The server serializes generations (one request computes exclusively); parallel sessions would share the CPU — or the single GPU queue — anyway.

## License / origin

Original development in Object Pascal. The training math follows the GPT-2 reference design (llm.c by A. Karpathy, MIT); GGUF is the open format of the llama.cpp project.
