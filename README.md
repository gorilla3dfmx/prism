# Prism — LLM Framework in Pure Delphi (Object Pascal)

Prism is an LLM framework implemented entirely in **Delphi 13** — **without third-party libraries**, using only the units that ship with Delphi (RTL, Indy, FMX). It runs on **Windows, Linux, macOS, Android, and iOS** and is designed for local operation on mobile devices.

## What Prism can do

| Feature | Status |
|---|---|
| **Train** your own GPT-style transformer models (full backprop, AdamW) | ✅ |
| **Load existing trained LLMs**: GGUF binary format (llama.cpp ecosystem) | ✅ |
| Llama-architecture inference: RMSNorm, RoPE, GQA, SwiGLU | ✅ |
| Quantized inference: Q8_0, Q4_0, Q4_1, F16, F32 (fused integer kernels) | ✅ |
| **Clustering / layer streaming**: the model does not have to fit entirely in RAM | ✅ |
| **Mixture-of-Experts** ("thematic areas", top-1 routing) incl. training | ✅ |
| **Self-verification** of answers (perplexity, self-consistency, critic) | ✅ |
| REST API compatible with **OpenAI** and **Ollama** (incl. streaming) | ✅ |
| **Multimodal training** via byte-level tokenization (text, image, audio, video, 3D, binary) | ✅ |
| Online fine-tuning via REST (`POST /api/train`) | ✅ |
| **Law layer**: exact expression/formula evaluation, tool calling (`<<calc: ...>>`), law-grounded answer falsification | ✅ |
| Domain-guided expert training (corpora → thematic areas, `x_areas` routing report) | ✅ |
| GPU backend via OpenCL (dynamically loaded, no SDK required) | ⚠️ experimental |
| Billion-parameter models | ✅ via GGUF + quantization + streaming (64-bit targets) |

**Realistic expectation:** Prism models you train yourself are *small* models (millions of parameters) — useful for domain-specific assistants, autocomplete, classification, and for learning/experimenting. For "real" conversational quality, load a pre-trained GGUF model (e.g. TinyLlama 1.1B, Qwen2 1.5B, Mistral 7B) — thanks to quantized kernels and layer streaming this also runs on devices with little RAM, though CPU-bound and therefore slower than llama.cpp.

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
│   ├── Prism.Gpu.pas           OpenCL backend (dynamically loaded)
│   └── Prism.RestServer.pas    REST API (Indy), OpenAI + Ollama compatible
├── app\
│   ├── PrismTrain.dpr          Training CLI (console)
│   ├── PrismServer.dpr         Server CLI (console)
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

Command line (example):

```bat
"C:\Program Files (x86)\Embarcadero\Studio\37.0\bin\dcc64.exe" -B -$O+ -U..\src app\PrismServer.dpr
```

> **Important for large models:** Always build 64-bit targets. All offsets/sizes in the code are `Int64` — files > 4 GB and billions of parameters are addressable, but only 64-bit processes can map them.

---

## Quick start A: Load an existing LLM (GGUF)

Get a pre-trained model in GGUF format (e.g. from Hugging Face: `tinyllama-1.1b-chat-v1.0.Q8_0.gguf`) and start:

```bat
PrismServer --model models\tinyllama-1.1b-chat.Q8_0.gguf --ctx 1024
```

With limited RAM (e.g. on mobile), enable layer streaming — only N transformer layers are then kept in memory at a time:

```bat
PrismServer --model models\mistral-7b.Q4_0.gguf --ctx 512 --stream-layers 6
```

Supported: GGUF v2/v3, tensor types **F32, F16, Q4_0, Q4_1, Q8_0** (please convert others with `llama-quantize`), architectures of the Llama family (llama, mistral, qwen2, …), tokenizers `llama` (SentencePiece) and `gpt2` (byte BPE).

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

With `"verify": true`, the response additionally contains:

```json
"x_verification": {
  "perplexity": 3.412,
  "self_consistency": 0.71,
  "critic_score": 0.83,
  "verdict": "pass"
}
```

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

`--gpu` attempts to dynamically load the system OpenCL (Windows `OpenCL.dll`, Linux/Android `libOpenCL.so`, macOS framework) — no third-party library, no SDK installation. Currently accelerates the large F32 MatVecs of your own models; weight buffers are cached on the GPU. Quantized GGUF kernels still run on the CPU. iOS has no OpenCL → automatic CPU fallback (Metal backend: roadmap).

---

## Limits & roadmap

- **Don't expect miracles:** Training on a CPU will not reach GPT quality. The strength is the complete, understandable, compile-anywhere stack plus running pre-trained GGUF models.
- Roadmap: Q4_K/Q5_K/Q6_K quantization, OpenCL kernels for quant MatVec, Metal backend (iOS/macOS), batch prefill (matmul instead of token loop), learned multimodal encoders, MoE load-balancing loss, GGUF export of your own models, speculative decoding.
- The GPT-2 BPE pretokenizer is simplified (no full regex) — tokenization can deviate minimally from the original in edge cases.
- The server serializes generations (one request computes exclusively); parallel sessions would share the CPU anyway.

## License / origin

Original development in Object Pascal. The training math follows the GPT-2 reference design (llm.c by A. Karpathy, MIT); GGUF is the open format of the llama.cpp project.
