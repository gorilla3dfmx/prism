# Prism — LLM-Framework in reinem Delphi (Object Pascal)

Prism ist ein komplett in **Delphi 13** implementiertes LLM-Framework — **ohne Third-Party-Libraries**, nur mit den in Delphi enthaltenen Units (RTL, Indy, FMX). Es läuft auf **Windows, Linux, macOS, Android und iOS** und ist für den lokalen Betrieb auf mobilen Geräten ausgelegt.

## Was Prism kann

| Feature | Status |
|---|---|
| Eigene GPT-artige Transformer-Modelle **trainieren** (voller Backprop, AdamW) | ✅ |
| **Existierende trainierte LLMs laden**: GGUF-Binärformat (llama.cpp-Ökosystem) | ✅ |
| Llama-Architektur-Inferenz: RMSNorm, RoPE, GQA, SwiGLU | ✅ |
| Quantisierte Inferenz: Q8_0, Q4_0, Q4_1, F16, F32 (Fused-Integer-Kernels) | ✅ |
| **Clustering / Layer-Streaming**: Modell muss nicht komplett in den RAM | ✅ |
| **Mixture-of-Experts** („thematische Areale", Top-1-Routing) inkl. Training | ✅ |
| **Selbst-Verifikation** der Antworten (Perplexität, Selbstkonsistenz, Critic) | ✅ |
| REST-API kompatibel zu **OpenAI** und **Ollama** (inkl. Streaming) | ✅ |
| **Multimodales Training** über Byte-Level-Tokenisierung (Text, Bild, Audio, Video, 3D, Binär) | ✅ |
| Online-Finetuning über REST (`POST /api/train`) | ✅ |
| GPU-Backend über OpenCL (dynamisch geladen, kein SDK nötig) | ⚠️ experimentell |
| Milliarden-Parameter-Modelle | ✅ über GGUF + Quantisierung + Streaming (64-Bit-Targets) |

**Realistische Erwartung:** Selbst trainierte Prism-Modelle sind *kleine* Modelle (Millionen Parameter) — sinnvoll für domänenspezifische Assistenten, Autocomplete, Klassifikation und zum Lernen/Experimentieren. Für „echte" Konversationsqualität lädt man ein fertig trainiertes GGUF-Modell (z. B. TinyLlama 1.1B, Qwen2 1.5B, Mistral 7B) — das läuft dank quantisierter Kernels und Layer-Streaming auch auf Geräten mit wenig RAM, CPU-bedingt aber langsamer als llama.cpp.

---

## Verzeichnisstruktur

```
E:\delphi\projects\llm\
├── README.md
├── src\                        Framework (alle Units ohne Fremdabhängigkeiten)
│   ├── Prism.Types.pas         Basistypen, Config, Parameter-Layout (Int64), RNG
│   ├── Prism.Vector.pas        Pointer-basierte Vektor-Kernels, Quantisierung (Q4/Q8/F16)
│   ├── Prism.Tensor.pas        Trainings-Kernels: Forward + Backward (llm.c-Port)
│   ├── Prism.Tokenizer.pas     Eigener Byte-Level-BPE-Tokenizer (multimodal-fähig)
│   ├── Prism.Model.pas         .prism-Checkpoint-Format, Gewichts-Provider
│   ├── Prism.Streaming.pas     Layer-/Experten-Cluster-Streaming (LRU) für .prism
│   ├── Prism.Gguf.pas          GGUF-Reader + SPM/GPT2-Tokenizer aus Metadaten
│   ├── Prism.Llama.pas         Llama-Architektur-Engine (GGUF), Layer-Streaming
│   ├── Prism.Inference.pas     Eigene Engine (inkl. MoE), Generator, Sampling
│   ├── Prism.Train.pas         Trainer (AdamW, MoE-Backprop), Online-Training
│   ├── Prism.Verify.pas        Selbst-Verifikation
│   ├── Prism.Multimodal.pas    Multimodale Korpus-Pipeline
│   ├── Prism.Gpu.pas           OpenCL-Backend (dynamisch geladen)
│   └── Prism.RestServer.pas    REST-API (Indy), OpenAI + Ollama kompatibel
├── app\
│   ├── PrismTrain.dpr          Trainings-CLI (Konsole)
│   ├── PrismServer.dpr         Server-CLI (Konsole)
│   └── mobile\
│       ├── PrismMobile.dpr     FMX-App (Android/iOS/Desktop)
│       ├── MainFormU.pas/.fmx
├── data\sample_corpus.txt      Beispiel-Trainingskorpus (deutsch)
└── model\                      Ablage für Modelle/Tokenizer
```

## Kompilieren

Alle `.dpr`-Dateien referenzieren ihre Units mit relativen Pfaden — einfach in der IDE öffnen:

1. **RAD Studio / Delphi 13** starten → `app\PrismTrain.dpr` bzw. `app\PrismServer.dpr` öffnen (Delphi erzeugt die `.dproj` automatisch) → Ziel **Win64** wählen → kompilieren.
2. **Release-Konfiguration verwenden!** Der Optimierungsunterschied ist bei den Rechenkernels enorm.
3. Mobile: `app\mobile\PrismMobile.dpr` öffnen, Zielplattform *Android 64-Bit* oder *iOS* hinzufügen, unter *Projekt → Deployment* die Modelldateien ins Dokumente-Verzeichnis deployen. Android braucht die Berechtigung **INTERNET**.

Kommandozeile (Beispiel):

```bat
"C:\Program Files (x86)\Embarcadero\Studio\37.0\bin\dcc64.exe" -B -$O+ -U..\src app\PrismServer.dpr
```

> **Wichtig für große Modelle:** Immer 64-Bit-Targets bauen. Alle Offsets/Größen im Code sind `Int64` — Dateien > 4 GB und Milliarden Parameter sind adressierbar, aber nur 64-Bit-Prozesse können sie mappen.

---

## Schnellstart A: Existierendes LLM laden (GGUF)

Ein fertig trainiertes Modell im GGUF-Format besorgen (z. B. von Hugging Face: `tinyllama-1.1b-chat-v1.0.Q8_0.gguf`) und starten:

```bat
PrismServer --model models\tinyllama-1.1b-chat.Q8_0.gguf --ctx 1024
```

Bei wenig RAM (z. B. mobil) Layer-Streaming aktivieren — es liegen dann nur N Transformer-Layer gleichzeitig im Speicher:

```bat
PrismServer --model models\mistral-7b.Q4_0.gguf --ctx 512 --stream-layers 6
```

Unterstützt: GGUF v2/v3, Tensor-Typen **F32, F16, Q4_0, Q4_1, Q8_0** (andere bitte mit `llama-quantize` umwandeln), Architekturen der Llama-Familie (llama, mistral, qwen2, …), Tokenizer `llama` (SentencePiece) und `gpt2` (Byte-BPE).

## Schnellstart B: Eigenes Modell trainieren

```bat
cd E:\delphi\projects\llm

:: 1. Tokenizer auf dem Korpus lernen (Byte-BPE)
PrismTrain tokenizer --corpus data\sample_corpus.txt --vocab 512 --out model\tokenizer.json

:: 2. Korpus tokenisieren
PrismTrain tokenize --corpus data\sample_corpus.txt --tokenizer model\tokenizer.json --out model\corpus.tokens

:: 3. Modell initialisieren (hier: ~3M Parameter; --experts 4 für MoE)
PrismTrain init --tokenizer model\tokenizer.json --dim 192 --layers 6 --heads 6 --seq 256 --experts 1 --out model\model.prism

:: 4. Trainieren (Loss sollte deutlich unter den Startwert ~ln(Vokabular) fallen)
PrismTrain train --model model\model.prism --tokens model\corpus.tokens --steps 3000 --batch 4 --seq 128 --lr 0.0003

:: 5. Testen
PrismTrain sample --model model\model.prism --tokenizer model\tokenizer.json --chat --prompt "Wer bist du?"

:: 6. Als Server bereitstellen (mit Selbst-Verifikation und Online-Training)
PrismServer --model model\model.prism --tokenizer model\tokenizer.json --verify --train
```

Der Beispiel-Korpus ist bewusst winzig; fürs Lernen reicht er (das Modell memoriert die Muster). Für brauchbare Ergebnisse: eigenen Korpus im gleichen Format aufbauen (`<|user|>Frage<|assistant|>Antwort<|eos|>` pro Zeile, plus Fließtext).

---

## REST-API

Der Server ist ein Drop-in-Ersatz für OpenAI-/Ollama-Endpunkte — bestehende Clients (SDKs, UIs) funktionieren ohne Anpassung.

### OpenAI-kompatibel

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

Antwort enthält bei `"verify": true` zusätzlich:

```json
"x_verification": {
  "perplexity": 3.412,
  "self_consistency": 0.71,
  "critic_score": 0.83,
  "verdict": "pass"
}
```

`stream: true` liefert Server-Sent Events (`data: {...}`, abgeschlossen mit `data: [DONE]`).

### Ollama-kompatibel

```bash
curl http://localhost:11434/api/chat     -d '{"model":"prism","messages":[{"role":"user","content":"Hallo"}]}'
curl http://localhost:11434/api/generate -d '{"model":"prism","prompt":"Es war einmal"}'
curl http://localhost:11434/api/tags
```

(Streaming per NDJSON, wie bei Ollama Standard.)

### Training über REST (`POST /api/train`)

Jede Datenart kann eingespeist werden — Text direkt, alles andere Base64-kodiert mit Modalitäts-Angabe:

```bash
# Chat-Paar
curl http://localhost:11434/api/train -d '{"user":"Was ist Prism?","assistant":"Ein LLM-Framework in Delphi."}'

# Fließtext
curl http://localhost:11434/api/train -d '{"text":"Delphi kompiliert nativ fuer fuenf Plattformen."}'

# Multimodal: Bild/Audio/Video/3D/Binär + Beschreibung
curl http://localhost:11434/api/train -d '{"data":"<base64>","modality":"image","description":"Ein roter Wuerfel"}'
```

Mit `--train` (nur `.prism`-Modelle im Full-Memory-Modus) finetuned ein Hintergrund-Thread sofort und speichert den Checkpoint; ohne `--train` wird das Sample im Korpus (`--corpus`) gesammelt und später offline mit `PrismTrain` trainiert.

---

## Konzepte

### Clustering / Speicher-Streaming

Statt das komplette Modell zu laden, hält Prism nur einen **Resident-Anteil** (Embeddings, finale Norm) dauerhaft im RAM. Transformer-Layer — und bei MoE einzelne **Experten** — werden als Cluster bedarfsweise von der Platte gelesen und in **LRU-Caches** gehalten (`--stream-layers N`, `--experts-cache N`). Kostet Latenz pro Cache-Miss (Platten-I/O), reduziert den Speicherbedarf aber von „Gesamtmodell" auf „N Layer + Resident". Funktioniert für `.prism` und `.gguf`.

### „Thematische Areale" = Mixture-of-Experts

`--experts N` beim `init` erzeugt pro Layer N FFN-Experten plus einen Router. Der Router wählt **pro Token genau einen Experten** (Top-1) — es wird also immer nur ein Untergraph gerechnet statt des ganzen Netzes, und beim Streaming müssen nur die tatsächlich angesprochenen Areale im Speicher liegen. Die Spezialisierung der Areale entsteht im Training von selbst (Router-Gradienten via Softmax-Backprop). Häufig genutzte Areale bleiben im LRU-Cache „warm" — genau die gewünschte Optimierung: Suche im Untergraph statt Volldurchlauf.

### Selbst-Verifikation

Drei unabhängige Signale pro Antwort: (1) **Perplexität** — wie sicher war das Modell bei der eigenen Antwort (Rescoring), (2) **Selbstkonsistenz** — Ähnlichkeit alternativer Samples zur Antwort, (3) **Critic-Pass** — das Modell bewertet seine Antwort selbst (P(„ja") vs. P(„nein")). Daraus wird `pass`/`warn`/`fail` abgeleitet; Schwellwerte in `TVerifier` konfigurierbar. Bei kleinen selbst trainierten Modellen ist der Critic naturgemäß schwach — Perplexität und Konsistenz tragen dann die Aussage.

### Multimodalität (Byte-Level)

Der Prism-Tokenizer arbeitet auf **Bytes** — damit ist jede Datenart tokenisierbar. Modalitäten werden durch Marker eingerahmt (`<|img|>…<|/img|>`, `<|aud|>`, `<|vid|>`, `<|3d|>`, `<|bin|>`), große Rohdaten werden per Stride-Sampling reduziert. Das ist der ehrliche, universelle Einstieg; gelernte Encoder (Patch-/Mel-Embeddings) sind der geplante nächste Schritt für ernsthafte Bild-/Audio-Qualität.

### Effizienz der Inferenz

- **KV-Cache** (rechnet nur gegen gecachte Keys/Values, nie den Prompt neu)
- **Prefill ohne Logits**: beim Einlesen des Prompts entfällt die teure Vokabular-Projektion komplett
- **Fused-Quant-Kernels**: Aktivierung wird einmal nach int8 quantisiert, dann reine Integer-MACs gegen Q4/Q8-Gewichte — kein F32-Aufblasen im RAM
- Pointer-basierte, entrollte Skalarprodukte; zeilenparallele MatVec über alle CPU-Kerne
- F16→F32 per Lookup-Tabelle

### GPU

`--gpu` versucht das System-OpenCL dynamisch zu laden (Windows `OpenCL.dll`, Linux/Android `libOpenCL.so`, macOS Framework) — keine Third-Party-Library, keine SDK-Installation. Beschleunigt derzeit die großen F32-MatVecs eigener Modelle; Gewichts-Buffer werden auf der GPU gecacht. Quantisierte GGUF-Kernels laufen noch auf der CPU. iOS hat kein OpenCL → automatischer CPU-Fallback (Metal-Backend: Roadmap).

---

## Grenzen & Roadmap

- **Kein Wunder erwarten:** Training auf einer CPU erreicht keine GPT-Qualität. Die Stärke ist der komplette, verständliche, überall kompilierbare Stack plus das Ausführen fertiger GGUF-Modelle.
- Roadmap: Q4_K/Q5_K/Q6_K-Quantisierung, OpenCL-Kernels für Quant-MatVec, Metal-Backend (iOS/macOS), Batch-Prefill (Matmul statt Token-Schleife), gelernte multimodale Encoder, MoE-Load-Balancing-Loss, GGUF-Export eigener Modelle, Speculative Decoding.
- GPT-2-BPE-Pretokenizer ist vereinfacht (kein volles Regex) — Tokenisierung kann in Randfällen minimal vom Original abweichen.
- Der Server serialisiert Generierungen (ein Request rechnet exklusiv); parallele Sessions teilen sich die CPU ohnehin.

## Lizenz / Herkunft

Eigenentwicklung in Object Pascal. Die Trainings-Mathematik folgt dem GPT-2-Referenzdesign (llm.c von A. Karpathy, MIT); GGUF ist das offene Format des llama.cpp-Projekts.
