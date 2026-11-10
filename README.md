# RadCopilot Local

RadCopilot Local is a local-first radiology AI workstation packaged as a modular Python application. It launches a local web server, opens a browser-based interface, uses a local Ollama model for inference, can optionally use Whisper for speech-to-text dictation, and supports report drafting, differential diagnosis, guideline lookup, benchmarking, retrieval-augmented generation (RAG), feedback capture, and local history.

## Reviewer demo

This is a local engineering prototype. Use synthetic data for demonstrations.

From a clean checkout, run this smoke test with Python 3.10 or later:

```bash
python -m unittest discover -s tests -v
```

The test starts two loopback HTTP servers on temporary ports. It fetches the
page and its CSS and JavaScript, lists a fixture model, and sends the synthetic
text `Synthetic test input.` through the app's Ollama adapter. The expected
reply is `Synthetic fixture response.` No model download, API key, patient
data, or third-party package is needed. The fixture checks HTTP wiring and
static assets. Model quality, dictation, and clinical validity are outside
this demo.

## Highlights

- Local browser UI served by Python
- Local Ollama inference
- Optional Whisper-based dictation
- Report drafting workflow with template-driven structure
- Differential diagnosis mode
- Guideline lookup mode
- Startup RAG index from local datasets
- Persistent RAG library with training and rating support
- Benchmark dataset loading and scoring
- Local settings, logs, and history

## Architecture Summary

RadCopilot combines several responsibilities in one local modular application:

- **Python bootstrap + server** for startup, routing, logging, and local services
- **Browser UI** for templates, findings entry, report rendering, settings, and review
- **Ollama** for LLM-based generation and repair passes
- **Whisper** for optional speech-to-text
- **RAG layer** for retrieval from findings-to-impression examples
- **Local persistence** through JSONL files, local datasets, history storage, and browser localStorage

## Runtime Requirements

### Required

- Python 3.10 or later
- [Ollama](https://ollama.com/) installed and available on the local machine
- A local model pulled into Ollama
- A modern web browser

### Optional

- `openai-whisper` for dictation
- `scikit-learn` and `numpy` for retrieval/vectorization features

## Default Runtime Settings

- **Local server port:** `7432`
- **Default Ollama URL:** `http://localhost:11434`
- **Default startup template:** `CT Chest`

## Quick Start

1. Install Python 3.10 or later.
2. Install Ollama.
3. Pull at least one local model in Ollama.
4. Start Ollama if it is not already running.
5. Run the application:

```bash
python -m radcopilot
