# RadCopilot Local

RadCopilot Local is a local-first radiology AI workstation packaged as a single Python application. It launches a local web server, opens a browser-based interface, uses a local Ollama model for inference, can optionally use Whisper for speech-to-text dictation, and supports report drafting, differential diagnosis, guideline lookup, benchmarking, retrieval-augmented generation (RAG), feedback capture, and local history.

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

RadCopilot combines several responsibilities in one launcher:

- **Python bootstrap + server** for startup, routing, logging, and local services
- **Browser UI** for templates, findings entry, report rendering, settings, and review
- **Ollama** for LLM-based generation and repair passes
- **Whisper** for optional speech-to-text
- **RAG layer** for retrieval from findings-to-impression examples
- **Local persistence** through JSONL files, local datasets, and browser localStorage

## Runtime Requirements

### Required

- Python 3.x
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

1. Install Python.
2. Install Ollama.
3. Pull at least one local model in Ollama.
4. Start Ollama if it is not already running.
5. Run the application:

```bash
python RadCopilot_corrected.py
```

6. The app will:
   - verify or start Ollama
   - build the startup RAG index when datasets are available
   - start the local HTTP server
   - open the browser UI

## Main User Workflows

### 1. Report Drafting

The primary workflow.

1. Enter the indication.
2. Enter patient age and sex if needed.
3. Enter or dictate findings.
4. Select or auto-detect a template.
5. Click **Generate Report**.
6. The system will:
   - map findings into sections
   - optionally retrieve similar examples from RAG
   - generate an impression
   - validate the output
   - attempt auto-fix and repair passes if needed
   - render the final report
   - save the result to local history

### 2. Differential Diagnosis

Switch to **Differential** mode and generate ranked differentials from findings.

### 3. Guideline Lookup

Switch to **Guidelines** mode and request guideline-based recommendations.

### 4. Dictation

Use the **Dictate** button to capture audio and transcribe findings when Whisper is available.

### 5. Benchmarking

Load supported dataset files and evaluate generated impressions against reference impressions.

### 6. Train RAG

Point the app to a dataset path to extract findings/impression pairs and expand the persistent RAG library.

## Supported Data and Retrieval Behavior

The application is designed to work with findings-to-impression pairs as its core retrieval unit.

Supported training and benchmark ingestion paths include formats such as:

- CSV
- XML
- TXT
- TGZ / TAR.GZ
- directories of text reports

Retrieval records are grouped by modality and used to provide similar examples during generation.

## Browser Modes

The main interactive modes are:

- **Report**
- **Differential**
- **Guidelines**

The broader UI also exposes:

- **Benchmark**
- **Train RAG**
- **Monitor / Logs / Editor**
- **Settings**

## Key Local Routes

The Python server exposes routes including:

- `/`
- `/index.html`
- `/whisper/status`
- `/whisper/transcribe`
- `/rag/examples`
- `/rag/status`
- `/rag/query`
- `/rag/train`
- `/rag/rate`
- `/benchmark/load`
- `/benchmark/load-path`
- `/benchmark/datasets`
- `/logs/recent`
- `/api/*`

## Local Data and Persistence

RadCopilot stores operational state locally.

- Browser `localStorage` for settings and report history
- JSONL logging for errors and validation events
- Local RAG library data
- Local dataset files for indexing, training, and benchmarking

## Privacy Posture

RadCopilot is designed as a **local-first** workflow.

Intended controls include:

- local Ollama inference
- local file-based storage
- browser localStorage
- PHI scrubbing before model submission

This is an implementation-oriented local privacy posture, not a formal compliance certification.

## Constraints

- The codebase is currently implemented as a very large single-file application
- Correctness depends on prompt logic, templates, and validation behavior
- Local setup is required
- Output quality depends on the selected Ollama model and available retrieval data

## Recommended Next Refactor

The current implementation would benefit from refactoring into smaller modules while preserving a single launcher entrypoint. The cleanest target is a modular monolith rather than many independent services.

## Suggested Project Layout

```text
radcopilot/
  main.py
  config.py
  server/
  ui/
  services/
  rag/
  report/
  benchmark/
```

## Notes

- Keep the terminal window open while RadCopilot is running.
- Use `Ctrl+C` in the terminal to stop the local server.
- If port `7432` is already in use, close other RadCopilot instances and restart.

## Status

This README reflects the current implementation snapshot represented by `RadCopilot_corrected.py`.
