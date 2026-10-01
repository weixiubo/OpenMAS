# OpenAGORA Core

OpenAGORA is a negotiation-driven adaptive multi-agent orchestration runtime.

## What is included
- Stable domain models and protocol interfaces
- Core engine with negotiation, runtime reconfiguration, cache lifecycle and OT-Alignment
- YAML versioned Prompt/Persona templates with strict variable schema validation
- Provider abstraction hardened with timeout/retry/backoff and provider-call audit events
- SQLite + JSONL observability and replay artifacts
- CLI commands: `run`, `replay`, `eval`, `doctor`, `manifest verify`, `export`, `template validate`

## Quick start
```bash
pip install -e .[dev]
openagora doctor --config configs/default.yaml
openagora run --config configs/default.yaml
openagora manifest verify --run-id <RUN_ID> --config configs/default.yaml
openagora export --run-id <RUN_ID> --config configs/default.yaml
```

## Local demo
```bash
python scripts/local_demo.py --config configs/default.yaml
```

## Local live smoke (not for CI)
```bash
pip install -e .[openai]
python scripts/live_smoke_openai.py --config configs/default.yaml --sample-limit 3
```

## Public API stability
OpenAGORA follows SemVer. `OpenAGORAEngine.run/replay` and `RunResult` fields are contract-tested and treated as stable interfaces for v1.x.
