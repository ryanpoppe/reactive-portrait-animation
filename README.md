# Reactive Portrait Animation

Python scaffold for the Reactive Portrait Animation project described in `rpa_paper.docx`.

## What this scaffold includes

- `uv`-managed Python project with a `src` layout
- Typer CLI with `info`, `demo`, and `preprocess` commands
- `pydantic-settings` configuration layer with `.env` support
- placeholder module boundaries for preprocessing, perception, reasoning, speech, and animation
- a committed `.env.example` for local or cloud-backed development
- basic `pytest`, `ruff`, and `mypy` setup

## Quickstart

```bash
uv sync
uv run rpa info
uv run rpa demo
```

Create a local env file with either `cp .env.example .env` on macOS/Linux or `copy .env.example .env` on Windows.

## Common commands

```bash
make install
make test
make lint
make fmt
make typecheck
make check
make run-demo
```

The `Makefile` is intentionally thin and only wraps `uv` or Python commands so it stays portable across Windows, macOS, and Linux.

## Layout

```text
configs/personas/      Persona definitions
src/reactive_portrait_animation/
  animation/           Face animation hooks
  perception/          Audio/video perception hooks
  pipeline/            Runtime orchestration
  preprocessing/       Offline preprocessing hooks
  reasoning/           Character response hooks
  speech/              Speech synthesis hooks
tests/                 Basic scaffold tests
```

## Next steps

1. Replace mock providers with one real LLM + TTS path.
2. Add preprocessing artifacts for masks, keypoints, and background loops.
3. Add latency benchmarking aligned with the paper's evaluation section.
