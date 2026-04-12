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
copy .env.example .env
uv run rpa info
uv run rpa demo
```

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
