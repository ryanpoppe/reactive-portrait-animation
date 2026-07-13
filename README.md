# Reactive Portrait Animation

Python scaffold for the Reactive Portrait Animation project described in `rpa_paper.docx`.

## What this scaffold includes

- `uv`-managed Python project with a `src` layout
- Typer CLI with `info`, `demo`, and `preprocess` commands
- SAM 2.1 portrait segmentation (`preprocessing/segmentation.py`): foreground mask + background plate
- `pydantic-settings` configuration layer with `.env` support
- placeholder module boundaries for perception, reasoning, speech, and animation
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

## Preprocessing (offline segmentation)

One-time setup on the GPU machine (full walkthrough in `docs/setup_models.md`):

```bash
uv sync --group preprocess   # numpy, opencv, CUDA torch, sam2 -- sources pinned in pyproject.toml
mkdir models
curl -L -o models/sam2.1_hiera_base_plus.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

Notes: always include `--group preprocess` on the GPU machine (plain `uv sync` prunes
those packages), and never `uv pip install` torch or sam2 manually -- ad-hoc installs
let sam2's PyPI resolution replace the CUDA torch wheel with the CPU build.

Then segment a portrait:

```bash
uv run rpa preprocess data/portrait.jpg
```

Writes to `.cache/rpa/<portrait_hash>/` (format contract in `docs/interfaces.md`):

- `mask.png` -- feathered foreground mask (subject = white)
- `background_plate.png` -- background with the subject zeroed out, ready for inpainting
- `metadata.json` -- face/subject boxes, SAM prompt points, artifact paths

The CLI prints a warning if the mask misses any positive prompt point -- inspect
`mask.png` before using such a result. Bust-style portraits on plain backgrounds
segment most reliably; low-contrast subject/background combinations (see
`data/lockheart.jpg`) are a known failure case.

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

1. LivePortrait integration + inference pipeline (`animation/`, see `docs/phase2_plan_ryan.md`).
2. Background inpainting + animation loop from `background_plate.png` (Zach).
3. Replace mock providers with one real LLM + TTS path.
4. Add latency benchmarking aligned with the paper's evaluation section.
