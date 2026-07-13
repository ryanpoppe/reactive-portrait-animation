# GPU Model Setup (Windows, RTX 4070)

One-time setup on the machine that runs preprocessing/inference.

## 1. Sync everything (numpy, opencv, CUDA torch, sam2)

```powershell
uv sync --group preprocess
```

`pyproject.toml` pins torch/torchvision to the cu124 index and sam2 to its git
repo via `[tool.uv.sources]`, so a single sync resolves them together — do NOT
`uv pip install` torch or sam2 manually (ad-hoc installs let sam2's PyPI
resolution replace the CUDA torch wheel with the CPU build).

Note: plain `uv sync` (without `--group preprocess`) prunes these packages
again. Always include the group on the GPU machine.

Verify: `uv run python -c "import torch; print(torch.cuda.is_available())"` → `True`.

## 2. SAM 2.1 checkpoint

```powershell
mkdir models
curl -L -o models/sam2.1_hiera_base_plus.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

Config defaults already point at `models/sam2.1_hiera_base_plus.pt` and the
`sam2.1_hiera_b+.yaml` model config (bundled inside the sam2 package).

Smoke test:

```powershell
uv run rpa preprocess data\test_portrait.jpg
```

Expected output: `mask.png`, `background_plate.png`, `metadata.json` under
`.cache/rpa/<hash>/`. Inspect the mask edge — it should be soft (feathered),
and cover hair and shoulders, not just the face.

## 3. LivePortrait + JoyVASA (Days 3–5)

These pin older dependencies (insightface, onnxruntime-gpu) that conflict with
this repo's Python 3.12 env. Run them in a **separate** env exposed as a local
service (`ANIMATION_SERVER_URL`, default `http://localhost:7000`):

```powershell
git clone https://github.com/KwaiVGI/LivePortrait ..\LivePortrait
git clone https://github.com/jdh-algo/JoyVASA ..\JoyVASA
# follow each repo's README with a dedicated conda env (python 3.10)
```

Wrapper/service implementation lands in `animation/` on Day 3 per
`docs/phase2_plan_ryan.md`.

## Troubleshooting

- `RuntimeError: Segmentation dependencies are not installed` → re-run step 1 in this env.
- `torch.cuda.is_available()` is `False` → a CPU torch wheel got in; re-run `uv sync --group preprocess` (never `uv pip install torch` directly).
- Haar cascade finds no face (stylized portraits) → segmentation falls back to a central subject box; check the mask and re-run with a photographic test image if it's poor.
- CUDA OOM is not expected: SAM 2.1 base+ needs ~4GB total.
