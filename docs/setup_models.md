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

## 3. JoyVASA + LivePortrait (animation backend)

JoyVASA generates audio-driven motion in LivePortrait's motion space and
renders with LivePortrait's pipeline — one env covers both. Its deps
(insightface, onnxruntime-gpu, Python 3.10 era) conflict with this repo's
Python 3.12 env, so it lives in a **separate conda env**; the repo invokes it
via subprocess (`RPA_ANIMATION_PROVIDER=local`).

### 3.1 Clone + env

```powershell
git clone https://github.com/jdh-algo/JoyVASA ..\JoyVASA
cd ..\JoyVASA
conda create -n joyvasa python=3.10 -y
conda activate joyvasa
python -m pip --version   # MUST show ...\envs\joyvasa\... (3.10), not system Python
python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
# ffmpeg must be on PATH (winget install Gyan.FFmpeg)
```

Gotchas:
- Always `python -m pip`, never bare `pip` — bare pip can resolve to the system
  Python 3.12, where xformers has no wheel and fails building from source.
- torch must be installed BEFORE requirements.txt (xformers needs it at
  install time). xformers 0.0.25.post1 pairs exactly with torch 2.2.2; if it
  still tries a source build, install it from the PyTorch index first:
  `python -m pip install xformers==0.0.25.post1 --index-url https://download.pytorch.org/whl/cu121`
- If `insightface` fails to build, install VS Build Tools or use
  `python -m pip install insightface --prefer-binary`.

### 3.2 Weights (per the JoyVASA README — verify against it)

```powershell
# from the JoyVASA repo root
huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"
huggingface-cli download jdh-algo/JoyVASA --local-dir pretrained_weights/JoyVASA
huggingface-cli download facebook/wav2vec2-base-960h --local-dir pretrained_weights/wav2vec2-base-960h
```

### 3.3 Sanity check inside the JoyVASA env

```powershell
python inference.py -r assets/examples/imgs/joyvasa_001.png -a assets/examples/audios/joyvasa_001.wav --animation_mode human --cfg_scale 2.0
```

### 3.4 Wire into this repo

In `.env`:

```text
RPA_ANIMATION_PROVIDER=local
RPA_JOYVASA_DIR=../JoyVASA
RPA_JOYVASA_PYTHON=C:\Users\<you>\miniconda3\envs\joyvasa\python.exe
```

Then, from this repo (Milestone 2 smoke test):

```powershell
uv run rpa animate data\snape.jpg ..\JoyVASA\assets\examples\audios\joyvasa_001.wav -o data\snape_talks.mp4
```

Notes:
- Model weights reload on every call (minutes, cold). Fine for offline
  preprocessing/Milestone 2; the warm long-lived service for the real-time
  loop comes with Phase 3/4 (`ANIMATION_SERVER_URL` is the reserved seam).
- The provider locates JoyVASA's newest output mp4 and copies it to
  `--output`; if JoyVASA changes its output layout, see
  `animation/joyvasa.py:find_output_video`.

## Troubleshooting

- `RuntimeError: Segmentation dependencies are not installed` → re-run step 1 in this env.
- `torch.cuda.is_available()` is `False` → a CPU torch wheel got in; re-run `uv sync --group preprocess` (never `uv pip install torch` directly).
- Haar cascade finds no face (stylized portraits) → segmentation falls back to a central subject box; check the mask and re-run with a photographic test image if it's poor.
- CUDA OOM is not expected: SAM 2.1 base+ needs ~4GB total.
