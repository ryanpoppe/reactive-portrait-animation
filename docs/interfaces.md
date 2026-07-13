# Module Interface Contracts (Ryan ⇄ Zach)

Agreed formats for the Phase 2 hand-offs. Change only by updating this file first.

## 1. Segmentation output (Ryan → Zach)

Produced by `rpa preprocess <portrait>` into `<RPA_CACHE_DIR>/<portrait_hash>/`:

| File | Format | Notes |
|---|---|---|
| `mask.png` | 8-bit grayscale, source resolution | Foreground mask `M`. Feathered edge (default 10px Gaussian, `RPA_FEATHER_RADIUS`). 255 = subject, 0 = background, intermediate = blend zone. |
| `background_plate.png` | 8-bit BGR, source resolution | Background plate `B`. Subject pixels zeroed where `mask > 127` — inpaint these before animating. |
| `metadata.json` | JSON | `portrait_hash`, `source_image`, `face_box` and `subject_box` as `[x0, y0, x1, y1]`, `width`, `height`, `feather_radius`, artifact paths. |

`portrait_hash` = first 16 hex chars of SHA-256 of the image bytes; same image ⇒ same cache dir on both machines.

## 2. Background loop (Zach → Ryan)

Write to the same cache dir:

- `background_loop.mp4` — inpainted + animated background, 30fps, source resolution, seamless loop (first ≈ last frame). 5–15s.
- Runtime reads it frame-indexed `t mod loop_length`; no audio track.

## 3. Compositing (Zach consumes Ryan's renderer output)

Per frame, animation side provides:

```python
face_frame: np.ndarray  # BGR, crop-sized, from LivePortrait render()
crop_box: tuple[int, int, int, int]  # where the crop sits in source coords (metadata.json subject_box-derived; final key TBD Day 4)
```

Compositor alpha-blends `face_frame` over the current background-loop frame using the crop of `mask.png`, feathered edge as alpha ramp.

## 4. Config keys added (already in `config.py`)

- `RPA_SAM_CHECKPOINT` (default `models/sam2.1_hiera_base_plus.pt`)
- `RPA_SAM_MODEL_CFG` (default `configs/sam2.1/sam2.1_hiera_b+.yaml`)
- `RPA_FEATHER_RADIUS` (default `10`)
