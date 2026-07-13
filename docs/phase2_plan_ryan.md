# Phase 2 Implementation Plan — Ryan's Tasks (3-Week Compressed Timeline)

**Scope:** Ryan's Phase 2 assignments from the Gantt chart — face segmentation (SAM2), LivePortrait integration + inference pipeline, and viseme pre-computation — compressed into **Week 1** of the 3-week schedule so Phases 3–5 fit in Weeks 2–3.

**Hardware:** RTX 4070 (16GB VRAM). Ample for every model below.

---

## Model Decisions

| Role | Paper's pick | Decision | Rationale |
|---|---|---|---|
| Face segmentation | SAM2 | **SAM 2.1 (hiera-base-plus)** | One-time offline task; mature tooling; matches paper. SAM 3 (Meta, Nov 2025) adds text-prompted segmentation ("person") and is a drop-in upgrade if point-prompting proves fiddly — one-line paper update. |
| Face animation renderer | LivePortrait | **LivePortrait** (confirmed) | Still the right implicit-keypoint choice: ~12.8ms/frame on 4090 → est. ~18–22ms on 4070, comfortably real-time. Diffusion methods (Hallo3, HunyuanPortrait) remain too slow for the 500ms loop. |
| Audio → motion | Custom Wav2Lip-encoder adaptation (Phase 4) | **JoyVASA pretrained motion generator** (primary), **Ditto** (fallback) | JoyVASA generates audio-driven motion directly in LivePortrait's motion space — it IS the audio→keypoint mapping the paper proposed to build by hand. Adopting it collapses the riskiest Phase 4 task into an integration task. Ditto (Ant Group, ACM MM 2025) is an end-to-end streaming real-time alternative with TensorRT support and low first-frame delay if JoyVASA latency disappoints. |
| Visemes | LivePortrait-warped mouth positions | **Keep, but demoted to fallback** | With a pretrained audio→motion model, cached visemes become the low-latency fallback path rather than the primary mechanism. Cache motion *parameters* (keypoint offsets), not warped pixels — cheaper and composable with idle motion. |
| ASR (Phase 3 preview) | Whisper small/medium | **faster-whisper (small)** + Silero VAD; evaluate **Parakeet TDT 0.6B** | Paper flags ASR as ~half of end-to-end latency. faster-whisper is ~4x faster than openai-whisper at equal accuracy. Parakeet TDT is dramatically faster still for English-only — worth a 1-hour bake-off in Week 2. |
| LLM / TTS | Claude/GPT-4o-mini, ElevenLabs | Unchanged (streaming APIs) | Zach's lane; scaffold config already supports both. |

**Paper impact:** minimal. §4.5's audio-to-motion pathway description changes from "lightweight audio encoder initialized from Wav2Lip" to "pretrained diffusion motion generator (JoyVASA) operating in LivePortrait's motion space" — a defensible improvement, and you keep the viseme cache as an ablation/fallback to discuss.

---

## Week 1 Schedule (Ryan)

### Days 1–2: Environment + SAM segmentation
- Stand up a **separate conda/venv for LivePortrait + SAM 2.1** (they pin older deps — insightface, onnxruntime — that will fight the repo's Python 3.12 env; see Risks). The repo's `animation_server_url` / `whisper_server_url` config already anticipates running models as local services — use that seam.
- Download checkpoints: SAM 2.1 hiera-base-plus, LivePortrait, JoyVASA.
- Implement `preprocessing/segmentation.py`:
  - Input: portrait image path. Output: foreground mask `M` (PNG, soft alpha), background plate `B` (foreground removed), saved under `RPA_CACHE_DIR/<portrait_hash>/`.
  - Prompt strategy: face-detection box (insightface, already a LivePortrait dep) as SAM box prompt; dilate mask to include hair/shoulders; 8–12px feathered edge for Zach's compositing.
- Wire into the existing `rpa preprocess` CLI command.
- **Hand-off to Zach by end of Day 2:** `B` + `M` in an agreed format (document in `docs/interfaces.md` — see Interface Contracts below).

### Days 3–5: LivePortrait integration + inference pipeline
- `animation/liveportrait.py`: wrap LivePortrait inference behind a provider interface matching `RPA_ANIMATION_PROVIDER=local`:
  - `prepare(portrait) -> PortraitAssets`: run appearance encoder + canonical keypoint extraction **once**, cache to disk (paper §4.1 step 4).
  - `render(motion_frame) -> np.ndarray`: warp + decode a single frame from cached assets.
- Day 3: sanity test with LivePortrait's stock **video-driven** mode (known-good path) to validate install and quality on 2–3 test portraits (photo, painting, illustration).
- Day 4: swap driving source to **JoyVASA**: `wav -> motion sequence -> render()` loop. This is the Milestone 2 critical path.
- Day 5: benchmark on the 4070: target ≥30fps sustained at 512×512, log per-stage ms (feature extraction / motion gen / warp+decode) — these numbers go straight into paper Table 1.

### Days 5–6: Viseme pre-computation (fallback path)
- `preprocessing/visemes.py`: drive the portrait with synthetic phoneme audio (espeak-ng per viseme class), capture the resulting **keypoint offsets** for ~15 standard visemes, cache as a small `.npz`.
- Runtime blend util: `viseme_weights -> keypoint offsets` (linear blend). No model inference — this is the guaranteed-latency fallback and a paper ablation.
- Timebox to 1.5 days. If JoyVASA works well, this can shrink to the cache + blend util without polish.

### Days 6–7: Milestone 2 integration with Zach
- Joint: composite `render()` output over Zach's animated background loop using `M` (his compositing task consumes your mask format).
- **Milestone 2 demo:** `rpa animate portrait.jpg speech.wav -> out.mp4` — static photo talks with lip-sync over an animated background.
- Record demo clips + benchmark numbers immediately (they feed §5 and de-risk the paper deadline).

---

## Interface Contracts with Zach (agree Day 1)

1. **Mask/plate format:** `mask.png` (8-bit alpha, feathered), `background_plate.png`, both at source resolution, in `RPA_CACHE_DIR/<portrait_hash>/`.
2. **Compositing API:** Zach's compositor takes `(face_frame, bbox, mask_crop)` per frame; you supply crop coordinates from `prepare()`.
3. **Background loop:** looping mp4/frames at 30fps; your pipeline reads it frame-indexed modulo loop length.

## Risks & Mitigations

- **Dependency hell (highest risk):** LivePortrait/JoyVASA pin Python ≤3.10-era deps; repo is Python 3.12. Mitigation: isolated model env (or Docker) exposing a thin local HTTP/queue service; the scaffold's `ANIMATION_SERVER_URL` config exists for exactly this. Budget half of Day 1 for this; don't fight it inline.
- **JoyVASA latency/chunking:** it generates motion in sliding windows — check first-chunk delay is acceptable for the eventual streaming loop. If not: switch to **Ditto** (streaming-first, TensorRT) — decision gate end of Day 4.
- **Stylized portraits warp badly** (paper §6 already admits this): keep one photographic test portrait as the demo-safe default.
- **Timeline compression:** viseme task is the designated slack — cut polish there first, never the LivePortrait integration (it's the critical path per the Gantt).

## Week 2–3 Preview (Ryan, for context)

- **Week 2:** Whisper/faster-whisper + Silero VAD (Phase 3), character LLM streaming integration; Parakeet bake-off.
- **Week 3:** audio→motion already largely done (JoyVASA) → reallocate saved time to full-pipeline integration, latency profiling, and paper Tables 1–2.
