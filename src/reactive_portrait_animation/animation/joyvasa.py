"""JoyVASA subprocess backend.

JoyVASA generates audio-driven motion in LivePortrait's implicit-keypoint
motion space and renders with LivePortrait's warp/decode pipeline, including
paste-back into the source frame — which is exactly the paper's section 4.5
pathway with the hand-built audio encoder replaced by a pretrained motion
generator (see docs/phase2_plan_ryan.md, Model Decisions).

Its dependency set (insightface, onnxruntime-gpu, Python 3.10 era) conflicts
with this repo's Python 3.12 environment, so we invoke its ``inference.py``
in its own environment via subprocess. Model weights reload on every call;
acceptable for the offline Milestone 2 path. The warm in-process service for
the real-time loop is a Phase 3/4 concern (``ANIMATION_SERVER_URL`` is the
reserved seam).

Configuration (``.env``):
- ``RPA_JOYVASA_DIR``: path to the JoyVASA clone
- ``RPA_JOYVASA_PYTHON``: python executable of the JoyVASA environment
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from reactive_portrait_animation.animation.provider import AnimatedVideo, AnimationJob
from reactive_portrait_animation.config import AppSettings

SETUP_HINT = (
    "JoyVASA backend is not configured. Clone + set up JoyVASA in its own "
    "python 3.10 env (see docs/setup_models.md section 3), then set "
    "RPA_JOYVASA_DIR and RPA_JOYVASA_PYTHON in .env."
)

INFERENCE_TIMEOUT_S = 900  # cold model load + generation can take minutes


def build_inference_command(
    python_exe: str, joyvasa_dir: Path, job: AnimationJob, animation_mode: str = "human"
) -> list[str]:
    """JoyVASA CLI invocation (kept separate for testability)."""
    return [
        python_exe,
        str(joyvasa_dir / "inference.py"),
        "-r",
        str(job.portrait.resolve()),
        "-a",
        str(job.audio.resolve()),
        "--animation_mode",
        animation_mode,
        "--cfg_scale",
        "2.0",
    ]


def find_output_video(joyvasa_dir: Path, since: float) -> Path | None:
    """Locate the video JoyVASA produced during this run.

    JoyVASA writes results into its own tree; rather than depending on its
    exact output layout, take the newest mp4 modified after the run started.
    """
    candidates = [
        p for p in joyvasa_dir.rglob("*.mp4") if p.stat().st_mtime >= since - 1.0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


class JoyVasaSubprocessProvider:
    def __init__(self, settings: AppSettings) -> None:
        self.joyvasa_dir = Path(settings.joyvasa_dir)
        self.python_exe = settings.joyvasa_python

    def animate(self, job: AnimationJob) -> AnimatedVideo:
        if not self.python_exe or not (self.joyvasa_dir / "inference.py").exists():
            raise RuntimeError(SETUP_HINT)

        command = build_inference_command(self.python_exe, self.joyvasa_dir, job)
        # With captured (non-console) output on Windows, Python defaults to
        # cp1252 and crashes on JoyVASA's emoji progress bars -- force UTF-8.
        child_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        started = time.time()
        try:
            completed = subprocess.run(
                command,
                cwd=self.joyvasa_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=child_env,
                timeout=INFERENCE_TIMEOUT_S,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"JoyVASA python executable not found: {self.python_exe!r}. {SETUP_HINT}"
            ) from exc
        if completed.returncode != 0:
            tail = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-25:])
            raise RuntimeError(
                f"JoyVASA inference failed (exit {completed.returncode}). Last output:\n{tail}"
            )

        produced = find_output_video(self.joyvasa_dir, since=started)
        if produced is None:
            raise RuntimeError(
                "JoyVASA reported success but no output mp4 was found under "
                f"{self.joyvasa_dir} -- check its output directory layout."
            )
        job.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(produced, job.output)
        elapsed_ms = (time.time() - started) * 1000.0
        return AnimatedVideo(video_path=job.output, elapsed_ms=elapsed_ms, backend="joyvasa")
