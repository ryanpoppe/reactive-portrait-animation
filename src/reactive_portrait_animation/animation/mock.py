"""Mock animation backend: instant placeholder output for tests and demos."""

from __future__ import annotations

import time

from reactive_portrait_animation.animation.provider import AnimatedVideo, AnimationJob


class MockAnimationProvider:
    def animate(self, job: AnimationJob) -> AnimatedVideo:
        started = time.time()
        job.output.parent.mkdir(parents=True, exist_ok=True)
        job.output.write_bytes(b"")  # placeholder artifact, not a playable video
        elapsed_ms = (time.time() - started) * 1000.0
        return AnimatedVideo(video_path=job.output, elapsed_ms=elapsed_ms, backend="mock")
