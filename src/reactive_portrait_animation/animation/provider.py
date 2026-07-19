"""Animation provider interface.

Milestone 2 contract: given a portrait image and a speech audio file, produce
a talking-head video. The heavy models (JoyVASA driving LivePortrait's
renderer) live in a separate Python 3.10 environment, so the default real
backend shells out to it; a mock backend keeps tests and the demo pipeline
runnable anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from reactive_portrait_animation.config import AppSettings


@dataclass(slots=True)
class AnimationJob:
    """Inputs for one offline animation run."""

    portrait: Path
    audio: Path
    output: Path


@dataclass(slots=True)
class AnimatedVideo:
    """Result of one offline animation run."""

    video_path: Path
    elapsed_ms: float
    backend: str


class AnimationProvider(Protocol):
    """Anything that can turn (portrait, audio) into a talking-head video."""

    def animate(self, job: AnimationJob) -> AnimatedVideo: ...


def build_animation_provider(settings: AppSettings) -> AnimationProvider:
    """Select the animation backend from ``RPA_ANIMATION_PROVIDER``.

    - ``local``: JoyVASA subprocess in its own env (the real path)
    - ``mock``: instant placeholder output (tests, scaffold demo)
    """
    if settings.animation_provider == "local":
        from reactive_portrait_animation.animation.joyvasa import JoyVasaSubprocessProvider

        return JoyVasaSubprocessProvider(settings)
    if settings.animation_provider == "mock":
        from reactive_portrait_animation.animation.mock import MockAnimationProvider

        return MockAnimationProvider()
    raise ValueError(
        f"Unsupported animation provider: {settings.animation_provider!r} "
        "(expected 'local' or 'mock')"
    )
