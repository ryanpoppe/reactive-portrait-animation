import os
import time
from pathlib import Path

import pytest

from reactive_portrait_animation.animation.joyvasa import (
    JoyVasaSubprocessProvider,
    build_inference_command,
    find_output_video,
)
from reactive_portrait_animation.animation.mock import MockAnimationProvider
from reactive_portrait_animation.animation.provider import (
    AnimationJob,
    build_animation_provider,
)
from reactive_portrait_animation.config import AppSettings


def make_job(tmp_path: Path) -> AnimationJob:
    portrait = tmp_path / "portrait.png"
    audio = tmp_path / "speech.wav"
    portrait.write_bytes(b"fake image")
    audio.write_bytes(b"fake audio")
    return AnimationJob(portrait=portrait, audio=audio, output=tmp_path / "out" / "video.mp4")


def test_factory_selects_mock() -> None:
    settings = AppSettings(RPA_ANIMATION_PROVIDER="mock")  # type: ignore[call-arg]

    provider = build_animation_provider(settings)

    assert isinstance(provider, MockAnimationProvider)


def test_factory_selects_joyvasa_for_local() -> None:
    settings = AppSettings(RPA_ANIMATION_PROVIDER="local")  # type: ignore[call-arg]

    provider = build_animation_provider(settings)

    assert isinstance(provider, JoyVasaSubprocessProvider)


def test_factory_rejects_unknown_provider() -> None:
    settings = AppSettings(RPA_ANIMATION_PROVIDER="openai")  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="Unsupported animation provider"):
        build_animation_provider(settings)


def test_mock_provider_writes_output(tmp_path: Path) -> None:
    job = make_job(tmp_path)

    result = MockAnimationProvider().animate(job)

    assert result.backend == "mock"
    assert result.video_path.exists()
    assert result.elapsed_ms >= 0


def test_joyvasa_raises_setup_hint_when_unconfigured(tmp_path: Path) -> None:
    settings = AppSettings(  # type: ignore[call-arg]
        RPA_ANIMATION_PROVIDER="local",
        RPA_JOYVASA_DIR=str(tmp_path / "missing"),
        RPA_JOYVASA_PYTHON="",
    )
    provider = JoyVasaSubprocessProvider(settings)

    with pytest.raises(RuntimeError, match="not configured"):
        provider.animate(make_job(tmp_path))


def test_build_inference_command_shape(tmp_path: Path) -> None:
    job = make_job(tmp_path)

    command = build_inference_command("python", tmp_path, job)

    assert command[0] == "python"
    assert command[1].endswith("inference.py")
    assert str(job.portrait.resolve()) in command
    assert str(job.audio.resolve()) in command
    assert "--animation_mode" in command


def test_find_output_video_picks_newest_recent(tmp_path: Path) -> None:
    old = tmp_path / "animations" / "old.mp4"
    new = tmp_path / "animations" / "new.mp4"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    past = time.time() - 3600
    os.utime(old, (past, past))

    found = find_output_video(tmp_path, since=time.time() - 60)

    assert found == new
    assert find_output_video(tmp_path, since=time.time() + 60) is None
