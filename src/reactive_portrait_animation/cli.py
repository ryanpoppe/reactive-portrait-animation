from pathlib import Path

import typer

from reactive_portrait_animation.config import get_settings
from reactive_portrait_animation.pipeline.app import MockPipeline

app = typer.Typer(help="Reactive Portrait Animation tooling.")
PORTRAIT_PATH_ARGUMENT = typer.Argument(
    ..., exists=False, help="Path to the source portrait image."
)


@app.command()
def info() -> None:
    """Show the active configuration."""
    settings = get_settings()
    typer.echo(f"Environment: {settings.environment}")
    typer.echo(f"Device: {settings.device}")
    typer.echo(f"Accelerator: {settings.accelerator}")
    typer.echo(f"LLM provider: {settings.llm_provider}")
    typer.echo(f"TTS provider: {settings.tts_provider}")
    typer.echo(f"Persona: {settings.persona_path}")


@app.command()
def demo() -> None:
    """Run the mock interactive pipeline."""
    settings = get_settings()
    pipeline = MockPipeline(settings=settings)
    result = pipeline.run_demo()
    typer.echo(f"Observation: {result.observation.summary}")
    typer.echo(f"Transcript: {result.transcript.text}")
    typer.echo(f"Response: {result.response.text}")
    typer.echo(f"Speech chunks: {len(result.speech_chunks)}")
    typer.echo(f"Animation frames: {result.animation.frame_count}")


@app.command()
def preprocess(
    portrait: Path = PORTRAIT_PATH_ARGUMENT,
    feather: int | None = typer.Option(
        None, help="Mask feather radius in pixels (default from settings)."
    ),
) -> None:
    """Offline preprocessing: segment the portrait into mask + background plate."""
    from reactive_portrait_animation.preprocessing.segmentation import segment_portrait

    if not portrait.exists():
        typer.echo(f"Portrait not found: {portrait}")
        raise typer.Exit(code=1)
    settings = get_settings()
    try:
        artifacts = segment_portrait(portrait, settings, feather_radius=feather)
    except (RuntimeError, ImportError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(f"Portrait hash: {artifacts.portrait_hash}")
    typer.echo(f"Face box: {artifacts.face_box}")
    typer.echo(f"Subject box: {artifacts.subject_box}")
    typer.echo(f"Mask: {artifacts.mask_path}")
    typer.echo(f"Background plate: {artifacts.plate_path}")
    typer.echo(f"Metadata: {artifacts.metadata_path}")


def main() -> None:
    app()
