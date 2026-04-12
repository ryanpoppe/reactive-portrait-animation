from reactive_portrait_animation.config import AppSettings
from reactive_portrait_animation.domain.models import (
    DemoResult,
    Response,
    SceneObservation,
    SpeechChunk,
    Transcript,
)


class MockPipeline:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def run_demo(self) -> DemoResult:
        observation = SceneObservation(
            summary="A viewer steps into frame and looks at the portrait."
        )
        transcript = Transcript(text="Hello there, who are you?")
        response = Response(
            text=(
                "I am a reactive portrait prototype, currently running with mock providers "
                f"on {self.settings.device}/{self.settings.accelerator}."
            )
        )
        speech_chunks = [
            SpeechChunk(text="I am a reactive portrait prototype,", order=1),
            SpeechChunk(text="currently running with mock providers.", order=2),
        ]
        return DemoResult(
            observation=observation,
            transcript=transcript,
            response=response,
            speech_chunks=speech_chunks,
            animation_frames=12,
        )
