from dataclasses import dataclass


@dataclass(slots=True)
class SceneObservation:
    summary: str


@dataclass(slots=True)
class Transcript:
    text: str


@dataclass(slots=True)
class Response:
    text: str


@dataclass(slots=True)
class SpeechChunk:
    text: str
    order: int


@dataclass(slots=True)
class AnimationResult:
    frame_count: int


@dataclass(slots=True)
class DemoResult:
    observation: SceneObservation
    transcript: Transcript
    response: Response
    speech_chunks: list[SpeechChunk]
    animation: AnimationResult
