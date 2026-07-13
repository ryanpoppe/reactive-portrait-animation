"""SAM-based portrait segmentation for offline preprocessing.

Produces the foreground mask ``M`` and background plate ``B`` described in
paper section 4.1 (step 1). Artifacts are written to
``<cache_dir>/<portrait_hash>/`` in the format documented in
``docs/interfaces.md``.

Heavy dependencies (torch, sam2, cv2, numpy) are imported lazily so the rest
of the package — and the test suite — works without them installed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from reactive_portrait_animation.config import AppSettings

Box = tuple[int, int, int, int]
"""Pixel box as (x0, y0, x1, y1)."""

Point = tuple[int, int]
"""Pixel point as (x, y)."""

MaskPredictor = Callable[["NDArray[Any]", Box, "list[Point]", "list[int]"], "NDArray[Any]"]
"""Callable mapping (BGR image, prompt box, prompt points, labels) -> float mask, (H, W).

Labels: 1 = foreground point, 0 = background point.
"""

MISSING_DEPS_HINT = (
    "Segmentation dependencies are not installed. On the GPU machine run:\n"
    "  uv sync --group preprocess\n"
    "(torch/sam2 sources are pinned in pyproject.toml -- do not uv pip install them "
    "manually), then download a SAM 2.1 checkpoint (see docs/setup_models.md)."
)


@dataclass(slots=True)
class SegmentationArtifacts:
    """Paths and metadata for one segmented portrait."""

    portrait_hash: str
    cache_dir: Path
    mask_path: Path
    plate_path: Path
    metadata_path: Path
    face_box: Box | None
    subject_box: Box
    width: int
    height: int
    feather_radius: int


def portrait_hash(image_path: Path) -> str:
    """Stable content hash used to key the per-portrait cache directory."""
    return hashlib.sha256(image_path.read_bytes()).hexdigest()[:16]


def portrait_cache_dir(settings: AppSettings, image_path: Path) -> Path:
    directory = settings.cache_dir / portrait_hash(image_path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def expand_box(
    box: Box,
    width: int,
    height: int,
    up: float = 1.0,
    side: float = 1.5,
    anchor_bottom: bool = True,
) -> Box:
    """Expand a face box to cover hair, shoulders, and torso.

    Expansion factors are fractions of the face box's own size, clipped to
    image bounds. With ``anchor_bottom`` (default) the box always extends to
    the bottom edge of the frame: portrait subjects are effectively always
    bottom-anchored, and face-relative expansion undershoots badly for
    small faces / full-body compositions.
    """
    x0, y0, x1, y1 = box
    box_w = x1 - x0
    box_h = y1 - y0
    return (
        max(0, int(x0 - side * box_w)),
        max(0, int(y0 - up * box_h)),
        min(width, int(x1 + side * box_w)),
        height if anchor_bottom else min(height, int(y1 + 3.0 * box_h)),
    )


def default_subject_box(width: int, height: int) -> Box:
    """Fallback prompt covering the central region of a typical portrait."""
    return (int(width * 0.12), int(height * 0.04), int(width * 0.88), height)


def subject_prompts(
    face_box: Box | None, width: int, height: int
) -> tuple[list[Point], list[int]]:
    """Build SAM point prompts for a portrait composition.

    Positives: face center plus a torso point below it (or center-frame
    fallbacks when no face was detected). Negatives: the top corners, which
    are background in essentially every portrait composition.
    """
    if face_box is not None:
        fx0, fy0, fx1, fy1 = face_box
        face_w = fx1 - fx0
        face_h = fy1 - fy0
        face_center: Point = ((fx0 + fx1) // 2, (fy0 + fy1) // 2)
        torso: Point = (face_center[0], min(height - 1, fy1 + int(1.5 * face_h)))
        shoulder_y = min(height - 1, fy1 + face_h)
        left_shoulder: Point = (max(0, face_center[0] - face_w), shoulder_y)
        right_shoulder: Point = (min(width - 1, face_center[0] + face_w), shoulder_y)
    else:
        face_center = (width // 2, height // 4)
        torso = (width // 2, height // 2)
        left_shoulder = (width // 3, int(height * 0.4))
        right_shoulder = (2 * width // 3, int(height * 0.4))
    positives = [face_center, torso, left_shoulder, right_shoulder]
    negatives: list[Point] = [(2, 2), (width - 3, 2)]
    points = [*positives, *negatives]
    labels = [1] * len(positives) + [0] * len(negatives)
    return points, labels


def pick_mask(
    masks: NDArray[Any],
    scores: NDArray[Any],
    points: list[Point],
    labels: list[int],
) -> NDArray[Any]:
    """Select the SAM candidate that best agrees with the prompt points.

    SAM's own score ranking can prefer degenerate masks (background, or the
    subject minus clothing), and a single forced output can ignore prompts
    outright. Rank candidates by (positives contained - negatives contained),
    tie-broken by SAM score.
    """
    best_i = 0
    best_key: tuple[int, float] | None = None
    for i in range(len(masks)):
        mask = masks[i]
        pos = sum(1 for (x, y), lab in zip(points, labels, strict=True) if lab and mask[y, x] > 0.5)
        neg = sum(
            1 for (x, y), lab in zip(points, labels, strict=True) if not lab and mask[y, x] > 0.5
        )
        key = (pos - neg, float(scores[i]))
        if best_key is None or key > best_key:
            best_i, best_key = i, key
    result: NDArray[Any] = masks[best_i]
    return result


def detect_face_box(image: NDArray[Any]) -> Box | None:
    """Detect the largest face with OpenCV's bundled Haar cascade.

    Deliberately dependency-light: this only needs to produce a rough box
    prompt for SAM, not a precise landmark fit.
    """
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
        )
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    except AttributeError:
        # cv2 build without objdetect (e.g. opencv-python 5.x): fall back to
        # the default subject box rather than failing preprocessing outright.
        print(
            "warning: this cv2 build lacks CascadeClassifier; using default subject box. "
            "Install opencv-python 4.x for face-box prompting."
        )
        return None
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    return (int(x), int(y), int(x + w), int(y + h))


def clean_mask(mask: NDArray[Any], kernel_size: int = 7) -> NDArray[Any]:
    """Denoise a uint8 mask: despeckle, keep largest blob, fill holes.

    SAM occasionally returns dithered speckle or disconnected background
    islands; a talking-portrait subject is by definition a single connected
    region without interior holes.
    """
    import cv2
    import numpy as np

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if count > 2:  # background + more than one blob
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        cleaned = np.where(labels == largest, 255, 0).astype(np.uint8)

    # fill interior holes: flood-fill background from the top-left corner,
    # anything unreached and unmasked is a hole
    flooded = cleaned.copy()
    flood_mask = np.zeros((cleaned.shape[0] + 2, cleaned.shape[1] + 2), np.uint8)
    cv2.floodFill(flooded, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flooded)
    filled: NDArray[Any] = cv2.bitwise_or(cleaned, holes)
    return filled


def feather_mask(mask: NDArray[Any], radius: int = 10) -> NDArray[Any]:
    """Soft-edge a uint8 mask with a Gaussian blur (paper section 4.5 step 5)."""
    import cv2

    if radius <= 0:
        return mask
    kernel = radius * 2 + 1
    blurred: NDArray[Any] = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    return blurred


def build_sam_predictor(settings: AppSettings) -> MaskPredictor:
    """Build a SAM 2.1 image predictor. Raises RuntimeError if deps missing."""
    try:
        import numpy as np
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:  # pragma: no cover - exercised only without deps
        raise RuntimeError(MISSING_DEPS_HINT) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(settings.sam_model_cfg, str(settings.sam_checkpoint), device=device)
    predictor = SAM2ImagePredictor(model)

    def predict(
        image: NDArray[Any], box: Box, points: list[Point], labels: list[int]
    ) -> NDArray[Any]:
        # BGR -> RGB; ascontiguousarray avoids negative strides torch rejects
        predictor.set_image(np.ascontiguousarray(image[..., ::-1]))
        masks, scores, _ = predictor.predict(
            box=np.asarray(box),
            point_coords=np.asarray(points),
            point_labels=np.asarray(labels),
            multimask_output=True,
        )
        return pick_mask(masks, scores, points, labels)

    return predict


def segment_portrait(
    image_path: Path,
    settings: AppSettings,
    predictor: MaskPredictor | None = None,
    feather_radius: int | None = None,
) -> SegmentationArtifacts:
    """Segment a portrait into foreground mask + background plate.

    Writes ``mask.png``, ``background_plate.png``, and ``metadata.json`` to the
    per-portrait cache directory and returns their paths. A custom
    ``predictor`` may be injected (tests, alternative models); by default a
    SAM 2.1 predictor is built from settings.
    """
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    radius = settings.feather_radius if feather_radius is None else feather_radius

    face_box = detect_face_box(image)
    subject_box = (
        expand_box(face_box, width, height)
        if face_box is not None
        else default_subject_box(width, height)
    )
    points, labels = subject_prompts(face_box, width, height)

    if predictor is None:
        predictor = build_sam_predictor(settings)
    mask = predictor(image, subject_box, points, labels)

    mask_u8 = ((np.asarray(mask, dtype=np.float32) > 0.5) * 255).astype(np.uint8)
    mask_u8 = clean_mask(mask_u8)
    mask_u8 = feather_mask(mask_u8, radius)

    missed = [
        (x, y)
        for (x, y), lab in zip(points, labels, strict=True)
        if lab and int(mask_u8[y, x]) <= 127
    ]
    if missed:
        print(
            f"warning: mask misses {len(missed)} positive prompt point(s) {missed} -- "
            "inspect mask.png before handing off"
        )

    plate = image.copy()
    plate[mask_u8 > 127] = 0

    cache = portrait_cache_dir(settings, image_path)
    artifacts = SegmentationArtifacts(
        portrait_hash=cache.name,
        cache_dir=cache,
        mask_path=cache / "mask.png",
        plate_path=cache / "background_plate.png",
        metadata_path=cache / "metadata.json",
        face_box=face_box,
        subject_box=subject_box,
        width=width,
        height=height,
        feather_radius=radius,
    )
    cv2.imwrite(str(artifacts.mask_path), mask_u8)
    cv2.imwrite(str(artifacts.plate_path), plate)

    metadata = asdict(artifacts)
    metadata["source_image"] = str(image_path)
    metadata["prompt_points"] = points
    metadata["prompt_labels"] = labels
    for key in ("cache_dir", "mask_path", "plate_path", "metadata_path"):
        metadata[key] = str(metadata[key])
    artifacts.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return artifacts
