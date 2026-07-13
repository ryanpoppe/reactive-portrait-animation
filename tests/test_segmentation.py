import json
from pathlib import Path
from typing import Any

import pytest

from reactive_portrait_animation.config import AppSettings
from reactive_portrait_animation.preprocessing.segmentation import (
    default_subject_box,
    expand_box,
    portrait_cache_dir,
    portrait_hash,
)

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from reactive_portrait_animation.preprocessing.segmentation import (  # noqa: E402
    clean_mask,
    feather_mask,
    segment_portrait,
    subject_prompts,
)


def make_settings(tmp_path: Path) -> AppSettings:
    return AppSettings(RPA_CACHE_DIR=str(tmp_path / "cache"))  # type: ignore[call-arg]


def write_test_image(path: Path, width: int = 64, height: int = 80) -> None:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[10:60, 20:44] = (10, 120, 200)  # a "subject" blob
    cv2.imwrite(str(path), image)


def test_expand_box_clips_to_image_bounds() -> None:
    box = expand_box((30, 20, 70, 60), width=100, height=100)

    x0, y0, x1, y1 = box
    assert 0 <= x0 < x1 <= 100
    assert 0 <= y0 < y1 <= 100
    assert y1 == 100  # torso expansion reaches frame bottom


def test_default_subject_box_within_bounds() -> None:
    x0, y0, x1, y1 = default_subject_box(320, 240)

    assert 0 <= x0 < x1 <= 320
    assert 0 <= y0 < y1 <= 240


def test_portrait_hash_stable_and_content_keyed(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    write_test_image(a)
    write_test_image(b, width=32)

    assert portrait_hash(a) == portrait_hash(a)
    assert portrait_hash(a) != portrait_hash(b)
    assert len(portrait_hash(a)) == 16


def test_portrait_cache_dir_created(tmp_path: Path) -> None:
    image = tmp_path / "p.png"
    write_test_image(image)
    settings = make_settings(tmp_path)

    cache = portrait_cache_dir(settings, image)

    assert cache.is_dir()
    assert cache.parent == settings.cache_dir


def test_subject_prompts_positive_and_negative_points() -> None:
    points, labels = subject_prompts((40, 20, 60, 40), width=100, height=200)

    assert len(points) == len(labels) == 6
    assert labels == [1, 1, 1, 1, 0, 0]
    assert points[0] == (50, 30)  # face center
    assert points[1][1] > 40  # torso point below the face box
    assert points[2][0] < 50 < points[3][0]  # shoulders straddle face center
    for x, y in points:
        assert 0 <= x < 100 and 0 <= y < 200


def test_pick_mask_prefers_prompt_agreement_over_score() -> None:
    from reactive_portrait_animation.preprocessing.segmentation import pick_mask

    h = w = 50
    good = np.zeros((h, w), dtype=np.float32)
    good[10:50, 10:40] = 1.0  # covers subject points, avoids corners
    inverted = 1.0 - good  # covers corners, misses subject
    masks = np.stack([inverted, good])
    scores = np.asarray([0.99, 0.10])  # SAM "prefers" the inverted mask

    points = [(25, 20), (25, 40), (2, 2), (w - 3, 2)]
    labels = [1, 1, 0, 0]

    picked = pick_mask(masks, scores, points, labels)

    assert (picked == good).all()


def test_clean_mask_despeckles_and_fills_holes() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255  # main subject blob
    mask[40:50, 40:50] = 0  # interior hole
    mask[5, 5] = 255  # speckle noise
    mask[90:92, 90:92] = 255  # disconnected island

    cleaned = clean_mask(mask)

    assert cleaned[45, 45] == 255  # hole filled
    assert cleaned[5, 5] == 0  # speckle removed
    assert cleaned[91, 91] == 0  # island removed
    assert cleaned[50, 22] == 255  # main blob intact


def test_feather_mask_softens_edges() -> None:
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 255

    feathered = feather_mask(mask, radius=4)

    intermediate = (feathered > 0) & (feathered < 255)
    assert intermediate.any()
    assert feather_mask(mask, radius=0) is mask


def test_segment_portrait_writes_artifacts(tmp_path: Path) -> None:
    image_path = tmp_path / "portrait.png"
    write_test_image(image_path)
    settings = make_settings(tmp_path)

    def stub_predictor(
        image: Any,
        box: tuple[int, int, int, int],
        points: list[tuple[int, int]],
        labels: list[int],
    ) -> Any:
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        x0, y0, x1, y1 = box
        mask[y0:y1, x0:x1] = 1.0
        return mask

    artifacts = segment_portrait(image_path, settings, predictor=stub_predictor)

    assert artifacts.mask_path.exists()
    assert artifacts.plate_path.exists()
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["portrait_hash"] == artifacts.portrait_hash
    assert metadata["source_image"] == str(image_path)

    mask = cv2.imread(str(artifacts.mask_path), cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (80, 64)
    assert mask.max() == 255
    plate = cv2.imread(str(artifacts.plate_path))
    # foreground removed from plate where mask is solid
    assert plate[mask > 250].max() == 0
