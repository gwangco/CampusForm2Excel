"""T-04 크기·노이즈·기울기 보정과 실패 안내."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from support import simulate_photo

from campusform2excel.alignment import align_page
from campusform2excel.preprocess import MAX_LONG_SIDE, MIN_SHORT_SIDE, PreprocessError, normalize


def test_t04_original_array_is_not_modified(blank_pages):
    photo = simulate_photo(blank_pages["attendance"], seed=1).image
    before = photo.copy()
    normalize(photo)
    assert np.array_equal(photo, before)


def test_t04_oversized_photo_scaled_down(blank_pages):
    photo = cv2.resize(blank_pages["attendance"], (3000, 4242))
    result = normalize(photo)
    assert max(result.image.shape[:2]) == MAX_LONG_SIDE
    assert result.scale == pytest.approx(MAX_LONG_SIDE / 4242)


def test_t04_noise_is_reduced(blank_pages):
    rng = np.random.default_rng(0)
    clean = np.full((1200, 1000, 3), 200, np.uint8)
    noisy = np.clip(clean + rng.normal(0, 8, clean.shape), 0, 255).astype(np.uint8)
    assert normalize(noisy).image.std() < noisy.std()


def test_t04_low_resolution_rejected_with_guidance():
    with pytest.raises(PreprocessError, match="다시 촬영"):
        normalize(np.full((MIN_SHORT_SIDE - 1, 2000, 3), 255, np.uint8))


def test_t04_blurry_photo_warned(blank_pages):
    blurred = cv2.GaussianBlur(blank_pages["attendance"], (0, 0), 12)
    assert any("흐릿" in warning for warning in normalize(blurred).warnings)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_t04_skew_is_corrected(blank_pages, templates, seed):
    template = templates[0]
    photo = simulate_photo(blank_pages[template.template_id], seed=seed, max_angle=6.0).image
    normalized = normalize(photo)
    aligned = align_page(normalized.image, template, 1, normalized.scale)
    assert aligned.image.shape[:2] == template.page_size[::-1]
    assert aligned.grid_score >= 0.9
