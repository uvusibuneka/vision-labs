"""TDD tests for LW1 variant 7: binary erosion 3x3, zero-padding border.

Border contract: outside image == 0. Output pixel is 255 iff all 9 pixels
of its 3x3 neighbourhood (with zero padding) are 255.

All expected arrays below are HARDCODED by hand, cv2.erode is never used
to compute them (per task constraint).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from lab1.erosion import binarize, erode_library, erode_manual

BORDER_CONTRACT = "zero-padding: outside image == 0 (background)"


def assert_binary(img: np.ndarray, shape: tuple[int, ...] | None = None) -> None:
    assert isinstance(img, np.ndarray), f"expected ndarray, got {type(img)}"
    assert img.dtype == np.uint8, f"expected uint8, got {img.dtype}"
    assert img.ndim == 2, f"expected 2D image, got ndim={img.ndim}"
    if shape is not None:
        assert img.shape == shape, f"expected shape {shape}, got {img.shape}"
    unique = set(np.unique(img).tolist())
    assert unique <= {0, 255}, f"binary image must contain only 0/255, got {unique}"


# ---------------------------------------------------------------- binarize
class TestBinarize:
    def test_threshold_gt_semantics(self):
        """OpenCV THRESH_BINARY semantics: dst=255 iff src > thresh."""
        gray = np.array([[0, 127, 128, 255]], dtype=np.uint8)
        expected = np.array([[0, 0, 255, 255]], dtype=np.uint8)
        out = binarize(gray, thresh=127)
        assert_binary(out, shape=gray.shape)
        np.testing.assert_array_equal(out, expected)

    def test_threshold_zero(self):
        gray = np.array([[0, 1, 254]], dtype=np.uint8)
        expected = np.array([[0, 255, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(binarize(gray, thresh=0), expected)

    def test_threshold_255_gives_all_black(self):
        gray = np.array([[0, 127, 255]], dtype=np.uint8)
        expected = np.zeros_like(gray)
        np.testing.assert_array_equal(binarize(gray, thresh=255), expected)

    def test_preserves_shape_and_dtype(self):
        rng = np.random.default_rng(7)
        gray = rng.integers(0, 256, size=(9, 11)).astype(np.uint8)
        out = binarize(gray)
        assert_binary(out, shape=(9, 11))


# ------------------------------------------------------- erosion: fixtures
def img_all_white_5x5():
    img = np.full((5, 5), 255, dtype=np.uint8)
    # Zero padding kills the outer frame; inner 3x3 survives.
    expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 255, 255, 255, 0],
            [0, 255, 255, 255, 0],
            [0, 255, 255, 255, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    return img, expected


def img_all_black_5x5():
    img = np.zeros((5, 5), dtype=np.uint8)
    return img, np.zeros((5, 5), dtype=np.uint8)


def img_single_center_pixel_5x5():
    img = np.zeros((5, 5), dtype=np.uint8)
    img[2, 2] = 255
    return img, np.zeros((5, 5), dtype=np.uint8)


def img_3x3_all_white():
    img = np.full((3, 3), 255, dtype=np.uint8)
    # Only the center's neighbourhood lies fully inside.
    expected = np.array(
        [[0, 0, 0], [0, 255, 0], [0, 0, 0]],
        dtype=np.uint8,
    )
    return img, expected


def img_center_3x3_in_7x7():
    """3x3 white square on black 7x7 -> single surviving pixel."""
    img = np.zeros((7, 7), dtype=np.uint8)
    img[2:5, 2:5] = 255
    expected = np.zeros((7, 7), dtype=np.uint8)
    expected[3, 3] = 255
    return img, expected


def img_center_5x5_in_7x7():
    """5x5 white square on black 7x7 -> 3x3 survivors (peeled by 1 px)."""
    img = np.zeros((7, 7), dtype=np.uint8)
    img[1:6, 1:6] = 255
    expected = np.zeros((7, 7), dtype=np.uint8)
    expected[2:5, 2:5] = 255
    return img, expected


def img_top_strip_touching_border_5x5():
    """White strip rows 0-1 touching top border -> fully erased (border=0)."""
    img = np.zeros((5, 5), dtype=np.uint8)
    img[0:2, :] = 255
    return img, np.zeros((5, 5), dtype=np.uint8)


def img_left_column_touching_border_5x5():
    img = np.zeros((5, 5), dtype=np.uint8)
    img[:, 0:2] = 255
    return img, np.zeros((5, 5), dtype=np.uint8)


MANUAL_CASES = {
    "all_white_5x5": img_all_white_5x5(),
    "all_black_5x5": img_all_black_5x5(),
    "single_pixel_5x5": img_single_center_pixel_5x5(),
    "all_white_3x3": img_3x3_all_white(),
    "square3_in_7x7": img_center_3x3_in_7x7(),
    "square5_in_7x7": img_center_5x5_in_7x7(),
    "top_strip_border": img_top_strip_touching_border_5x5(),
    "left_strip_border": img_left_column_touching_border_5x5(),
}


# ------------------------------------------------------------- manual impl
class TestErodeManual:
    @pytest.mark.parametrize("name", list(MANUAL_CASES))
    def test_hardcoded_cases(self, name):
        img, expected = MANUAL_CASES[name]
        out = erode_manual(img)
        assert_binary(out, shape=img.shape)
        np.testing.assert_array_equal(out, expected, err_msg=f"case={name}")

    def test_1x1_white_erodes_to_black(self):
        """1x1: neighbourhood includes padding zeros -> 0."""
        out = erode_manual(np.full((1, 1), 255, dtype=np.uint8))
        np.testing.assert_array_equal(out, np.zeros((1, 1), dtype=np.uint8))

    def test_1x1_black_stays_black(self):
        out = erode_manual(np.zeros((1, 1), dtype=np.uint8))
        np.testing.assert_array_equal(out, np.zeros((1, 1), dtype=np.uint8))

    def test_result_size_type_and_values(self):
        rng = np.random.default_rng(42)
        img = (rng.integers(0, 2, size=(10, 12)) * 255).astype(np.uint8)
        out = erode_manual(img)
        assert_binary(out, shape=(10, 12))


# ------------------------------------------------------------ library impl
class TestErodeLibrary:
    @pytest.mark.parametrize("name", list(MANUAL_CASES))
    def test_hardcoded_cases(self, name):
        """Same hardcoded expectations: library must use zero-padding border."""
        img, expected = MANUAL_CASES[name]
        out = erode_library(img)
        assert_binary(out, shape=img.shape)
        np.testing.assert_array_equal(out, expected, err_msg=f"case={name}")

    def test_1x1_white_erodes_to_black(self):
        out = erode_library(np.full((1, 1), 255, dtype=np.uint8))
        np.testing.assert_array_equal(out, np.zeros((1, 1), dtype=np.uint8))

    def test_result_size_type_and_values(self):
        rng = np.random.default_rng(123)
        img = (rng.integers(0, 2, size=(8, 9)) * 255).astype(np.uint8)
        out = erode_library(img)
        assert_binary(out, shape=(8, 9))


# ------------------------------------------------------------- equivalence
class TestEquivalence:
    @pytest.mark.parametrize("name", list(MANUAL_CASES))
    def test_library_equals_manual_on_patterns(self, name):
        img, _ = MANUAL_CASES[name]
        np.testing.assert_array_equal(erode_library(img), erode_manual(img))

    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize("shape", [(8, 8), (10, 15), (16, 16)])
    def test_library_equals_manual_on_random(self, seed, shape):
        rng = np.random.default_rng(seed)
        img = (rng.integers(0, 2, size=shape) * 255).astype(np.uint8)
        np.testing.assert_array_equal(erode_library(img), erode_manual(img))

    def test_binarize_then_erode_pipeline_agrees(self):
        rng = np.random.default_rng(99)
        gray = rng.integers(0, 256, size=(12, 12)).astype(np.uint8)
        binary = binarize(gray, thresh=127)
        assert_binary(binary, shape=(12, 12))
        np.testing.assert_array_equal(erode_library(binary), erode_manual(binary))


# ------------------------------------------------------------------ PNG/JPG
class TestImageIO:
    def test_png_roundtrip_exact(self, tmp_path):
        """Synthetic binary -> PNG -> imread must be bit-exact."""
        img, _ = img_center_5x5_in_7x7()
        p = tmp_path / "binary.png"
        assert cv2.imwrite(str(p), img)
        loaded = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        assert loaded is not None
        assert loaded.dtype == np.uint8
        np.testing.assert_array_equal(loaded, img)

    def test_jpeg_loadable_and_binarizable(self, tmp_path):
        """JPEG is lossy: check shape/dtype survive, binarize output is binary."""
        img, _ = img_center_5x5_in_7x7()
        # Upscale so JPEG artifacts don't dominate the check.
        big = cv2.resize(img, (70, 70), interpolation=cv2.INTER_NEAREST)
        p = tmp_path / "binary.jpg"
        assert cv2.imwrite(str(p), big, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        loaded = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        assert loaded is not None
        assert loaded.shape == big.shape
        assert loaded.dtype == np.uint8
        out = binarize(loaded, thresh=127)
        assert_binary(out, shape=big.shape)
