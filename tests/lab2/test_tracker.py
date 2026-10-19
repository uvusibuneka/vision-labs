"""Tests for LW2: keypoint-based planar object tracking.

The template is a synthetic textured image, so the true homography is known
exactly and the projected box can be compared with the expected corners.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from lab2.cli import main
from lab2.tracker import KeypointTracker, TrackerConfig, draw_tracking

W, H = 320, 240


# ------------------------------------------------------------------ helpers
def make_template(seed: int = 0) -> np.ndarray:
    """Textured BGR image: smooth noise plus a few hard-edged shapes."""
    rng = np.random.default_rng(seed)
    coarse = rng.normal(128, 55, (30, 40)).astype(np.float32)
    image = cv2.resize(coarse, (W, H), interpolation=cv2.INTER_CUBIC)
    image = cv2.GaussianBlur(image, (0, 0), 1.0)
    image = np.clip(image, 0, 255).astype(np.uint8)
    for _ in range(10):
        x = int(rng.integers(0, W - 60))
        y = int(rng.integers(0, H - 50))
        cv2.rectangle(
            image,
            (x, y),
            (x + int(rng.integers(20, 60)), y + int(rng.integers(15, 45))),
            int(rng.integers(0, 256)),
            -1,
        )
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def render(template: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(
        template,
        homography,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(24, 24, 24),
    )


def expected_quad(homography: np.ndarray) -> np.ndarray:
    corners = np.float32([[0, 0], [W, 0], [W, H], [0, H]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, homography).reshape(-1, 2)


def translation(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def scaling(factor: float) -> np.ndarray:
    tx = (1.0 - factor) * W / 2.0
    ty = (1.0 - factor) * H / 2.0
    return np.array([[factor, 0.0, tx], [0.0, factor, ty], [0.0, 0.0, 1.0]])


def assert_quad_close(actual: np.ndarray, expected: np.ndarray, tol: float = 2.5) -> None:
    assert actual is not None
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=tol, rtol=0)


@pytest.fixture(scope="module")
def template() -> np.ndarray:
    return make_template()


# ------------------------------------------------------------------ tracker
class TestKeypointTracker:
    def test_empty_template_rejected(self):
        with pytest.raises(ValueError):
            KeypointTracker(np.zeros((0, 0, 3), np.uint8))

    def test_flat_template_rejected(self):
        with pytest.raises(ValueError):
            KeypointTracker(np.full((120, 160, 3), 127, np.uint8))

    def test_unknown_detector_rejected(self, template):
        with pytest.raises(ValueError, match="unknown detector"):
            KeypointTracker(template, TrackerConfig(detector="surf"))

    def test_identity_frame(self, template):
        tracker = KeypointTracker(template)
        result = tracker.process(template)
        assert result.found
        assert result.matches > 0
        assert result.inliers >= tracker.config.strong_inliers
        assert_quad_close(result.quad, expected_quad(np.eye(3)), tol=0.5)
        assert result.area == pytest.approx(W * H, rel=0.01)

    @pytest.mark.parametrize(
        ("dx", "dy"),
        [(0, 0), (35, 22), (-28, 17), (12, -19)],
    )
    def test_translation(self, template, dx, dy):
        tracker = KeypointTracker(template)
        homography = translation(dx, dy)
        result = tracker.process(render(template, homography))
        assert result.found
        assert_quad_close(result.quad, expected_quad(homography))

    @pytest.mark.parametrize("factor", [1.0, 0.8, 0.65, 0.5])
    def test_scale(self, template, factor):
        tracker = KeypointTracker(template)
        homography = scaling(factor)
        result = tracker.process(render(template, homography))
        assert result.found
        assert_quad_close(result.quad, expected_quad(homography))
        assert result.area == pytest.approx(W * H * factor**2, rel=0.03)

    def test_perspective(self, template):
        tracker = KeypointTracker(template)
        homography = np.array(
            [
                [0.86, 0.06, 25.0],
                [-0.04, 0.90, 25.0],
                [6e-5, 3e-5, 1.0],
            ]
        )
        result = tracker.process(render(template, homography))
        assert result.found
        assert_quad_close(result.quad, expected_quad(homography))

    def test_blank_frame_not_found(self, template):
        tracker = KeypointTracker(template)
        assert not tracker.process(np.full((H, W, 3), 128, np.uint8)).found

    def test_keyframe_added_for_small_object(self, template):
        # A single jump from a full frame to half scale is not physical for a
        # real video, so relax the temporal area guard for this test.
        tracker = KeypointTracker(template, TrackerConfig(temporal_area_ratio=10.0))
        assert tracker.process(template).found
        result = tracker.process(render(template, scaling(0.5)))
        assert result.found
        assert result.keyframe_added
        assert tracker.keyframe_count == 1

    def test_keyframe_database_still_tracks(self, template):
        tracker = KeypointTracker(template, TrackerConfig(temporal_area_ratio=10.0))
        tracker.process(template)
        tracker.process(render(template, scaling(0.5)))
        homography = translation(15, 10) @ scaling(0.5)
        result = tracker.process(render(template, homography))
        assert result.found
        assert_quad_close(result.quad, expected_quad(homography))


class TestGeometryValidation:
    def setup_method(self):
        self.tracker = KeypointTracker(make_template())
        self.frame_area = float(W * H)

    @staticmethod
    def square(cx: float, cy: float, side: float) -> np.ndarray:
        half = side / 2.0
        return np.float32(
            [
                [cx - half, cy - half],
                [cx + half, cy - half],
                [cx + half, cy + half],
                [cx - half, cy + half],
            ]
        )

    def test_regular_square_valid(self):
        quad = self.square(160, 120, 100)
        assert self.tracker._valid_geometry(quad, 10_000.0, 100, gap=10)

    def test_tiny_quad_invalid(self):
        quad = self.square(160, 120, 8)
        assert not self.tracker._valid_geometry(quad, 64.0, 100, gap=10)

    def test_huge_quad_invalid(self):
        quad = self.square(160, 120, 320)
        area = float(cv2.contourArea(quad))
        assert area > self.frame_area * self.tracker.config.max_box_area_ratio
        assert not self.tracker._valid_geometry(quad, area, 100, gap=10)

    def test_concave_quad_invalid(self):
        quad = np.float32([[40, 40], [280, 40], [40, 200], [280, 200]])
        area = abs(float(cv2.contourArea(quad)))
        assert not self.tracker._valid_geometry(quad, area, 100, gap=10)

    def test_sliver_quad_invalid(self):
        quad = np.float32([[20, 100], [300, 104], [300, 108], [20, 104]])
        area = abs(float(cv2.contourArea(quad)))
        assert area >= self.tracker.config.min_box_area
        assert not self.tracker._valid_geometry(quad, area, 100, gap=10)

    def test_far_away_quad_invalid(self):
        quad = self.square(2000, 2000, 100)
        assert not self.tracker._valid_geometry(quad, 10_000.0, 100, gap=10)

    def test_temporal_jump_invalid(self):
        self.tracker._last_area = 10_000.0
        self.tracker._last_frame = 0
        jump = self.square(160, 120, 200)
        assert not self.tracker._valid_geometry(jump, 40_000.0, 100, gap=1)

    def test_temporal_smooth_change_valid(self):
        self.tracker._last_area = 10_000.0
        self.tracker._last_frame = 0
        smooth = self.square(160, 120, 141)
        area = abs(float(cv2.contourArea(smooth)))
        assert self.tracker._valid_geometry(smooth, area, 100, gap=1)


class TestDrawing:
    def test_box_and_label_are_drawn(self, template):
        tracker = KeypointTracker(template)
        result = tracker.process(template)
        canvas = template.copy()
        draw_tracking(canvas, result, "Object")
        assert not np.array_equal(canvas, template)
        green = (canvas[:, :, 1] > 180) & (canvas[:, :, 0] < 100) & (canvas[:, :, 2] < 100)
        assert int(green.sum()) > 200

    def test_no_draw_without_detection(self, template):
        tracker = KeypointTracker(template)
        canvas = template.copy()
        result = tracker.process(np.full((H, W, 3), 128, np.uint8))
        draw_tracking(canvas, result, "Object")
        assert np.array_equal(canvas, template)


# ---------------------------------------------------------------------- CLI
def write_test_video(path: Path, template: np.ndarray, frames: int = 8) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (W, H),
    )
    assert writer.isOpened()
    for index in range(frames):
        homography = translation(6.0 * index, 4.0 * index) @ scaling(1.0 - 0.02 * index)
        writer.write(render(template, homography))
    writer.release()


def test_cli_tracks_video(tmp_path):
    template = make_template(7)
    source = tmp_path / "input.avi"
    output = tmp_path / "tracked.mp4"
    stats_path = tmp_path / "stats.json"
    write_test_video(source, template)

    code = main(
        [
            str(source),
            "-o",
            str(output),
            "--label",
            "Test",
            "--stats",
            str(stats_path),
            "--quiet",
        ]
    )
    assert code == 0
    assert output.is_file()

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["frames"] == 8
    assert stats["frames_with_box"] >= 7
    assert stats["frames_with_box_ratio"] >= 0.85
    assert stats["config"]["detector"] == "sift"

    capture = cv2.VideoCapture(str(output))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
    ok, frame = capture.read()
    capture.release()
    assert ok and frame.shape == (H, W, 3)


def test_cli_missing_input(tmp_path):
    assert main([str(tmp_path / "missing.avi"), "--quiet"]) == 1
