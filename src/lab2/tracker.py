"""Keypoint-based planar object tracker.

The first video frame is used as the object template.  SIFT (or ORB)
descriptors of the template are matched against every next frame, the
matches are filtered with Lowe's ratio test and a homography is estimated
with RANSAC.  The four corners of the template are projected with that
homography, which gives an oriented box around the object.

To stay robust when the object becomes much smaller than in the first
frame (the camera moves away), the tracker keeps a small database of
"keyframes": descriptor sets harvested from successfully tracked frames.
Every keyframe stores its points in *template* coordinates, therefore all
keyframes can be matched in one pass and give a direct template -> frame
homography.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class TrackerConfig:
    """Tunable parameters of :class:`KeypointTracker`."""

    detector: str = "sift"
    max_features: int = 1200
    scale_pyramid: tuple[float, ...] = (1.0, 0.7, 0.5, 0.35, 0.25, 0.18)
    ratio: float = 0.75
    ransac_threshold: float = 3.0
    ransac_iterations: int = 5000
    ransac_confidence: float = 0.999
    min_inliers: int = 10
    strong_inliers: int = 30
    max_keyframes: int = 10
    keyframe_descriptors: int = 400
    keyframe_area_ratio: float = 0.40
    keyframe_scale_step: float = 0.75
    keyframe_interval: int = 60
    min_box_area: float = 150.0
    max_box_area_ratio: float = 1.2
    temporal_area_ratio: float = 3.0


@dataclass
class TrackResult:
    """Result of processing one frame."""

    found: bool = False
    homography: np.ndarray | None = None
    quad: np.ndarray | None = None
    inliers: int = 0
    matches: int = 0
    area: float = 0.0
    inlier_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    keyframe_added: bool = False


class KeypointTracker:
    """Track a flat textured object given its large image on frame 0."""

    def __init__(self, template_bgr: np.ndarray, config: TrackerConfig | None = None) -> None:
        if template_bgr is None or template_bgr.size == 0:
            raise ValueError("empty template frame")
        self.config = config or TrackerConfig()
        self.template = template_bgr
        self.height, self.width = template_bgr.shape[:2]
        self.frame_area = float(self.width * self.height)

        self.detector, self.norm = _make_detector(self.config.detector, self.config.max_features)
        self.matcher = cv2.BFMatcher(self.norm)
        self._descriptor_size = 32 if self.config.detector == "orb" else 128

        template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        self._base_des, self._base_pts = self._describe_pyramid(template_gray)
        if len(self._base_des) == 0:
            raise ValueError("not enough keypoints in the first frame")

        self._keyframes: list[_Keyframe] = []
        self._db_des = self._base_des
        self._db_pts = self._base_pts
        self._frame_index = 0
        self._last_area: float | None = None
        self._last_frame: int = -10_000

        xs = np.float32([[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]])
        self._corners = xs.reshape(-1, 1, 2)

    # ------------------------------------------------------------------ public
    def process(self, frame_bgr: np.ndarray) -> TrackResult:
        """Match *frame_bgr* against the template/keyframe database."""
        frame_index = self._frame_index
        self._frame_index += 1

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        result = TrackResult()

        if descriptors is None or len(keypoints) < 4 or len(self._db_des) == 0:
            return result

        knn = self.matcher.knnMatch(self._db_des, descriptors, k=2)
        good = [
            m
            for m, n in (pair for pair in knn if len(pair) == 2)
            if m.distance < self.config.ratio * n.distance
        ]
        result.matches = len(good)
        if len(good) < 8:
            return result

        src = np.float32([self._db_pts[m.queryIdx] for m in good])
        dst = np.float32([keypoints[m.trainIdx].pt for m in good])
        homography, mask = cv2.findHomography(
            src,
            dst,
            cv2.USAC_MAGSAC,
            self.config.ransac_threshold,
            maxIters=self.config.ransac_iterations,
            confidence=self.config.ransac_confidence,
        )
        if homography is None or mask is None:
            return result

        mask = mask.ravel().astype(bool)
        inliers = int(mask.sum())
        result.inliers = inliers
        if inliers < self.config.min_inliers:
            return result

        quad = cv2.perspectiveTransform(self._corners, homography).reshape(-1, 2)
        area = abs(float(cv2.contourArea(quad.astype(np.float32))))
        gap = frame_index - self._last_frame
        if not self._valid_geometry(quad, area, inliers, gap):
            return result

        result.found = True
        result.homography = homography
        result.quad = quad
        result.area = area
        result.inlier_points = dst[mask]
        self._last_area = area
        self._last_frame = frame_index

        result.keyframe_added = self._maybe_add_keyframe(
            frame_index, area, inliers, descriptors, good, mask, src
        )
        return result

    @property
    def keyframe_count(self) -> int:
        return len(self._keyframes)

    # ----------------------------------------------------------------- private
    def _describe_pyramid(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Detect features on several rescaled copies of the template.

        Coordinates are mapped back to the original template, so a single
        matching pass covers a wide range of object scales.
        """
        descriptors: list[np.ndarray] = []
        points: list[np.ndarray] = []
        for scale in self.config.scale_pyramid:
            if scale <= 0:
                continue
            if scale == 1.0:
                image = gray
            else:
                interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                image = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interpolation)
            keypoints, des = self.detector.detectAndCompute(image, None)
            if des is None or len(keypoints) < 4:
                continue
            pts = np.float32([kp.pt for kp in keypoints]) / scale
            descriptors.append(des)
            points.append(pts)
        if not descriptors:
            return np.empty((0, self._descriptor_size), np.float32), np.empty((0, 2), np.float32)
        return np.vstack(descriptors), np.vstack(points)

    def _valid_geometry(self, quad: np.ndarray, area: float, inliers: int, gap: int) -> bool:
        cfg = self.config
        if not np.isfinite(quad).all():
            return False
        if area < cfg.min_box_area or area > self.frame_area * cfg.max_box_area_ratio:
            return False
        if not cv2.isContourConvex(quad.astype(np.float32)):
            return False

        center = quad.mean(axis=0)
        margin = 0.25 * max(self.width, self.height)
        if not (-margin <= center[0] <= self.width + margin):
            return False
        if not (-margin <= center[1] <= self.height + margin):
            return False

        sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        if sides.min() < 4.0:
            return False

        # Reject degenerate "slivers": a real projected rectangle fills a
        # reasonable part of its bounding box and is not needle-shaped.
        x_span = float(np.ptp(quad[:, 0]))
        y_span = float(np.ptp(quad[:, 1]))
        if x_span > 2.5 * self.width or y_span > 2.5 * self.height:
            return False
        bbox_area = x_span * y_span
        if bbox_area <= 1.0 or area / bbox_area < 0.15:
            return False
        perimeter = float(sides.sum())
        compactness = 4.0 * np.pi * area / (perimeter * perimeter)
        if compactness < 0.15:
            return False

        # A weak detection must not make the box jump in scale.  This filters
        # random RANSAC solutions on motion-blurred or occluded frames.
        if self._last_area is not None and gap <= 2:
            ratio = max(area / self._last_area, self._last_area / area)
            if ratio > cfg.temporal_area_ratio:
                return False
        return True

    def _maybe_add_keyframe(
        self,
        frame_index: int,
        area: float,
        inliers: int,
        descriptors: np.ndarray,
        good: list[cv2.DMatch],
        mask: np.ndarray,
        src: np.ndarray,
    ) -> bool:
        cfg = self.config
        if inliers < cfg.strong_inliers:
            return False
        if frame_index == 0 or area >= cfg.keyframe_area_ratio * self.frame_area:
            return False

        kfs = self._keyframes
        smallest = min((k.area for k in kfs), default=float("inf"))
        scale_changed = area < cfg.keyframe_scale_step * smallest
        stale = not kfs or frame_index - kfs[-1].frame_index > cfg.keyframe_interval
        if not (scale_changed or stale):
            return False
        if len(kfs) >= cfg.max_keyframes:
            self._drop_largest_keyframe()
            kfs = self._keyframes

        indices = np.array(
            [m.trainIdx for m, keep in zip(good, mask, strict=False) if keep][
                : cfg.keyframe_descriptors
            ],
            dtype=np.int64,
        )
        if len(indices) < 20:
            return False
        points = src[mask][: cfg.keyframe_descriptors].astype(np.float32)
        block = _Keyframe(
            descriptors=descriptors[indices].copy(),
            points=points,
            area=area,
            frame_index=frame_index,
        )
        kfs.append(block)
        self._rebuild_database()
        return True

    def _drop_largest_keyframe(self) -> None:
        if not self._keyframes:
            return
        largest = max(self._keyframes, key=lambda k: k.area)
        self._keyframes = [k for k in self._keyframes if k is not largest]
        self._rebuild_database()

    def _rebuild_database(self) -> None:
        if not self._keyframes:
            self._db_des = self._base_des
            self._db_pts = self._base_pts
            return
        self._db_des = np.vstack([self._base_des] + [k.descriptors for k in self._keyframes])
        self._db_pts = np.vstack([self._base_pts] + [k.points for k in self._keyframes])


@dataclass
class _Keyframe:
    descriptors: np.ndarray
    points: np.ndarray
    area: float
    frame_index: int


def _make_detector(name: str, max_features: int) -> tuple[cv2.Feature2D, int]:
    if name == "sift":
        detector = cv2.SIFT_create(
            nfeatures=max_features,
            contrastThreshold=0.02,
            edgeThreshold=10,
        )
        return detector, cv2.NORM_L2
    if name == "orb":
        detector = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=10,
            fastThreshold=7,
        )
        return detector, cv2.NORM_HAMMING
    raise ValueError(f"unknown detector {name!r}; expected 'sift' or 'orb'")


def draw_tracking(
    frame: np.ndarray,
    result: TrackResult,
    label: str,
    *,
    draw_keypoints: bool = True,
) -> np.ndarray:
    """Draw the projected box, matched keypoints and the label on *frame*."""
    if not result.found or result.quad is None:
        return frame

    quad = np.round(result.quad).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [quad], True, (0, 255, 0), 2, cv2.LINE_AA)

    if draw_keypoints and len(result.inlier_points):
        for x, y in result.inlier_points:
            cv2.drawMarker(
                frame,
                (int(round(x)), int(round(y))),
                (0, 255, 0),
                cv2.MARKER_CROSS,
                5,
                1,
                cv2.LINE_AA,
            )

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(0.8, frame.shape[1] / 1100.0))
    thickness = max(1, int(round(scale * 2)))
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
    top = result.quad[np.argmin(result.quad[:, 1])]
    x = int(round(top[0]))
    y = int(round(top[1])) - 8
    if y - text_h - baseline < 0:
        y = int(round(result.quad[:, 1].max())) + text_h + baseline + 8
    x = max(2, min(x, frame.shape[1] - text_w - 4))
    y = max(text_h + baseline + 2, min(y, frame.shape[0] - 4))
    cv2.putText(frame, label, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, label, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame
