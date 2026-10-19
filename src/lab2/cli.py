"""Command line interface: track an object in a video file.

The first frame of the video must show the object large (it is the
template).  On every frame where the object is recognised a green box and
a text label are drawn, the result is written to a new video file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from lab2.tracker import KeypointTracker, TrackerConfig, draw_tracking


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="track",
        description="Трекинг плоского объекта по ключевым точкам (ЛР2).",
    )
    p.add_argument("input", help="входное видео; первый кадр — крупное изображение объекта")
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="выходной видеофайл (по умолчанию <имя>_tracked.mp4)",
    )
    p.add_argument("--label", default="Object", help="подпись рядом с рамкой")
    p.add_argument("--detector", choices=["sift", "orb"], default="sift", help="детектор точек")
    p.add_argument(
        "--max-features",
        type=int,
        default=1200,
        help="максимум ключевых точек на изображение",
    )
    p.add_argument(
        "--min-inliers",
        type=int,
        default=10,
        help="минимум инлайеров RANSAC для отрисовки рамки",
    )
    p.add_argument("--ratio", type=float, default=0.75, help="порог теста Лоу (distance ratio)")
    p.add_argument(
        "--no-keypoints",
        action="store_true",
        help="не рисовать найденные ключевые точки",
    )
    p.add_argument("--stats", default=None, help="сохранить статистику прогона в JSON")
    p.add_argument("--quiet", action="store_true", help="не печатать прогресс")
    return p


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
    """Try a couple of fourcc codes so the writer works on any build."""
    for tag, ext in (("mp4v", ".mp4"), ("avc1", ".mp4"), ("XVID", ".avi")):
        target = path if path.suffix.lower() == ext else path.with_suffix(ext)
        fourcc = cv2.VideoWriter_fourcc(*tag)
        writer = cv2.VideoWriter(str(target), fourcc, fps, size)
        if writer.isOpened():
            return writer, target
        writer.release()
    raise RuntimeError(f"could not open a video writer for {path}")


def _default_output(src: Path) -> Path:
    return src.with_name(f"{src.stem}_tracked.mp4")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    src = Path(args.input)
    if not src.is_file():
        print(f"видеофайл не найден: {src}", file=sys.stderr)
        return 1

    capture = cv2.VideoCapture(str(src))
    if not capture.isOpened():
        print(f"не удалось открыть видео: {src}", file=sys.stderr)
        return 1

    ok, template = capture.read()
    if not ok or template is None:
        print(f"не удалось прочитать первый кадр: {src}", file=sys.stderr)
        capture.release()
        return 1

    height, width = template.shape[:2]
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    config = TrackerConfig(
        detector=args.detector,
        max_features=args.max_features,
        min_inliers=args.min_inliers,
        ratio=args.ratio,
    )
    try:
        tracker = KeypointTracker(template, config)
    except ValueError as exc:
        print(f"ошибка инициализации трекера: {exc}", file=sys.stderr)
        capture.release()
        return 1

    out_path = Path(args.output) if args.output else _default_output(src)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer, out_path = _open_writer(out_path, fps, (width, height))

    written = 0
    found_frames = 0
    inliers: list[int] = []
    areas: list[float] = []
    started = time.perf_counter()

    frame = template
    while True:
        result = tracker.process(frame)
        if result.found:
            found_frames += 1
            inliers.append(result.inliers)
            areas.append(result.area)
        draw_tracking(
            frame,
            result,
            args.label,
            draw_keypoints=not args.no_keypoints,
        )
        writer.write(frame)
        written += 1
        if not args.quiet and written % 50 == 0:
            print(f"\rкадр {written}/{total or '?'}", end="", file=sys.stderr, flush=True)
        ok, frame = capture.read()
        if not ok:
            break

    writer.release()
    capture.release()
    elapsed = time.perf_counter() - started
    if not args.quiet:
        print(file=sys.stderr)

    stats = {
        "input": str(src),
        "output": str(out_path),
        "label": args.label,
        "detector": args.detector,
        "frames": written,
        "frames_with_box": found_frames,
        "frames_with_box_ratio": round(found_frames / written, 4) if written else 0.0,
        "keyframes": tracker.keyframe_count,
        "mean_inliers": round(float(np.mean(inliers)), 2) if inliers else 0.0,
        "min_inliers": int(min(inliers)) if inliers else 0,
        "max_inliers": int(max(inliers)) if inliers else 0,
        "mean_box_area": round(float(np.mean(areas)), 1) if areas else 0.0,
        "elapsed_sec": round(elapsed, 2),
        "processing_fps": round(written / elapsed, 2) if elapsed > 0 else 0.0,
        "config": asdict(config),
    }
    if args.stats:
        Path(args.stats).write_text(
            json.dumps(stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
