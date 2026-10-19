from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from lab1.erosion import binarize, erode_library, erode_manual


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Бинарная эрозия 3x3 (ЛР1, вар. 7)")
    p.add_argument("input", help="входное изображение (jpg/png/...)")
    p.add_argument("-o", "--output", default=None, help="куда сохранить")
    p.add_argument("--thresh", type=int, default=127, help="порог бинаризации (0-255)")
    p.add_argument("--method", choices=["library", "manual"], default="library")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.thresh <= 255:
        print(f"thresh должен быть 0-255, got {args.thresh}", file=sys.stderr)
        return 2
    src = Path(args.input)
    if not src.is_file():
        print(f"файл не найден: {src}", file=sys.stderr)
        return 1
    gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"не удалось прочитать изображение: {src}", file=sys.stderr)
        return 1
    binary = binarize(gray, thresh=args.thresh)
    eroded = erode_library(binary) if args.method == "library" else erode_manual(binary)
    dst = Path(args.output) if args.output else src.with_name(f"{src.stem}_eroded.png")
    if not cv2.imwrite(str(dst), eroded):
        print(f"не удалось записать: {dst}", file=sys.stderr)
        return 1
    print(
        f"ok: {src} ({gray.shape[1]}x{gray.shape[0]}) -> {dst} "
        f"[{args.method}, thresh={args.thresh}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
