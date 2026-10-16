"""LW1 variant 7: binary erosion with 3x3 square structuring element.

Contract (fixed for tests):
- Binary image: 2D ``np.ndarray``, dtype ``uint8``, values only ``{0, 255}``.
- ``binarize``: 2D grayscale ``uint8`` -> binary ``uint8`` via ``src > thresh``.
- ``erode_library`` / ``erode_manual``: binary -> binary, same HxW, dtype ``uint8``.
- Kernel: 3x3 square of ones.
- Border: zero padding (outside image == 0, background).
  Consequence: any output pixel whose 3x3 neighbourhood touches the image
  border or contains a 0 is 0. In particular the outer 1-px frame of the
  output is always 0 (except degenerate images smaller than 3x3, which
  are all 0 unless proven otherwise by tests).
- Empty image (size == 0): raise ``ValueError``.
- Non-binary input to erode_*: raise ``ValueError``.
"""

from __future__ import annotations

import cv2
import numpy as np


def _check_gray(gray: np.ndarray) -> None:
    if not isinstance(gray, np.ndarray):
        raise ValueError(f"expected np.ndarray, got {type(gray)}")
    if gray.size == 0:
        raise ValueError("empty image")
    if gray.ndim != 2:
        raise ValueError(f"expected 2D grayscale image, got ndim={gray.ndim}")
    if gray.dtype != np.uint8:
        raise ValueError(f"expected uint8 dtype, got {gray.dtype}")


def _check_binary(binary: np.ndarray) -> None:
    if not isinstance(binary, np.ndarray):
        raise ValueError(f"expected np.ndarray, got {type(binary)}")
    if binary.size == 0:
        raise ValueError("empty image")
    if binary.ndim != 2:
        raise ValueError(f"expected 2D image, got ndim={binary.ndim}")
    if binary.dtype != np.uint8:
        raise ValueError(f"expected uint8 dtype, got {binary.dtype}")
    unique = set(np.unique(binary).tolist())
    if not unique <= {0, 255}:
        raise ValueError(f"binary image must contain only 0/255, got {sorted(unique)}")


def binarize(gray: np.ndarray, thresh: int = 127) -> np.ndarray:
    """Binarize grayscale image: 255 where gray > thresh, else 0."""
    _check_gray(gray)
    out = np.where(gray > thresh, 255, 0).astype(np.uint8)
    return out


def erode_library(binary: np.ndarray) -> np.ndarray:
    """Erode binary image with 3x3 square via OpenCV (zero-padding border)."""
    _check_binary(binary)
    kernel = np.ones((3, 3), dtype=np.uint8)
    out = cv2.erode(
        binary,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # cv2.erode preserves dtype/shape; enforce uint8 explicitly.
    return np.asarray(out, dtype=np.uint8)


def erode_manual(binary: np.ndarray) -> np.ndarray:
    """Erode binary image with 3x3 square via native Python (zero-padding border)."""
    _check_binary(binary)
    h, w = binary.shape
    padded = np.pad(binary, 1, mode="constant", constant_values=0)
    out = np.zeros((h, w), dtype=np.uint8)
    # Native nested loops: output pixel is 255 iff all 9 neighbours are 255.
    for i in range(h):
        for j in range(w):
            if (
                padded[i, j] == 255
                and padded[i, j + 1] == 255
                and padded[i, j + 2] == 255
                and padded[i + 1, j] == 255
                and padded[i + 1, j + 1] == 255
                and padded[i + 1, j + 2] == 255
                and padded[i + 2, j] == 255
                and padded[i + 2, j + 1] == 255
                and padded[i + 2, j + 2] == 255
            ):
                out[i, j] = 255
    return out
