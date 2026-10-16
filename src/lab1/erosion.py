"""LW1 variant 7: binary erosion with 3x3 square structuring element.

Contract:
- Binary image: 2D ``np.ndarray``, dtype ``uint8``, values only ``{0, 255}``.
- ``binarize``: 2D grayscale ``uint8`` -> binary ``uint8`` via ``src > thresh``.
- ``erode_library`` / ``erode_manual``: binary -> binary, same HxW, ``uint8``.
- Kernel: 3x3 square of ones, zero padding (outside image == 0).
- Empty image (size == 0): raise ``ValueError``.
- Non-binary input to erode_*: raise ``ValueError``.
"""

from __future__ import annotations

import cv2
import numpy as np

KERNEL_SIZE = 3
FOREGROUND = 255


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
    if not bool(np.all((binary == 0) | (binary == FOREGROUND))):
        raise ValueError("binary image must contain only 0/255")


def binarize(gray: np.ndarray, thresh: int = 127) -> np.ndarray:
    """Binarize grayscale image: 255 where gray > thresh, else 0."""
    _check_gray(gray)
    if not 0 <= thresh <= 255:
        raise ValueError(f"thresh must be in [0, 255], got {thresh}")
    return np.where(gray > thresh, FOREGROUND, 0).astype(np.uint8)


def erode_library(binary: np.ndarray) -> np.ndarray:
    """Erode binary image with 3x3 square via OpenCV (zero-padding border)."""
    _check_binary(binary)
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), dtype=np.uint8)
    out = cv2.erode(binary, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return np.asarray(out, dtype=np.uint8)


def erode_manual(binary: np.ndarray) -> np.ndarray:
    """Erode binary image with 3x3 square via plain Python loops.

    Output pixel is foreground iff every pixel of its 3x3 neighbourhood
    (zero-padded outside the image) is foreground. Early exit on the
    first background pixel inside the window.
    """
    _check_binary(binary)
    height, width = binary.shape
    pad = KERNEL_SIZE // 2
    padded = np.pad(binary, pad, mode="constant", constant_values=0)
    out = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            eroded = True
            for ki in range(KERNEL_SIZE):
                for kj in range(KERNEL_SIZE):
                    if padded[i + ki, j + kj] != FOREGROUND:
                        eroded = False
                        break
                if not eroded:
                    break
            if eroded:
                out[i, j] = FOREGROUND
    return out
