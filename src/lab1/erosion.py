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
    _check_gray(gray)
    if not 0 <= thresh <= 255:
        raise ValueError(f"thresh must be in [0, 255], got {thresh}")
    return np.where(gray > thresh, FOREGROUND, 0).astype(np.uint8)


def erode_library(binary: np.ndarray) -> np.ndarray:
    _check_binary(binary)
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), dtype=np.uint8)
    out = cv2.erode(binary, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return np.asarray(out, dtype=np.uint8)


def erode_manual(binary: np.ndarray) -> np.ndarray:
    _check_binary(binary)
    height, width = binary.shape
    padded = np.pad(binary, KERNEL_SIZE // 2, mode="constant", constant_values=0)
    mask = (padded // FOREGROUND).astype(np.uint8)
    windows = np.lib.stride_tricks.sliding_window_view(mask, (KERNEL_SIZE, KERNEL_SIZE))
    cols = windows.reshape(height * width, KERNEL_SIZE * KERNEL_SIZE).astype(np.int32)
    kernel_vec = np.ones((KERNEL_SIZE * KERNEL_SIZE, 1), dtype=np.int32)
    counts = cols @ kernel_vec
    out = (counts.reshape(height, width) == KERNEL_SIZE * KERNEL_SIZE).astype(np.uint8)
    return (out * FOREGROUND).astype(np.uint8)
