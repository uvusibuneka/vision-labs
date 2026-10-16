from __future__ import annotations

import cv2
import numpy as np

KERNEL_SIZE = 3
FOREGROUND = 255


def binarize(gray: np.ndarray, thresh: int = 127) -> np.ndarray:
    return np.where(gray > thresh, FOREGROUND, 0).astype(np.uint8)


def erode_library(binary: np.ndarray) -> np.ndarray:
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), dtype=np.uint8)
    out = cv2.erode(binary, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return np.asarray(out, dtype=np.uint8)


def erode_manual(binary: np.ndarray) -> np.ndarray:
    height, width = binary.shape
    padded = np.pad(binary, KERNEL_SIZE // 2, mode="constant", constant_values=0)
    mask = (padded // FOREGROUND).astype(np.uint8)
    windows = np.lib.stride_tricks.sliding_window_view(mask, (KERNEL_SIZE, KERNEL_SIZE))
    cols = windows.reshape(height * width, KERNEL_SIZE * KERNEL_SIZE).astype(np.int32)
    kernel_vec = np.ones((KERNEL_SIZE * KERNEL_SIZE, 1), dtype=np.int32)
    counts = cols @ kernel_vec
    out = (counts.reshape(height, width) == KERNEL_SIZE * KERNEL_SIZE).astype(np.uint8)
    return (out * FOREGROUND).astype(np.uint8)
