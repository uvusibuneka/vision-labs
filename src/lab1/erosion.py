from __future__ import annotations

import cv2
import numpy as np

KERNEL_SIZE = 3


def binarize(gray: np.ndarray, thresh: int = 127) -> np.ndarray:
    return np.where(gray > thresh, 255, 0).astype(np.uint8)


def erode_library(binary: np.ndarray) -> np.ndarray:
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), dtype=np.uint8)
    out = cv2.erode(binary, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return np.asarray(out, dtype=np.uint8)


def erode_manual(binary: np.ndarray) -> np.ndarray:
    m = binary.astype(bool)
    padded = np.pad(m, 1, mode="constant", constant_values=False)
    top, mid, bot = padded[:-2, :], padded[1:-1, :], padded[2:, :]
    out = (
        top[:, :-2]
        & top[:, 1:-1]
        & top[:, 2:]
        & mid[:, :-2]
        & mid[:, 1:-1]
        & mid[:, 2:]
        & bot[:, :-2]
        & bot[:, 1:-1]
        & bot[:, 2:]
    )
    return out.astype(np.uint8) * 255
