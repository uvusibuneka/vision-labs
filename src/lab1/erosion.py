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
    padded = np.pad(binary, 1, mode="constant", constant_values=0)
    top = padded[:-2, :]
    mid = padded[1:-1, :]
    bot = padded[2:, :]
    out = (
        (top[:, :-2] == FOREGROUND)
        & (top[:, 1:-1] == FOREGROUND)
        & (top[:, 2:] == FOREGROUND)
        & (mid[:, :-2] == FOREGROUND)
        & (mid[:, 1:-1] == FOREGROUND)
        & (mid[:, 2:] == FOREGROUND)
        & (bot[:, :-2] == FOREGROUND)
        & (bot[:, 1:-1] == FOREGROUND)
        & (bot[:, 2:] == FOREGROUND)
    )
    return out.astype(np.uint8) * FOREGROUND
