from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import disk, remove_small_objects, opening, closing


@dataclass
class BlobDetectorConfig:
    temporal_window: int = 5
    spatial_sigma: float = 1.0

    weight_var: float = 1.0
    weight_diff: float = 1.0

    threshold_mode: Literal["otsu", "percentile"] = "otsu"
    percentile_threshold: float = 99.0
    threshold_scale: float = 1.0

    min_area: int = 20
    max_area: int = 5000

    opening_radius: int = 1
    closing_radius: int = 2

    smooth_score_sigma: float = 1.0


def _ensure_3d(stack: np.ndarray) -> np.ndarray:
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"Expected stack with shape (T, Y, X), got {stack.shape}")
    return stack.astype(np.float32, copy=False)


def _robust_normalize(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    lo = np.percentile(img, p_low)
    hi = np.percentile(img, p_high)
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - lo) / (hi - lo)
    return np.clip(out, 0, 1).astype(np.float32)


def temporal_variance(stack: np.ndarray, window: int = 5) -> np.ndarray:
    stack = _ensure_3d(stack)
    if window < 2:
        raise ValueError("temporal_window must be >= 2")

    half = window // 2
    out = np.zeros_like(stack, dtype=np.float32)

    for t in range(stack.shape[0]):
        t0 = max(0, t - half)
        t1 = min(stack.shape[0], t + half + 1)
        out[t] = np.var(stack[t0:t1], axis=0)

    return out


def frame_difference(stack: np.ndarray) -> np.ndarray:
    stack = _ensure_3d(stack)
    out = np.zeros_like(stack, dtype=np.float32)

    if stack.shape[0] == 1:
        return out

    out[0] = np.abs(stack[1] - stack[0])
    out[-1] = np.abs(stack[-1] - stack[-2])

    for t in range(1, stack.shape[0] - 1):
        out[t] = 0.5 * np.abs(stack[t + 1] - stack[t - 1])

    return out


def detection_score(stack: np.ndarray, cfg: BlobDetectorConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = _ensure_3d(stack)

    # Light spatial smoothing first to suppress pixel noise
    if cfg.spatial_sigma > 0:
        smoothed = np.stack(
            [gaussian_filter(frame, sigma=cfg.spatial_sigma) for frame in stack],
            axis=0,
        )
    else:
        smoothed = stack.copy()

    tvar = temporal_variance(smoothed, window=cfg.temporal_window)
    fdiff = frame_difference(smoothed)

    score = np.zeros_like(smoothed, dtype=np.float32)
    for t in range(stack.shape[0]):
        v = _robust_normalize(tvar[t])
        d = _robust_normalize(fdiff[t])
        s = cfg.weight_var * v + cfg.weight_diff * d

        if cfg.smooth_score_sigma > 0:
            s = gaussian_filter(s, sigma=cfg.smooth_score_sigma)

        score[t] = _robust_normalize(s)

    return score, tvar, fdiff


def _threshold_frame(score_frame: np.ndarray, cfg: BlobDetectorConfig) -> np.ndarray:
    if cfg.threshold_mode == "otsu":
        try:
            thr = threshold_otsu(score_frame)
        except ValueError:
            thr = 1.0
    elif cfg.threshold_mode == "percentile":
        thr = np.percentile(score_frame, cfg.percentile_threshold)
    else:
        raise ValueError(f"Unknown threshold_mode: {cfg.threshold_mode}")

    thr *= cfg.threshold_scale
    mask = score_frame > thr
    return mask


def detect_blobs(
    stack: np.ndarray,
    cfg: BlobDetectorConfig | None = None,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Parameters
    ----------
    stack
        Registered movie, shape (T, Y, X)
    cfg
        Blob detector configuration

    Returns
    -------
    label_stack
        Labeled mask stack, shape (T, Y, X). 0 = background.
    detections_df
        One row per blob per frame.
    score_stack
        Detection score stack used for thresholding.
    """
    if cfg is None:
        cfg = BlobDetectorConfig()

    stack = _ensure_3d(stack)
    score_stack, _, _ = detection_score(stack, cfg)

    label_stack = np.zeros_like(stack, dtype=np.int32)
    rows: list[dict] = []

    open_fp = disk(cfg.opening_radius) if cfg.opening_radius > 0 else None
    close_fp = disk(cfg.closing_radius) if cfg.closing_radius > 0 else None

    for t in range(stack.shape[0]):
        mask = _threshold_frame(score_stack[t], cfg)

        if open_fp is not None:
            mask = opening(mask, open_fp)
        if close_fp is not None:
            mask = closing(mask, close_fp)

        mask = remove_small_objects(mask, max_size=cfg.min_area)

        lab = label(mask)

        filtered = np.zeros_like(lab, dtype=np.int32)
        next_id = 1

        for region in regionprops(lab, intensity_image=score_stack[t]):
            if region.area < cfg.min_area or region.area > cfg.max_area:
                continue

            coords = region.coords
            filtered[coords[:, 0], coords[:, 1]] = next_id

            y, x = region.centroid
            minr, minc, maxr, maxc = region.bbox

            rows.append(
                {
                    "frame": t,
                    "blob_id": next_id,
                    "area": int(region.area),
                    "centroid_y": float(y),
                    "centroid_x": float(x),
                    "bbox_min_y": int(minr),
                    "bbox_min_x": int(minc),
                    "bbox_max_y": int(maxr),
                    "bbox_max_x": int(maxc),
                    "mean_score": float(region.intensity_mean),
                    "max_score": float(np.max(score_stack[t][coords[:, 0], coords[:, 1]])),
                }
            )
            next_id += 1

        label_stack[t] = filtered

    detections_df = pd.DataFrame(rows)
    return label_stack, detections_df, score_stack



if __name__ == "__main__":
    
    from pathlib import Path
    from tifffile import imread, imwrite
    
    img_path = Path('/home/ben/Downloads/red-1.tif')
    stack = imread(img_path)
    
    cfg = BlobDetectorConfig(
    temporal_window=5,
    spatial_sigma=1.0,
    weight_var=1.0,
    weight_diff=0.5,
    threshold_mode="otsu",
    threshold_scale=0.8,
    min_area=80,
    max_area=3000,
    opening_radius=1,
    closing_radius=1,
    smooth_score_sigma=1.0,
)

    labels, det_df, score = detect_blobs(stack, cfg)

    print(det_df.head())
    print(f"Total detections: {len(det_df)}")
    imwrite(img_path.with_name(img_path.stem + "_labels.tif"), labels.astype(np.uint16))
        
        
    


