from typing import Tuple, Optional
import cv2
import numpy as np


def estimate_homography(
    pts0: np.ndarray, pts1: np.ndarray, ransac_reproj_threshold: float = 3.0
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Estimates Homography matrix H mapping pts1 -> pts0 using OpenCV RANSAC.
    """
    if pts0 is None or pts1 is None or len(pts0) < 4:
        return None, None, 0

    H, mask = cv2.findHomography(pts1, pts0, cv2.RANSAC, ransac_reproj_threshold)
    inliers_count = int(np.sum(mask)) if mask is not None else 0
    return H, mask, inliers_count


def extract_uncalibrated_servoing_error(
    H: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extracts visual feature vector s (8x1) and error vector e = s - s* (8x1)
    for Uncalibrated Visual Servoing.
    s* corresponds to the normalized Identity matrix [1, 0, 0, 0, 1, 0, 0, 0]^T.
    """
    if H is None:
        return None, None

    # Normalize by H[2, 2]
    H_norm = H / H[2, 2]

    # Extract 8 independent components
    s = np.array(
        [
            H_norm[0, 0],
            H_norm[0, 1],
            H_norm[0, 2],
            H_norm[1, 0],
            H_norm[1, 1],
            H_norm[1, 2],
            H_norm[2, 0],
            H_norm[2, 1],
        ],
        dtype=np.float32,
    ).reshape(8, 1)

    s_star = np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32).reshape(8, 1)
    e = s - s_star

    return s, e
