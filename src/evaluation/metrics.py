from typing import Dict, Any, Optional
import cv2
import numpy as np
from src.core.context import PipelineContext
from src.evaluation.homography import (
    estimate_homography,
    extract_uncalibrated_servoing_error,
)


def compute_pipeline_metrics(
    context: PipelineContext,
    ground_truth_H: Optional[np.ndarray] = None,
    ransac_reproj_threshold: float = 3.0,
) -> Dict[str, Any]:
    """
    Computes all quantitative metrics from the executed PipelineContext.
    """
    # 1. Estimate homography if not already computed
    if context.homography is None and context.kpts_ref_orig is not None:
        H, mask, inliers = estimate_homography(
            context.kpts_ref_orig,
            context.kpts_cur_orig,
            ransac_reproj_threshold=ransac_reproj_threshold,
        )
        context.homography = H
        context.mask_inliers = mask
        context.inliers_count = inliers

    matches_count = len(context.matches) if context.matches is not None else 0
    inliers_count = context.inliers_count
    inlier_ratio = (
        (inliers_count / matches_count * 100.0) if matches_count > 0 else 0.0
    )

    # 2. Servoing error vector
    s, e_vector = extract_uncalibrated_servoing_error(context.homography)
    servoing_error_norm = (
        float(np.linalg.norm(e_vector)) if e_vector is not None else None
    )

    # 3. Corner alignment error in pixels
    corner_error_px = None
    if context.homography is not None:
        # Assume standard image dimensions from tensor shape
        H_img, W_img = context.img_ref_raw.shape[1], context.img_ref_raw.shape[2]
        corners_base = np.array(
            [[0, 0], [W_img, 0], [W_img, H_img], [0, H_img]], dtype=np.float32
        ).reshape(-1, 1, 2)

        corners_est = cv2.perspectiveTransform(corners_base, context.homography)

        if ground_truth_H is not None:
            corners_gt = cv2.perspectiveTransform(corners_base, ground_truth_H)
            corner_error_px = float(
                np.mean(np.linalg.norm(corners_gt - corners_est, axis=2))
            )

    total_time_sec = sum(context.step_times.values())

    metrics = {
        "matches_count": matches_count,
        "inliers_count": inliers_count,
        "inlier_ratio_pct": round(inlier_ratio, 2),
        "stop_layer": context.stop_layer,
        "corner_error_px": round(corner_error_px, 4) if corner_error_px is not None else None,
        "servoing_error_norm": round(servoing_error_norm, 6) if servoing_error_norm is not None else None,
        "servoing_error_vector": e_vector.flatten().tolist() if e_vector is not None else None,
        "total_time_sec": round(total_time_sec, 4),
        "step_times": {k: round(v, 4) for k, v in context.step_times.items()},
    }

    return metrics
