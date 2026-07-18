from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import torch
import numpy as np


@dataclass
class PipelineContext:
    """
    Data container passed through every step of the pipeline.
    Holds raw images, intermediate processed images, edge maps, ROI offsets,
    feature keypoints, matches, homography, and timing metadata.
    """

    # Raw input tensors (C, H, W) in [0.0, 1.0]
    img_ref_raw: torch.Tensor
    img_cur_raw: torch.Tensor

    # Processed tensors (updated through pipeline steps)
    img_ref_proc: torch.Tensor = None
    img_cur_proc: torch.Tensor = None

    # Edge maps (H, W) uint8
    edge_map_ref: Optional[np.ndarray] = None
    edge_map_cur: Optional[np.ndarray] = None

    # ROI offsets (x_start, y_start) for ref and cur images
    offset_ref: Tuple[int, int] = (0, 0)
    offset_cur: Tuple[int, int] = (0, 0)
    crop_size_ref: Optional[Tuple[int, int]] = None  # (crop_w, crop_h)
    crop_size_cur: Optional[Tuple[int, int]] = None

    # Keypoints and Matches
    kpts_ref_orig: Optional[np.ndarray] = None  # in original image space
    kpts_cur_orig: Optional[np.ndarray] = None  # in original image space
    matches: Optional[np.ndarray] = None
    stop_layer: int = -1

    # Homography & Inliers
    homography: Optional[np.ndarray] = None
    mask_inliers: Optional[np.ndarray] = None
    inliers_count: int = 0

    # Timing metrics per step (step_name -> duration in seconds)
    step_times: Dict[str, float] = field(default_factory=dict)

    # Custom extra metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.img_ref_proc is None:
            self.img_ref_proc = self.img_ref_raw.clone()
        if self.img_cur_proc is None:
            self.img_cur_proc = self.img_cur_raw.clone()
