import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

from src.core.base_step import BaseStep
from src.core.context import PipelineContext
from src.core.registry import register_step

# Lazy-loaded singleton models
_extractor = None
_matcher = None


def get_feature_models(
    max_num_keypoints: int = 2048,
    depth_confidence: float = 0.95,
    width_confidence: float = 0.99,
    filter_threshold: float = 0.1,
    device: str = None,
):
    global _extractor, _matcher
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if _extractor is None:
        _extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)

    if _matcher is None:
        _matcher = (
            LightGlue(
                feature="superpoint",
                flash=True,
                depth_confidence=depth_confidence,
                width_confidence=width_confidence,
            )
            .eval()
            .to(device)
        )

    return _extractor, _matcher, device


@register_step("feature_matching")
class FeatureMatchingStep(BaseStep):
    """
    Step: Deep Feature Extraction & Matching (SuperPoint + LightGlue)
    Extracts keypoints and computes matches.
    Translates cropped ROI keypoint coordinates back to full image space.
    """

    def __init__(
        self,
        name: str = "feature_matching",
        enabled: bool = True,
        max_num_keypoints: int = 2048,
        filter_threshold: float = 0.1,
        depth_confidence: float = 0.95,
        width_confidence: float = 0.99,
        **kwargs,
    ):
        super().__init__(name=name, enabled=enabled, **kwargs)
        self.max_num_keypoints = max_num_keypoints
        self.filter_threshold = filter_threshold
        self.depth_confidence = depth_confidence
        self.width_confidence = width_confidence

    def process(self, context: PipelineContext) -> PipelineContext:
        extractor, matcher, device = get_feature_models(
            max_num_keypoints=self.max_num_keypoints,
            depth_confidence=self.depth_confidence,
            width_confidence=self.width_confidence,
        )

        img0_dev = context.img_ref_proc.to(device)
        img1_dev = context.img_cur_proc.to(device)

        # 1. Feature Extraction
        feats0 = extractor.extract(img0_dev)
        feats1 = extractor.extract(img1_dev)

        # 2. Adaptive LightGlue Matching
        matches0 = matcher(
            {
                "image0": feats0,
                "image1": feats1,
                "filter_threshold": self.filter_threshold,
            }
        )

        # Extract stop_layer
        if "stop" in matches0:
            stop_val = matches0["stop"]
            stop_layer = (
                stop_val.item() if hasattr(stop_val, "item") else int(stop_val)
            )
        else:
            stop_layer = -1

        context.stop_layer = stop_layer

        # Remove batch dimension
        feats0, feats1, matches0 = [rbd(x) for x in [feats0, feats1, matches0]]

        kpts0, kpts1 = feats0["keypoints"], feats1["keypoints"]
        matches = matches0["matches"]

        m_kpts0_crop = kpts0[matches[..., 0]]
        m_kpts1_crop = kpts1[matches[..., 1]]

        context.matches = matches.cpu().numpy()

        # 3. Translate coordinates back to original image space
        offset0_tensor = torch.tensor(
            [context.offset_ref[0], context.offset_ref[1]],
            device=m_kpts0_crop.device,
        )
        offset1_tensor = torch.tensor(
            [context.offset_cur[0], context.offset_cur[1]],
            device=m_kpts1_crop.device,
        )

        m_kpts0_orig = (m_kpts0_crop + offset0_tensor).cpu().numpy()
        m_kpts1_orig = (m_kpts1_crop + offset1_tensor).cpu().numpy()

        context.kpts_ref_orig = m_kpts0_orig
        context.kpts_cur_orig = m_kpts1_orig

        return context
