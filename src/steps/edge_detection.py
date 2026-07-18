from typing import Optional
import cv2
import numpy as np
import torch

from src.core.base_step import BaseStep
from src.core.context import PipelineContext
from src.core.registry import register_step

# Lazy-loaded predictor for OctHED
_octhed_predictor = None


def get_octhed_predictor():
    global _octhed_predictor
    if _octhed_predictor is None:
        from octHED.predict import Predictor

        _octhed_predictor = Predictor()
    return _octhed_predictor


@register_step("edge_detection")
class EdgeDetectionStep(BaseStep):
    """
    Step: Edge Detection (CE / OctHED)
    Extracts edge maps used to guide graph construction in superpixels.
    Method options: 'canny', 'octhed', 'none'.
    """

    def __init__(
        self,
        name: str = "edge_detection",
        enabled: bool = True,
        method: str = "canny",
        threshold1: float = 50.0,
        threshold2: float = 150.0,
        save_predictions: bool = False,
        **kwargs,
    ):
        super().__init__(name=name, enabled=enabled, **kwargs)
        self.method = method.lower()
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.save_predictions = save_predictions

    def _canny(self, tensor: torch.Tensor) -> np.ndarray:
        img_np = (tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        img_np = np.ascontiguousarray(img_np)
        return cv2.Canny(img_np, self.threshold1, self.threshold2)

    def _octhed(self, tensor: torch.Tensor) -> np.ndarray:
        predictor = get_octhed_predictor()
        img_np = (tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_bgr = np.ascontiguousarray(img_bgr)

        edge_tensor = predictor.predict(img_bgr, save=self.save_predictions)
        edge_np = edge_tensor.squeeze().cpu().numpy()
        return np.clip(edge_np * 255.0, 0, 255).astype(np.uint8)

    def process(self, context: PipelineContext) -> PipelineContext:
        if self.method == "canny":
            context.edge_map_ref = self._canny(context.img_ref_proc)
            context.edge_map_cur = self._canny(context.img_cur_proc)
        elif self.method == "octhed":
            context.edge_map_ref = self._octhed(context.img_ref_proc)
            context.edge_map_cur = self._octhed(context.img_cur_proc)
        else:  # 'none'
            context.edge_map_ref = None
            context.edge_map_cur = None

        return context
