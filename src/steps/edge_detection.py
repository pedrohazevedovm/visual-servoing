from typing import Optional
import cv2
import numpy as np
import torch

from src.core.base_step import BaseStep
from src.core.context import PipelineContext
from src.core.registry import register_step

# Lazy-loaded predictor dictionary for OctHED/HED
_octhed_predictors = {}


def get_octhed_predictor(method: str, lite_mode: bool = False):
    global _octhed_predictors
    key = (method.lower(), lite_mode)
    if key not in _octhed_predictors:
        from octHED.predict import Predictor

        _octhed_predictors[key] = Predictor(model_path=method, lite_mode=lite_mode)
    return _octhed_predictors[key]


@register_step("edge_detection")
class EdgeDetectionStep(BaseStep):
    """
    Step: Edge Detection (CE / OctHED)
    Extracts edge maps used to guide graph construction in superpixels.
    Method options: 'canny', 'octhed', 'hed', 'none'.
    """

    def __init__(
        self,
        name: str = "edge_detection",
        enabled: bool = True,
        method: str = "canny",
        threshold1: float = 50.0,
        threshold2: float = 150.0,
        save_predictions: bool = False,
        scale_factor: float = 1.0,
        use_amp: bool = True,
        lite_mode: bool = False,
        **kwargs,
    ):
        super().__init__(name=name, enabled=enabled, **kwargs)
        self.method = method.lower()
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.save_predictions = save_predictions
        self.scale_factor = scale_factor
        self.use_amp = use_amp
        self.lite_mode = lite_mode

        # Pre-warm model weights into memory during initialization so I/O disk loading is excluded from step timing
        if self.enabled and self.method in ("octhed", "hed"):
            get_octhed_predictor(self.method, lite_mode=self.lite_mode)

    def _canny(self, tensor: torch.Tensor) -> np.ndarray:
        img_np = (tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        img_np = np.ascontiguousarray(img_np)
        return cv2.Canny(img_np, self.threshold1, self.threshold2)

    def _octhed(self, tensor: torch.Tensor) -> np.ndarray:
        predictor = get_octhed_predictor(self.method, lite_mode=self.lite_mode)
        img_np = (tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_bgr = np.ascontiguousarray(img_bgr)

        edge_tensor = predictor.predict(
            img_bgr,
            save=self.save_predictions,
            scale_factor=self.scale_factor,
            use_amp=self.use_amp,
        )
        edge_np = edge_tensor.squeeze().cpu().numpy()
        return np.clip(edge_np * 255.0, 0, 255).astype(np.uint8)
    
    def _hed(self, tensor: torch.Tensor) -> np.ndarray:
        predictor = get_octhed_predictor(self.method, lite_mode=self.lite_mode)
        img_np = (tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_bgr = np.ascontiguousarray(img_bgr)

        edge_tensor = predictor.predict(
            img_bgr,
            save=self.save_predictions,
            scale_factor=self.scale_factor,
            use_amp=self.use_amp,
        )
        edge_np = edge_tensor.squeeze().cpu().numpy()
        return np.clip(edge_np * 255.0, 0, 255).astype(np.uint8)

    def process(self, context: PipelineContext) -> PipelineContext:
        if self.method == "canny":
            context.edge_map_ref = self._canny(context.img_ref_proc)
            context.edge_map_cur = self._canny(context.img_cur_proc)
        elif self.method in ("octhed", "hed"):
            predictor = get_octhed_predictor(self.method, lite_mode=self.lite_mode)

            # Pass PyTorch RGB tensors directly (N=2, C=3, H, W) - Zero CPU/NumPy conversion before GPU
            batch_tensors = torch.stack(
                [context.img_ref_proc, context.img_cur_proc], dim=0
            )

            edge_tensors = predictor.predict_batch(
                batch_tensors,
                save=self.save_predictions,
                scale_factor=self.scale_factor,
                use_amp=self.use_amp,
            )

            edge_ref_np = edge_tensors[0].squeeze().cpu().numpy()
            edge_cur_np = edge_tensors[1].squeeze().cpu().numpy()

            context.edge_map_ref = np.clip(edge_ref_np * 255.0, 0, 255).astype(np.uint8)
            context.edge_map_cur = np.clip(edge_cur_np * 255.0, 0, 255).astype(np.uint8)
        else:  # 'none'
            context.edge_map_ref = None
            context.edge_map_cur = None

        return context
