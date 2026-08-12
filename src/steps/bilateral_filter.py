from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import torch

from src.core.base_step import BaseStep
from src.core.context import PipelineContext
from src.core.registry import register_step


@register_step("bilateral_filter")
class BilateralFilterStep(BaseStep):
    """
    Step: Bilateral Filter (BF)
    Smoothes high-frequency noise while keeping sharp structural edges.
    """

    def __init__(
        self,
        name: str = "bilateral_filter",
        enabled: bool = True,
        d: int = 9,
        sigmaColor: float = 75.0,
        sigmaSpace: float = 75.0,
        **kwargs,
    ):
        super().__init__(name=name, enabled=enabled, **kwargs)
        self.d = d
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace

    def _apply_filter(self, tensor: torch.Tensor) -> torch.Tensor:
        img_np = (tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        img_np = np.ascontiguousarray(img_np)
        img_filtered = cv2.bilateralFilter(
            img_np, d=self.d, sigmaColor=self.sigmaColor, sigmaSpace=self.sigmaSpace
        )
        return torch.from_numpy(img_filtered).permute(2, 0, 1).float() / 255.0

    def process(self, context: PipelineContext) -> PipelineContext:
        context.img_ref_proc = self._apply_filter(context.img_ref_proc)
        context.img_cur_proc = self._apply_filter(context.img_cur_proc)
        return context
