import torch
from src.core.base_step import BaseStep
from src.core.context import PipelineContext
from src.core.registry import register_step


@register_step("roi_crop")
class ROICropStep(BaseStep):
    """
    Step: Attention ROI Center Crop
    Crops central region of image by percentage (pct_w, pct_h).
    Tracks pixel offsets to map keypoints back to original image space later.
    """

    def __init__(
        self,
        name: str = "roi_crop",
        enabled: bool = True,
        pct_w: float = 0.3,
        pct_h: float = 0.5,
        **kwargs,
    ):
        super().__init__(name=name, enabled=enabled, **kwargs)
        self.pct_w = pct_w
        self.pct_h = pct_h

    def _crop(self, tensor: torch.Tensor):
        C, H, W = tensor.shape
        crop_w = int(W * self.pct_w)
        crop_h = int(H * self.pct_h)

        x_start = (W - crop_w) // 2
        y_start = (H - crop_h) // 2

        cropped = tensor[:, y_start : y_start + crop_h, x_start : x_start + crop_w]
        return cropped, (x_start, y_start), (crop_w, crop_h)

    def process(self, context: PipelineContext) -> PipelineContext:
        (
            context.img_ref_proc,
            context.offset_ref,
            context.crop_size_ref,
        ) = self._crop(context.img_ref_proc)
        (
            context.img_cur_proc,
            context.offset_cur,
            context.crop_size_cur,
        ) = self._crop(context.img_cur_proc)

        return context
