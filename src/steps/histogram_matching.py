import numpy as np
import torch
from skimage.exposure import match_histograms

from src.core.base_step import BaseStep
from src.core.context import PipelineContext
from src.core.registry import register_step


@register_step("histogram_matching")
class HistogramMatchingStep(BaseStep):
    """
    Step: Color Histogram Matching (HM)
    Matches the color distribution of img_cur_proc to img_ref_proc.
    """

    def process(self, context: PipelineContext) -> PipelineContext:
        src_np = context.img_cur_proc.permute(1, 2, 0).cpu().numpy()
        tmpl_np = context.img_ref_proc.permute(1, 2, 0).cpu().numpy()

        matched_np = match_histograms(src_np, tmpl_np, channel_axis=2)
        matched_np = np.clip(matched_np, 0.0, 1.0)

        context.img_cur_proc = (
            torch.from_numpy(matched_np).permute(2, 0, 1).float()
        )
        return context
