from concurrent.futures import ThreadPoolExecutor
import sys
from typing import Optional
import cv2
import numpy as np
import torch

from src.core.base_step import BaseStep
from src.core.context import PipelineContext
from src.core.registry import register_step

# Add boruvka to sys.path
sys.path.insert(0, "boruvka-superpixel/pybuild")
try:
    import boruvka_superpixel
except ImportError:
    boruvka_superpixel = None


@register_step("superpixel_reduction")
class SuperpixelReductionStep(BaseStep):
    """
    Step: Superpixel Image Reduction (SH)
    Reconstructs image by averaging pixels within superpixels.
    Algorithm options: 'boruvka', 'slic', 'meanshift', 'none'.
    """

    def __init__(
        self,
        name: str = "superpixel_reduction",
        enabled: bool = True,
        algorithm: str = "boruvka",
        n_superpixels: int = 100,
        compactness: float = 10.0,
        sp: int = 20,
        sr: int = 40,
        **kwargs,
    ):
        super().__init__(name=name, enabled=enabled, **kwargs)
        self.algorithm = algorithm.lower()
        self.n_superpixels = n_superpixels
        self.compactness = compactness
        self.sp = sp
        self.sr = sr

    def _boruvka(
        self, tensor: torch.Tensor, edge_map: Optional[np.ndarray]
    ) -> torch.Tensor:
        if boruvka_superpixel is None:
            raise ImportError(
                "boruvka_superpixel extension is not compiled or available."
            )

        img_np = (tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        img_np = np.ascontiguousarray(img_np)

        if edge_map is None:
            edge_map = np.zeros(img_np.shape[:2], dtype=np.uint8)

        bosupix = boruvka_superpixel.BoruvkaSuperpixel()
        bosupix.build_2d(img_np, edge_map)
        out = bosupix.average(self.n_superpixels, 3, img_np)

        return torch.from_numpy(out).permute(2, 0, 1).float() / 255.0

    def _slic(self, tensor: torch.Tensor) -> torch.Tensor:
        from skimage.color import rgb2lab
        from skimage.segmentation import slic

        img_np = tensor.permute(1, 2, 0).cpu().numpy()
        img_lab = rgb2lab(img_np)

        segments = slic(
            img_lab,
            n_segments=self.n_superpixels,
            compactness=self.compactness,
            start_label=0,
        )

        output = np.zeros_like(img_np)
        for seg_val in np.unique(segments):
            mask = segments == seg_val
            output[mask] = img_np[mask].mean(axis=0)

        return torch.from_numpy(output).permute(2, 0, 1).float()

    def _meanshift(self, tensor: torch.Tensor) -> torch.Tensor:
        img_np = (tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        filtered_np = cv2.pyrMeanShiftFiltering(img_np, self.sp, self.sr)
        return torch.from_numpy(filtered_np).permute(2, 0, 1).float() / 255.0

    def process(self, context: PipelineContext) -> PipelineContext:
        if self.algorithm == "boruvka":
            with ThreadPoolExecutor(max_workers=2) as executor:
                f_ref = executor.submit(
                    self._boruvka, context.img_ref_proc, context.edge_map_ref
                )
                f_cur = executor.submit(
                    self._boruvka, context.img_cur_proc, context.edge_map_cur
                )
                context.img_ref_proc = f_ref.result()
                context.img_cur_proc = f_cur.result()
        elif self.algorithm == "slic":
            with ThreadPoolExecutor(max_workers=2) as executor:
                f_ref = executor.submit(self._slic, context.img_ref_proc)
                f_cur = executor.submit(self._slic, context.img_cur_proc)
                context.img_ref_proc = f_ref.result()
                context.img_cur_proc = f_cur.result()
        elif self.algorithm == "meanshift":
            with ThreadPoolExecutor(max_workers=2) as executor:
                f_ref = executor.submit(self._meanshift, context.img_ref_proc)
                f_cur = executor.submit(self._meanshift, context.img_cur_proc)
                context.img_ref_proc = f_ref.result()
                context.img_cur_proc = f_cur.result()
        else:  # 'none'
            pass

        return context
