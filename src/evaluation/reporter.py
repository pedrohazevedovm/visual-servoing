import csv
import json
from pathlib import Path
from typing import Dict, Any, List
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightglue import viz2d

from src.core.context import PipelineContext


class Reporter:
    """
    Handles rendering visualization figures and exporting CSV / JSON metric reports.
    """

    @staticmethod
    def render_visualization(
        context: PipelineContext,
        title: str = "Pipeline Results",
        background_type: str = "raw",  # 'raw' or 'processed'
    ) -> plt.Figure:
        """
        Renders Matplotlib plot of keypoint matches and homography polygon projection.
        """
        if background_type == "processed":
            img0 = context.img_ref_proc
            img1 = context.img_cur_proc
        else:
            img0 = context.img_ref_raw
            img1 = context.img_cur_raw

        plt.close("all")
        plt.figure(figsize=(12, 6))
        viz2d.plot_images([img0, img1])

        # Plot matches
        if (
            context.kpts_ref_orig is not None
            and context.kpts_cur_orig is not None
            and len(context.kpts_ref_orig) > 0
        ):
            if context.mask_inliers is not None and len(context.mask_inliers) > 0:
                inliers_mask = context.mask_inliers.ravel() == 1
                viz2d.plot_matches(
                    torch.tensor(context.kpts_ref_orig[inliers_mask]),
                    torch.tensor(context.kpts_cur_orig[inliers_mask]),
                    color="lime",
                    lw=0.3,
                )
            else:
                viz2d.plot_matches(
                    torch.tensor(context.kpts_ref_orig),
                    torch.tensor(context.kpts_cur_orig),
                    color="yellowgreen",
                    lw=0.2,
                )

        # Plot ROI rectangle if active
        if context.offset_ref != (0, 0) and context.crop_size_ref is not None:
            crop_w, crop_h = context.crop_size_ref
            rect = plt.Rectangle(
                (context.offset_ref[0], context.offset_ref[1]),
                crop_w,
                crop_h,
                edgecolor="red",
                facecolor="none",
                linestyle="--",
                linewidth=1.5,
            )
            plt.gca().add_patch(rect)

        # Plot Homography warped bounding box on current image
        if context.homography is not None:
            W0, H0 = img0.shape[2], img0.shape[1]
            cantos_base = np.array(
                [[0, 0], [W0, 0], [W0, H0], [0, H0]], dtype=np.float32
            ).reshape(-1, 1, 2)
            try:
                cantos_projetados = cv2.perspectiveTransform(
                    cantos_base, np.linalg.inv(context.homography)
                )
                cantos_plot = cantos_projetados.squeeze() + np.array([W0, 0])
                polygon = plt.Polygon(
                    cantos_plot,
                    edgecolor="cyan",
                    facecolor="none",
                    linewidth=2.0,
                    linestyle="-",
                )
                plt.gca().add_patch(polygon)
            except Exception:
                pass

        matches_count = len(context.matches) if context.matches is not None else 0
        text_label = f"{title}\nMatches: {matches_count} | Inliers: {context.inliers_count} | Stop Layer: {context.stop_layer}"
        viz2d.add_text(0, text_label, fs=11)

        return plt.gcf()

    @staticmethod
    def save_json(data: Dict[str, Any], filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def save_csv_summary(rows: List[Dict[str, Any]], filepath: Path):
        if not rows:
            return
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
