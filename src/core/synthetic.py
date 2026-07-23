from typing import Tuple, Optional
import cv2
import numpy as np
import torch


class SyntheticTransformGenerator:
    """
    Generates synthetic image transformations (Rotation, Scale, Translation, Perspective)
    from a single input image and computes the exact Ground Truth Homography matrix (H_gt).
    """

    def __init__(
        self,
        angle_deg: float = 10.0,
        scale: float = 1.0,
        tx: float = 20.0,
        ty: float = -15.0,
        perspective_x: float = 0.0,
        perspective_y: float = 0.0,
    ):
        self.angle_deg = angle_deg
        self.scale = scale
        self.tx = tx
        self.ty = ty
        self.perspective_x = perspective_x
        self.perspective_y = perspective_y

    def compute_ground_truth_homography(self, width: int, height: int) -> np.ndarray:
        """
        Computes the 3x3 Ground Truth Homography matrix H_gt for the given image dimensions.
        """
        cx, cy = width / 2.0, height / 2.0

        # 1. 2D Similarity transform (Rotation + Scale around center)
        M_rot = cv2.getRotationMatrix2D((cx, cy), self.angle_deg, self.scale)

        # 2. Add Translation
        M_rot[0, 2] += self.tx
        M_rot[1, 2] += self.ty

        # Convert 2x3 Affine matrix to 3x3 Homography matrix
        H_gt = np.eye(3, dtype=np.float32)
        H_gt[:2, :] = M_rot

        # 3. Optional Perspective Warp
        if self.perspective_x != 0.0 or self.perspective_y != 0.0:
            H_gt[2, 0] += self.perspective_x * 1e-4
            H_gt[2, 1] += self.perspective_y * 1e-4

        return H_gt

    def apply(
        self, img_ref_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Applies synthetic transformation to reference image PyTorch tensor (3, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
                (img_ref_tensor, img_cur_tensor, H_gt)
        """
        channels, height, width = img_ref_tensor.shape

        # Compute Ground Truth Homography
        H_gt = self.compute_ground_truth_homography(width, height)

        # Convert PyTorch tensor (3, H, W) [0..1] to OpenCV BGR image
        img_np = (
            img_ref_tensor.permute(1, 2, 0).cpu().numpy() * 255.0
        ).astype(np.uint8)

        # Warp image using H_gt
        img_cur_np = cv2.warpPerspective(
            img_np, H_gt, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )

        # Convert warped BGR image back to PyTorch RGB tensor (3, H, W) [0..1]
        img_cur_tensor = torch.from_numpy(img_cur_np).permute(2, 0, 1).float() / 255.0

        return img_ref_tensor, img_cur_tensor, H_gt


def create_synthetic_pair_from_file(
    image_path: str,
    angle_deg: float = 10.0,
    scale: float = 1.0,
    tx: float = 20.0,
    ty: float = -15.0,
    perspective_x: float = 0.0,
    perspective_y: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Helper function to load a single image file and create a synthetic transformed pair with H_gt.
    """
    from lightglue.utils import load_image

    img_ref = load_image(image_path)
    generator = SyntheticTransformGenerator(
        angle_deg=angle_deg,
        scale=scale,
        tx=tx,
        ty=ty,
        perspective_x=perspective_x,
        perspective_y=perspective_y,
    )
    return generator.apply(img_ref)
