import os
import sys
from typing import Optional, Union

# Ensure octHED directory is in sys.path so internal imports (models, utils) resolve
OCTHED_DIR = os.path.dirname(os.path.abspath(__file__))
if OCTHED_DIR not in sys.path:
    sys.path.insert(0, OCTHED_DIR)

import cv2 as cv
import numpy as np
import torch
from torch import nn
from torchvision.utils import save_image

from models.octave_model_full import OCTHEDFULL
from utils import load_checkpoint

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'trained_models',
    'OctHED Source',
    'epoch-4-checkpoint.pt',
)


class Predictor:
    """
    Class to load the OctHED model once and run predictions on images.
    """

    def __init__(
        self,
        net: Optional[nn.Module] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if net is not None:
            self.net = net
        else:
            if model_path is None:
                model_path = DEFAULT_MODEL_PATH

            self.net = torch.nn.DataParallel(OCTHEDFULL(self.device, alpha=0.5))
            load_checkpoint(
                self.net,
                torch.optim.SGD(self.net.parameters()),
                model_path,
                self.device,
            )

        self.net = self.net.to(self.device)
        self.net.eval()

    def preprocess(self, image: Union[str, np.ndarray]) -> torch.Tensor:
        """
        Preprocesses an image (path or BGR numpy array) into a batch tensor.
        """
        if isinstance(image, str) or hasattr(image, '__fspath__'):
            img_path = str(image)
            img_arr = cv.imread(img_path)
            if img_arr is None:
                raise ValueError(f"Could not read image from path: {img_path}")
        elif isinstance(image, np.ndarray):
            img_arr = image
        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}. Expected path string or numpy.ndarray."
            )

        img_float = img_arr.astype(np.float32)
        img_sub = img_float - np.array(
            (104.00698793, 116.66876762, 122.67891434), dtype=np.float32
        )
        img_transposed = np.transpose(img_sub, (2, 0, 1))  # HWC to CHW
        tensor_img = (
            torch.from_numpy(np.expand_dims(img_transposed, 0))
            .float()
            .to(self.device)
        )
        return tensor_img

    def predict(
        self,
        image: Union[str, np.ndarray],
        save: bool = False,
        save_path: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Runs edge detection prediction on an image.

        Args:
            image (str | np.ndarray): Image file path or BGR numpy array.
            save (bool): If True, saves the predicted image. Default False.
            save_path (str, optional): Custom path to save image.

        Returns:
            torch.Tensor: Prediction output tensor (shape 1x1xHxW).
        """
        tensor_img = self.preprocess(image)

        with torch.no_grad():
            pred_list = self.net(tensor_img)
            prediction = pred_list[-1]

        if save:
            if save_path is None:
                preds_dir = './preds/'
                if isinstance(image, str):
                    image_name = os.path.basename(image)
                    save_path = os.path.join(preds_dir, 'border_' + image_name)
                else:
                    save_path = os.path.join(preds_dir, 'output.png')

            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            save_image(prediction, save_path)

        return prediction


_global_predictor: Optional[Predictor] = None


def predict_image(
    image: Union[str, np.ndarray],
    save: bool = False,
    save_path: Optional[str] = None,
    net: Optional[nn.Module] = None,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Convenience function to predict edge map for a single image.

    Args:
        image (str | np.ndarray): Image path or loaded BGR numpy array.
        save (bool): Flag to save output image file. Default False.
        save_path (str, optional): Destination file path if save=True.
        net (nn.Module, optional): Pre-loaded PyTorch model.
        model_path (str, optional): Path to model weights checkpoint.
        device (str, optional): Device to run inference ('cuda' or 'cpu').

    Returns:
        torch.Tensor: Edge prediction tensor.
    """
    global _global_predictor
    if net is not None:
        predictor = Predictor(net=net, device=device)
    else:
        if _global_predictor is None or model_path is not None or device is not None:
            _global_predictor = Predictor(model_path=model_path, device=device)
        predictor = _global_predictor

    return predictor.predict(image, save=save, save_path=save_path)


class PredictClass(nn.Module):
    """
    Legacy class kept for backward compatibility when predicting a full folder.
    """

    def __init__(
        self, net: nn.Module, folder_images: str = './ValidateImages/'
    ):
        super().__init__()
        self.predictor = Predictor(net=net)
        self.folder_images = folder_images

        images_path_list = []
        if os.path.exists(folder_images):
            for file in os.listdir(folder_images):
                if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    images_path_list.append(os.path.join(folder_images, file))
        self.images_path_list = images_path_list

    def predict_folder(self, save: bool = True):
        for file_path in self.images_path_list:
            print(f'>>> Processing the image {file_path} <<<')
            self.predictor.predict(file_path, save=save)


if __name__ == '__main__':
    folder_images = './ValidateImages/'
    if os.path.exists(folder_images):
        test_files = [
            os.path.join(folder_images, f)
            for f in os.listdir(folder_images)
            if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        if test_files:
            print(f"Testing prediction on {test_files[0]}...")
            predictor = Predictor()
            out = predictor.predict(test_files[0], save=True)
            print(f"Prediction shape: {out.shape}")

