import os
import math
from typing import Optional, Union
import cv2 as cv
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image

from models.octave_model_full import OCTHEDFULL
from models.hed_model import HED
from utils import load_checkpoint

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'trained_models',
    'OctHED Source',
    'epoch-4-checkpoint.pt'
)

DEFAULT_MODEL_PATH_HED = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'trained_models',
    'OctHED Source',
    'checkpoint_hed.pt'
)

def apply_directional_nms(edge_map: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """
    Aplica Non-Maximum Suppression Direcional em Tensores de borda na GPU.
    Formato esperado: (B, 1, H, W)
    """
    device = edge_map.device
    
    # 1. Cria os filtros de Sobel na GPU para achar a direção da borda
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    dx = F.conv2d(edge_map, sobel_x, padding=1)
    dy = F.conv2d(edge_map, sobel_y, padding=1)
    
    # 2. Calcula o ângulo do gradiente em graus e mapeia para 0-180
    angle = torch.atan2(dy, dx) * (180.0 / math.pi)
    angle[angle < 0] += 180.0
    
    # 3. Quantiza as direções em 4 eixos principais
    # 0: Horizontal (Left-Right)
    # 1: Diagonal (Top-Right a Bottom-Left)
    # 2: Vertical (Top-Bottom)
    # 3: Diagonal Oposta (Top-Left a Bottom-Right)
    direction = torch.zeros_like(angle, dtype=torch.uint8)
    direction[(angle >= 22.5) & (angle < 67.5)] = 1
    direction[(angle >= 67.5) & (angle < 112.5)] = 2
    direction[(angle >= 112.5) & (angle < 157.5)] = 3
    # Ângulos > 157.5 ou < 22.5 continuam sendo 0
    
    # 4. Faz um shift na imagem para pegar os vizinhos usando fatiamento (muito mais rápido)
    padded = F.pad(edge_map, (1, 1, 1, 1), mode='replicate')
    
    # Centro
    c = edge_map
    
    # Vizinhos Eixo 0 (Horizontal)
    mag_0_1 = padded[:, :, 1:-1, :-2]
    mag_0_2 = padded[:, :, 1:-1, 2:]
    
    # Vizinhos Eixo 1 (Diagonal 45)
    mag_1_1 = padded[:, :, :-2, 2:]
    mag_1_2 = padded[:, :, 2:, :-2]
    
    # Vizinhos Eixo 2 (Vertical)
    mag_2_1 = padded[:, :, :-2, 1:-1]
    mag_2_2 = padded[:, :, 2:, 1:-1]
    
    # Vizinhos Eixo 3 (Diagonal 135)
    mag_3_1 = padded[:, :, :-2, :-2]
    mag_3_2 = padded[:, :, 2:, 2:]
    
    # 5. Aplica a máscara: O pixel só sobrevive se for maior/igual aos dois vizinhos da SUA direção
    keep = torch.zeros_like(edge_map, dtype=torch.bool)
    
    keep |= (direction == 0) & (c >= mag_0_1) & (c >= mag_0_2)
    keep |= (direction == 1) & (c >= mag_1_1) & (c >= mag_1_2)
    keep |= (direction == 2) & (c >= mag_2_1) & (c >= mag_2_2)
    keep |= (direction == 3) & (c >= mag_3_1) & (c >= mag_3_2)
    
    # Limpa ruídos de fundo (mantém a sua trava original de confiança)
    keep &= (c > threshold)
    
    # Retorna o mapa afinado
    return torch.where(keep, c, torch.zeros_like(c))


class Predictor:
    """
    Class to load the OctHED model once and run predictions on images.
    """

    def __init__(
        self,
        net: Optional[nn.Module] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        lite_mode: bool = False,
    ):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.lite_mode = lite_mode

        if net is not None:
            self.net = net
        else:
            if model_path == 'hed':
                ckpt_path = DEFAULT_MODEL_PATH_HED
                self.net = torch.nn.DataParallel(HED(self.device, lite_mode=lite_mode))
            else:
                ckpt_path = (
                    DEFAULT_MODEL_PATH
                    if (model_path is None or model_path == 'octhed')
                    else model_path
                )
                self.net = torch.nn.DataParallel(OCTHEDFULL(self.device, alpha=0.5, lite_mode=lite_mode))

            load_checkpoint(
                self.net,
                torch.optim.SGD(self.net.parameters()),
                ckpt_path,
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

    def preprocess_batch(self, images: list) -> torch.Tensor:
        """
        Preprocesses a list of images (paths or BGR numpy arrays) into a batch tensor (N, 3, H, W).
        """
        tensors = []
        for img in images:
            if isinstance(img, str) or hasattr(img, '__fspath__'):
                img_path = str(img)
                img_arr = cv.imread(img_path)
                if img_arr is None:
                    raise ValueError(f"Could not read image from path: {img_path}")
            elif isinstance(img, np.ndarray):
                img_arr = img
            else:
                raise TypeError(
                    f"Unsupported image type: {type(img)}. Expected path string or numpy.ndarray."
                )

            img_float = img_arr.astype(np.float32)
            img_sub = img_float - np.array(
                (104.00698793, 116.66876762, 122.67891434), dtype=np.float32
            )
            img_transposed = np.transpose(img_sub, (2, 0, 1))  # HWC to CHW
            tensors.append(img_transposed)

        batch_np = np.stack(tensors, axis=0)  # Shape (N, 3, H, W)
        tensor_img = torch.from_numpy(batch_np).float().to(self.device)
        return tensor_img

    def preprocess_tensor_batch(self, tensors: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses a PyTorch tensor batch (N, 3, H, W) in RGB [0..1] directly into
        mean-subtracted BGR tensor (N, 3, H, W) on self.device without CPU roundtrips.
        """
        if tensors.dim() == 3:
            tensors = tensors.unsqueeze(0)  # Shape (1, 3, H, W)

        tensors = tensors.to(self.device)
        # Reorder RGB to BGR: channels [2, 1, 0]
        bgr_tensors = tensors[:, [2, 1, 0], :, :] * 255.0

        mean = torch.tensor(
            [104.00698793, 116.66876762, 122.67891434],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)

        return bgr_tensors - mean

    def predict(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        save: bool = False,
        save_path: Optional[str] = None,
        scale_factor: float = 1.0,
        use_amp: bool = True,
        use_nms: bool = False,
    ) -> torch.Tensor:
        """
        Runs edge detection prediction on an image.

        Args:
            image (str | np.ndarray | torch.Tensor): Image file path, BGR numpy array, or PyTorch RGB Tensor.
            save (bool): If True, saves the predicted image. Default False.
            save_path (str, optional): Custom path to save image.
            scale_factor (float): Downsampling scale factor (e.g. 0.5 for 4x faster execution). Default 1.0.
            use_amp (bool): If True and on CUDA, uses Automatic Mixed Precision (FP16). Default True.
            use_nms (bool): If True, applies 2D GPU Non-Maximum Suppression (edge thinning). Default False.

        Returns:
            torch.Tensor: Prediction output tensor (shape 1x1xHxW).
        """
        if isinstance(image, torch.Tensor):
            tensor_img = self.preprocess_tensor_batch(image)
        else:
            tensor_img = self.preprocess(image)

        orig_h, orig_w = tensor_img.shape[2], tensor_img.shape[3]

        if scale_factor < 1.0:
            tensor_img_scaled = F.interpolate(
                tensor_img, scale_factor=scale_factor, mode="bilinear", align_corners=False
            )
        else:
            tensor_img_scaled = tensor_img

        with torch.inference_mode():
            if use_amp and (self.device == 'cuda' or (isinstance(self.device, torch.device) and self.device.type == 'cuda')):
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    pred_list = self.net(tensor_img_scaled)
            else:
                pred_list = self.net(tensor_img_scaled)

            prediction = pred_list[-1].float()

        if scale_factor < 1.0:
            prediction = F.interpolate(
                prediction, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )

        if use_nms:
            prediction = apply_directional_nms(prediction)

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

    def predict_batch(
        self,
        images: Union[list, tuple, torch.Tensor],
        save: bool = False,
        save_paths: Optional[list[str]] = None,
        scale_factor: float = 1.0,
        use_amp: bool = True,
        use_nms: bool = False,
    ) -> torch.Tensor:
        """
        Runs edge detection prediction on a batch of images in a single GPU pass.

        Args:
            images (list | tuple | torch.Tensor): List of image file paths, BGR numpy arrays, or stacked/list PyTorch Tensors.
            save (bool): If True, saves the predicted images. Default False.
            save_paths (list[str], optional): Custom list of paths to save images.
            scale_factor (float): Downsampling scale factor (e.g. 0.5 for 4x faster execution). Default 1.0.
            use_amp (bool): If True and on CUDA, uses Automatic Mixed Precision (FP16). Default True.
            use_nms (bool): If True, applies 2D GPU Non-Maximum Suppression (edge thinning). Default False.

        Returns:
            torch.Tensor: Prediction output tensor (shape Nx1xHxW).
        """
        if isinstance(images, torch.Tensor):
            tensor_img = self.preprocess_tensor_batch(images)
        elif isinstance(images, (list, tuple)) and len(images) > 0 and isinstance(images[0], torch.Tensor):
            stacked = torch.stack(list(images), dim=0)
            tensor_img = self.preprocess_tensor_batch(stacked)
        else:
            tensor_img = self.preprocess_batch(images)

        orig_h, orig_w = tensor_img.shape[2], tensor_img.shape[3]

        if scale_factor < 1.0:
            tensor_img_scaled = F.interpolate(
                tensor_img, scale_factor=scale_factor, mode="bilinear", align_corners=False
            )
        else:
            tensor_img_scaled = tensor_img

        with torch.inference_mode():
            if use_amp and (self.device == 'cuda' or (isinstance(self.device, torch.device) and self.device.type == 'cuda')):
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    pred_list = self.net(tensor_img_scaled)
            else:
                pred_list = self.net(tensor_img_scaled)

            prediction = pred_list[-1].float()  # shape (N, 1, H_scaled, W_scaled)

        if scale_factor < 1.0:
            prediction = F.interpolate(
                prediction, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )

        if use_nms:
            prediction = apply_directional_nms(prediction)

        if save:
            if save_paths is None:
                preds_dir = './octHED/preds/'
                save_paths = [
                    os.path.join(preds_dir, f'border_{i}.png')
                    for i in range(len(prediction))
                ]

            for idx, s_path in enumerate(save_paths):
                save_dir = os.path.dirname(s_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                save_image(prediction[idx], s_path)

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
    folder_images = './octHED/ValidateImages/'
    if os.path.exists(folder_images):
        test_files = [
            os.path.join(folder_images, f)
            for f in os.listdir(folder_images)
            if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        if test_files:
            for test_file in test_files:
                print(f"Testing prediction on {test_file}...")
                predictor = Predictor(lite_mode=False)
                out = predictor.predict(test_file, save=True, use_nms=True)
                print(f"Prediction shape: {out.shape}")

