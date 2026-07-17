import os

import cv2 as cv
import numpy as np
import torch
from torch import nn
from torchvision.utils import save_image

from models.octave_model_full import OCTHEDFULL
from utils import load_checkpoint

images_files = ['.png']
device = 'cuda'


class PredictClass(nn.Module):
    def __init__(
        self, net: nn.Module, folder_images: str = './ValidateImages/'
    ):
        super().__init__()

        # Init
        self.net = net
        self.folder_images = folder_images

        images_path_list = []
        for file in os.listdir(folder_images):
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                images_path_list.append(os.path.join(folder_images, file))
        assert len(images_path_list) > 0

        self.images_path_list = images_path_list

    def predict_folder(self):
        preds_dir = './preds/'

        if not (os.path.exists(preds_dir)):
            os.makedirs(preds_dir)

        with torch.no_grad():
            self.net.eval()

            for file_path in self.images_path_list:
                print(f'>>> Processing the image {file_path} <<<')
                image = cv.imread(file_path).astype(np.float32)
                image = image - np.array((
                    104.00698793,  # Minus statistics.
                    116.66876762,
                    122.67891434,
                ))
                image = np.transpose(image, (2, 0, 1))  # HWC to CHW.
                image = image.astype(np.float32)
                image = torch.from_numpy(np.expand_dims(image, 0))

                pred_list = self.net(image)
                _, _, h, w = pred_list[0].shape
                interm_images = torch.zeros((len(pred_list), 1, h, w))
                # for i in range(len(pred_list)):
                #     # Only fetch the first image in the batch.
                #     interm_images[i, 0, :, :] = pred_list[i][0, 0, :, :]
                pred_list[-1]
                image_name = file_path.split('/')[-1]
                save_image(
                    pred_list[-1],
                    os.path.join(preds_dir, 'border_' + image_name),
                )


if __name__ == '__main__':
    net = torch.nn.DataParallel(OCTHEDFULL(device, alpha=0.5))
    load_checkpoint(
        net,
        torch.optim.SGD(net.parameters()),
        './trained_models/OctHED Source/epoch-4-checkpoint.pt',
        device,
    )

    p = PredictClass(net.to(device))
    p.predict_folder()
