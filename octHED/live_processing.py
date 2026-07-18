from main import load_checkpoint
from models.octave_model import OCTHED
import numpy as np
import torch
import torch.nn as nn
import cv2 as cv
import time 




def torch_to_np(tensor : torch.Tensor):
    tensor = (tensor.cpu()).detach()
    array = (tensor.numpy() * 255).astype(np.uint8)
    array = np.squeeze(array)

    array = np.permute_dims(array, (1, 0))

    return array
    



def np_to_torch(array : np.ndarray):
    tensor = torch.from_numpy(array)
    
    tensor = tensor.permute(2, 1, 0)
    tensor = tensor.unsqueeze_(0).float()
    return tensor

def main():
    ALPHA = 0.5
    DEVICE = 'cuda'
    OCT_LAYERS = ['conv2', 'conv3', 'conv4', 'conv5']
    NET = OCTHED(DEVICE, alpha = ALPHA, octave_layers= OCT_LAYERS)
    NET = nn.DataParallel(NET)
    NET.eval()
    PRINT_FREQ = 30
    camera = cv.VideoCapture(0)
    MODEL = load_checkpoint(NET, opt= None, path = './trained_models/OctaveVGG10EpochsAlpha05/epoch-4-checkpoint.pt')
    running = True
    key = 0
    i = 0
    while(running and key & 0xff != ord('q')):
        running, image = camera.read()
        

        key = cv.waitKey(1) 

        cv.imshow('Imagem original', image)

        image = np_to_torch(image)
        present = time.time()
        edge_map = NET(image)[-1]
        future = time.time()
        time_passed = future - present
        edge_map = torch_to_np(edge_map)

        cv.imshow('Mapa de bordas: OCTHED', edge_map)
        i = i + 1
        if i > PRINT_FREQ:
            print(f"Time passed to process {time_passed} in {edge_map.shape}")
            i = 0
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
