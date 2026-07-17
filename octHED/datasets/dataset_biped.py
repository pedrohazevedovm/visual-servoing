import os
from os.path import join, splitext
import cv2 
import numpy as np
import torch.utils.data as data


class BipedDataset(data.Dataset):
    def __init__(self, dataset_dir='./data/BIPED', split='train', hsv : bool = False):
        # Set dataset directory and split.
        self.dataset_dir = dataset_dir
        self.split = split
        self.use_hsv = hsv 
        # Read the list of images and (possible) edges.
        if self.split == 'train':
            self.list_path = join(self.dataset_dir, 'train_pair.lst')
        else:  # Assume test.
            self.list_path = join(self.dataset_dir, 'test.lst')
        with open(self.list_path, 'r') as f:
            lines = f.readlines()

        lines = [line.strip() for line in lines]  # Remove the newline at last.
        if self.split == 'train':
            pairs = [line.split() for line in lines]
            self.filelist = [pair[0] for pair in pairs]
            self.edges_path = [pair[1] for pair in pairs]
        else:
            self.filelist = lines
            self.images_name = []  # Used to save temporary edges.
            for path in self.filelist:
                folder, filename = os.path.split(path)
                name, ext = splitext(filename)
                self.images_name.append(name)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        edge = None
        if self.split == 'train':
            # Get edge.
            edge_path = join(self.dataset_dir, self.edges_path[index])
            edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
            edge = edge[np.newaxis, :, :]  # Add one channel at first (CHW).
            edge[edge < 127.5] = 0.0
            edge[edge >= 127.5] = 1.0
            edge = edge.astype(np.float32)

        # Get image.
        image_path = join(self.dataset_dir, self.filelist[index])
        image = cv2.imread(image_path)
        if self.use_hsv:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image[:, :, 0] = image[:, :, 0] * 2 # Range [0, 255]
        
        if not(self.use_hsv):
            image = image - np.array((
                104.00698793,  # Minus statistics.
                116.66876762,
                122.67891434,
            ))

        image = np.transpose(image, (2, 0, 1))  # HWC to CHW.
        image = image.astype(np.float32)  # To float32.

        # Return image and (possible) edge.
        if self.split == 'train':
            return image, edge
        else:
            return image
        

def create_train_pair():
    path = './data/BIPED/'
    train_path = 'edges/imgs/train/rgbr/real/'
    gt_path = 'edges/edge_maps/train/rgbr/real/'
    imgs_train = os.listdir(path + train_path)
    imgs_gt = os.listdir(path + gt_path)

    pairs = []
    for train_image_file_name in imgs_train:
        name_train = train_image_file_name.split('.')[0]
        for gt_image_file_name in imgs_gt:
            name_gt = gt_image_file_name.split('.')[0]
            if name_train == name_gt:
                print(name_train, name_gt)
                path_train_image = train_path + train_image_file_name
                path_gt_image = gt_path + gt_image_file_name
                pairs.append(path_train_image + ' ' + path_gt_image + '\n')

    with open(os.path.join(path, 'train_pair.lst'), 'w') as f:
        for pair in pairs:
            f.write(pair)

        f.close()