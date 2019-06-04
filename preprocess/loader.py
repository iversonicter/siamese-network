# Author: Wang Yongjie
# Email : yongjie.wang@ntu.edu.sg

import os
import torch
import torch.utils.data as Data
from PIL import Image
from torchvision import transforms

class Face(Data.Dataset):

    def __init__(self, images, labels, transform = None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        img1 = Image.open(image[0])
        img2 = Image.open(image[1])

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

