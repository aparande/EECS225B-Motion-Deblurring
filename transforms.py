import torch
import numpy as np
from torchvision import transforms
from PIL import Image

class RandomSquareCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, *images):
        h, w  = images[0].size
        new_h, new_w = self.size, self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        cropped_images = []
        for img in images:
            img = img.crop((left, top, left + new_w, top + new_h))
            cropped_images.append(img)
        return cropped_images

LOW_RES_TRANSFORMS = transforms.Compose([
    transforms.CenterCrop(320),
    transforms.Resize(256, interpolation=Image.BICUBIC),
    transforms.ToTensor()
])

HIGH_RES_TRANSFORMS = transforms.Compose([
    transforms.CenterCrop(320),
    transforms.ToTensor()
])

AUGMENTATION_TRANFORMS = RandomSquareCrop(320)