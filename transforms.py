import torch
import numpy as np
from torchvision import transforms
from PIL import Image

class RandomSquareCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, *images):
        w, h  = images[0].size
        new_h, new_w = self.size, self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        cropped_images = []
        for img in images:
            img = img.crop((left, top, left + new_w, top + new_h))
            cropped_images.append(img)
        return cropped_images

class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p
    def __call__(self, *images):
        if np.random.random_sample() > self.p:
            return images
        
        flipped_images = []
        for img in images:
            flipped_images.append(img.transpose(Image.FLIP_LEFT_RIGHT))
        return flipped_images

class RandomVerticalFlip(object):
    def __init__(self, p):
        self.p = p
    def __call__(self, *images):
        if np.random.random_sample() > self.p:
            return images
        
        flipped_images = []
        for img in images:
            flipped_images.append(img.transpose(Image.FLIP_TOP_BOTTOM))
        return flipped_images

class ComposedTransforms(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *images):
        for trans in self.transforms:
            images = trans(*images)
        return images

LOW_RES_TRANSFORMS = transforms.Compose([
    transforms.CenterCrop(320),
    transforms.Resize(256, interpolation=Image.BICUBIC),
    transforms.ToTensor()
])

HIGH_RES_TRANSFORMS = transforms.Compose([
    transforms.CenterCrop(320),
    transforms.ToTensor()
])

AUGMENTATION_TRANFORMS = ComposedTransforms([
    RandomSquareCrop(320),
    RandomHorizontalFlip(0.5),
    RandomVerticalFlip(0.5)
])