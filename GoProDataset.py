import torch
from torch.utils.data import Dataset
import os
from transforms import *

class GoProDataset(Dataset):
    def __init__(self, path, transforms_low = LOW_RES_TRANSFORMS, 
                            transforms_high = HIGH_RES_TRANSFORMS,
                            augmentation_transforms = AUGMENTATION_TRANFORMS):
        self.transforms_low = transforms_low
        self.transforms_high = transforms_high
        self.augmentation_transforms = augmentation_transforms

        self.img_paths = []
        for subdir in os.listdir(path):
            dirpath = os.path.join(path, subdir)
            for name in os.listdir(os.path.join(dirpath, "blur")):
                self.img_paths.append((dirpath, name))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        subdir, filename = self.img_paths[idx]
        blurred_path = os.path.join(subdir, "blur", filename)
        unblurred_path = os.path.join(subdir, "sharp", filename)

        unblurred = Image.open(unblurred_path)
        blurred = Image.open(blurred_path)

        if self.augmentation_transforms:
            unblurred, blurred = self.augmentation_transforms(unblurred, blurred)

        blurred_low = self.transforms_low(blurred)
        blurred_high = self.transforms_high(blurred)
        unblurred = self.transforms_high(unblurred)

        return blurred_low, blurred_high, unblurred

if __name__ == "__main__":
    ds = GoProDataset("gopro/train")
    print(len(ds))