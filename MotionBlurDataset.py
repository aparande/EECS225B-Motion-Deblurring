import torch
from torch.utils.data import Dataset
import os
from transforms import *

HEAVY_BLUR_NAME = "_heavy_blur.jpg"
MILD_BLUR_NAME = "_mild_blur.jpg"

class MotionBlurDataset(Dataset):
    def __init__(self, path, transforms_low = LOW_RES_TRANSFORMS, 
                            transforms_high = HIGH_RES_TRANSFORMS,
                            augmentation_transforms = AUGMENTATION_TRANFORMS):
        self.transforms_low = transforms_low
        self.transforms_high = transforms_high
        self.augmentation_transforms = augmentation_transforms

        self.image_path = os.path.join(path, "Images")
        with open(os.path.join(path, "ids.txt"), 'r') as id_file:
            self.ids = [img_id.replace("\n", "") for img_id in id_file.readlines()]

        self.img_paths = [os.path.join(self.image_path, img_id, name) for img_id in self.ids for name in os.listdir(os.path.join(self.image_path, img_id)) if "_" in name]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        blurred_path = self.img_paths[idx]
        unblurred_path = blurred_path[:blurred_path.find("_")] + ".jpg"

        unblurred = Image.open(unblurred_path)
        blurred = Image.open(blurred_path)

        if self.augmentation_transforms:
            unblurred, blurred = self.augmentation_transforms(unblurred, blurred)

        blurred_low = self.transforms_low(blurred)
        blurred_high = self.transforms_high(blurred)
        unblurred = self.transforms_high(unblurred)

        return blurred_low, blurred_high, unblurred
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ds = MotionBlurDataset("data/train")
    print(len(ds))
    # input_low_image, input_high_image, output_image = ds[0]
    # plt.subplot(2,3,1)
    # plt.imshow(input_low_image.T)
    # plt.subplot(2,3,2)
    # plt.imshow(input_high_image.T)
    # plt.subplot(2, 3, 3)
    # plt.imshow(output_image.T)

    # input_low_image, input_high_image, output_image = ds[1]
    # plt.subplot(2,3,4)
    # plt.imshow(input_low_image.T)
    # plt.subplot(2,3,5)
    # plt.imshow(input_high_image.T)
    # plt.subplot(2, 3, 6)
    # plt.imshow(output_image.T)
    # plt.show()