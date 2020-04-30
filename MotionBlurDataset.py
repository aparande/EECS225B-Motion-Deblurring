import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image

HEAVY_BLUR_NAME = "_heavy_blur.jpg"
MILD_BLUR_NAME = "_mild_blur.jpg"

LOW_RES_TRANSFORMS = transforms.Compose([
    transforms.CenterCrop(320),
    transforms.Resize(160, interpolation=Image.BICUBIC),
    transforms.ToTensor()
])

HIGH_RES_TRANSFORMS = transforms.Compose([
    transforms.CenterCrop(320),
    transforms.ToTensor()
])

class MotionBlurDataset(Dataset):
    def __init__(self, path, transforms_low = LOW_RES_TRANSFORMS, transforms_high = HIGH_RES_TRANSFORMS):
        self.transforms_low = transforms_low
        self.transforms_high = transforms_high

        self.image_path = os.path.join(path, "images")
        with open(os.path.join(path, "train_ids.txt"), 'r') as id_file:
            self.ids = [img_id.replace("\n", "") for img_id in id_file.readlines()]

    def __len__(self):
        return len(self.ids) * 2

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx % 2 == 0:
            suffix = HEAVY_BLUR_NAME
        else:
            suffix = MILD_BLUR_NAME

        img_idx = idx // 2

        unblurred = Image.open(os.path.join(self.image_path, self.ids[img_idx]+".jpg"))
        blurred = Image.open(os.path.join(self.image_path, self.ids[img_idx]+suffix))

        blurred_low = self.transforms_low(blurred)
        blurred_high = self.transforms_high(blurred)
        unblurred = self.transforms_high(unblurred)

        return blurred_low, blurred_high, unblurred
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ds = MotionBlurDataset("data/train")
    print(len(ds))
    input_low_image, input_high_image, output_image = ds[0]
    plt.subplot(2,3,1)
    plt.imshow(input_low_image.T)
    plt.subplot(2,3,2)
    plt.imshow(input_high_image.T)
    plt.subplot(2, 3, 3)
    plt.imshow(output_image.T)

    input_low_image, input_high_image, output_image = ds[1]
    plt.subplot(2,3,4)
    plt.imshow(input_low_image.T)
    plt.subplot(2,3,5)
    plt.imshow(input_high_image.T)
    plt.subplot(2, 3, 6)
    plt.imshow(output_image.T)
    plt.show()