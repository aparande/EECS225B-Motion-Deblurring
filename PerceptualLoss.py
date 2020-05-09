import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

VGG_TRANSFORM = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PerceptualLoss(torch.nn.Module):
    def __init__(self, device=None):
        super(PerceptualLoss, self).__init__()
        self.vgg = torch.hub.load('pytorch/vision:v0.5.0', 'vgg16', pretrained="True").features[:3].eval()
        self.vgg.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

        if device is not None:
            self.vgg = self.vgg.to(device)
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)

    def forward(self, input_img, target_img):
        input_img = (input_img - self.mean) / self.std
        target_img = (target_img - self.mean) / self.std

        input_features = self.vgg(input_img)
        target_features = self.vgg(target_img)

        return nn.functional.mse_loss(input_features, target_features)
