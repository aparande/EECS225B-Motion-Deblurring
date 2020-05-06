import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

VGG_TRANSFORM = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Truncate network to RELU 3_3
        self.vgg = model = torch.hub.load('pytorch/vision:v0.5.0', 'vgg16', pretrained="True").features[:23].eval()
        self.vgg.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    def forward(self, input_img, target_img):
        #input_img = VGG_TRANSFORM(input_img)
        #target_img = VGG_TRANSFORM(target_img)
        input_img = (input_img - self.mean) / self.std
        target_img = (target_img - self.mean) / self.std

        input_features = self.vgg(input_img)
        target_features = self.vgg(target_img)

        return nn.functional.mse_loss(input_features, target_features)
