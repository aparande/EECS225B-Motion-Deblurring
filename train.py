import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import models, transforms

import matplotlib.pyplot as plt

from tqdm import tqdm

from MotionBlurDataset import MotionBlurDataset
from metrics import psnr

import numpy as np
import argparse
import os

class Trainer():
    def __init__(self, dataset, model, optimizer, criterion, output_name, batch_size):
        self.dataset = dataset
        self.train_loader, self.val_loader = self.split_data(batch_size)

        self.output_name = output_dir
    
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

    def split_data(self, batch_size):
        train_size = int(1 * len(self.dataset))
        val_size = len(self.dataset) - train_size

        train_set, val_set = torch.utils.data.random_split(self.dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)

        return train_loader, val_loader

    def validate_model(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            for data, targets in self.val_loader:
                preds = self.model(data)
                loss = self.criterion(torch.flatten(preds).float(), torch.flatten(targets).float())
                total_loss += loss.item()
            print('Validation Loss: {:.6f}'.format(total_loss / len(self.val_loader)))
            return total_loss / len(self.val_loader)

    def train_model(self, num_epochs=25, resume=False):
        if resume:
            train_losses = list(np.load("{}/train_losses.npy".format(self.output_name)))
            train_psnr = list(np.load("{}/train_psnr.npy".format(self.output_name)))
            #val_losses = list(np.load("{}/val_losses.npy".format(self.output_name)))
        else:
            train_losses = []
            train_psnr = []
            #val_losses = []

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            for batch_idx, (lr, hr, target) in tqdm(enumerate(self.train_loader)):
                self.model.train()
                lr = lr.to(self.device)
                hr = hr.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()

                preds = self.model(lr, hr)

                loss = self.criterion(preds, target)
                loss.backward()

                self.optimizer.step()
                if (batch_idx + 1) % 10 == 0:
                    model_psnr = psnr(target, preds).item()
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t PSNR: {:.6f}\n'.format(epoch, 
                                                                                batch_idx * len(data), 
                                                                                len(self.train_loader.dataset), 
                                                                                100. * batch_idx / len(self.train_loader),
                                                                                loss.item(),
                                                                                model_psnr))
                    train_psnr.append(model_psnr)
                train_losses.append(loss.item())
            torch.save(self.model.state_dict(), '{}/model.pth'.format(self.output_name))
            torch.save(self.optimizer.state_dict(), '{}/optimizer.pth'.format(self.output_name))
            #val_loss = self.validate_model()
            #val_losses.append(val_loss)
        np.save('{}/train_losses.npy'.format(self.output_name), train_losses)
        np.save('{}/train_psnr.npy'.format(self.output_name), train_psnr)
        #np.save('results/{}/val_losses.npy'.format(self.output_name), val_losses)
        self.plot_history(num_epochs, train_losses, train_psnr)


    def plot_history(self, epochs, train_losses, val_losses):
        train_epochs = np.linspace(0, epochs, len(train_losses))
        val_epochs = np.linspace(0, epochs, len(val_losses))
        plt.clf()
        plt.plot(train_epochs, train_losses)
        plt.plot(val_epochs, val_losses)
        plt.xlabel("Epochs")
        plt.legend(["Training Loss", "Train PSNR"])
        plt.savefig('{}/history.png'.format(self.output_name))

parser = argparse.ArgumentParser(description='HDRNet Training')

parser.add_argument('--luma-bins', type=int, default=8)
parser.add_argument('--channel-multiplier', default=1, type=int)
parser.add_argument('--spatial-bin', type=int, default=16)
parser.add_argument('--batch-norm', action='store_true', help='If set use batch norm')

parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--dataset', type=str, default='', help='Dataset path with input/output dirs', required=True)
parser.add_argument('--output-dir', type=str, default='models', help="Trained model directory")

params = vars(parser.parse_args())

print('PARAMS:')
print(params)

os.makedirs(params['output_dir'], exist_ok=True)

dataset = MotionBlurDataset(params['dataset'])
model = HDRPointwiseNN(params=params)
optimizer = optim.Adam(model.parameters(), lr=params['lr'])
criterion = nn.MSELoss(reduction="mean")

trainer = Trainer(dataset, model, optimizer, criterion, params['output_dir'], params['batch_size'])
trainer.train_model(num_epochs=params['epochs'], resume=False)