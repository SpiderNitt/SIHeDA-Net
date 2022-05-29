import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import Tensor
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
import PIL
import pandas as pd
matplotlib.style.use('ggplot')
torch.cuda.empty_cache()
import os 
import random
from typing import Type, Any, Callable, Union, List, Optional

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn.deterministic = True
    # cudnn.benchmark = False

set_seed(1234)

class Dataset(object):
    
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])

class DatasetMNIST(Dataset):
    
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1))
        label = self.data.iloc[index, 0]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label


def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 64, 64)
    return x

class CvBlock(nn.Module):
    """(Conv2d => BN => ReLU) x 2"""

    def __init__(self, in_ch, out_ch):
        super(CvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convblock(x)


class DownBlock(nn.Module):
    """Downscale + (Conv2d => BN => ReLU)*2"""

    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CvBlock(out_ch, out_ch),
        )

    def forward(self, x):
      
        return self.convblock(x)


class UpBlock(nn.Module):
    """(Conv2d => BN => ReLU)*2 + Upscale"""

    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.convblock = nn.Sequential(
            CvBlock(in_ch, in_ch),
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )

    def forward(self, x):

        return self.convblock(x)


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            DownBlock(1, 32),
            DownBlock(32, 64),
            DownBlock(64, 128),
            DownBlock(128, 256),
            nn.Flatten(),
        )
        self.z_mean = nn.Linear(4096, 256)
        self.z_var = nn.Linear(4096, 256)
        self.z_up = nn.Linear(256, 4096)
        self.decoder = nn.Sequential(
            nn.Linear(256,4096),
            Reshape(-1, 256, 4, 4),
            UpBlock(256, 128),
            UpBlock(128, 64),
            UpBlock(64, 32),
            UpBlock(32, 1),
        )

    def encode(self, x):
        h1 = self.encoder(x)
    
        return self.z_mean(h1), self.z_var(h1)

    @staticmethod
    def reparametrize(mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):

        h3 = self.decoder(z)
        return F.sigmoid(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        # print(mu.shape, logvar.shape)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

image_coder = VAE()
image_coder.load_state_dict(torch.load('/home/ubuntu/Latent-Transfer/DomainB/outputs/models/cnn_vae_151.pth'))
image_coder.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64,64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

train_data = DatasetMNIST('/home/ubuntu/Latent-Transfer/DomainB/DomainB_dataset/sign_mnist_train.csv', transform = transform)
test_data = DatasetMNIST('/home/ubuntu/Latent-Transfer/DomainB/DomainB_dataset/sign_mnist_test.csv', transform = transform)

batch_size = 1
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=True
)

dataloader = train_loader

recon_img = []
recon_label = []

for batch_idx, data in enumerate(tqdm(dataloader)):
    
    img, label = data
    img = Variable(img)

    recon_batch, mu, logvar = image_coder(img)
    save = to_img(recon_batch.cpu().data)

    if batch_idx % 100 == 0:
        save_image(save, "/home/ubuntu/Latent-Transfer/DomainB/check_images/image_{}.png".format(batch_idx))

    recon_img.append(recon_batch.detach().numpy())
    recon_label.append(label.detach().numpy())

ri = np.asarray(recon_img)
rl = np.asarray(recon_label)

np.save('/home/ubuntu/Latent-Transfer/DomainB/recon_images_train', ri)
np.save('/home/ubuntu/Latent-Transfer/DomainB/recon_labels_train', rl)