import os
import random
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset, random_split
from tqdm import tqdm
torch.set_printoptions(precision=4)


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def prepare_dataloader(
    batch_size: int = 128,
    num_samples: int = 10000,
    spread: float = 0.5,
    split: float = 0.8,
    seed: int = 1234,
):
    """
    Creates the dataset
    :param batch_size: the batch size
    :param num_samples: number of items per class/label/letter
    :param spread: the std for normal sampling
    :param split: train-val split (<1). Number given is allotted as the train %
    :param seed: seed for the "random" dataset generator
    :return: the dataloaders
    """

    np.random.seed(seed)

    ideal_sensory_values = np.random.randint(-24, 23, (24, 16))
    dataset = list()
    classes = 24
    for letter in range(classes):
        for _ in range(num_samples):
            sensors = []

            for sensor in ideal_sensory_values[letter]:
                sensors.append(np.random.normal(loc=sensor, scale=spread))
            
            sensors= np.array(sensors)
            # if np.random.choice([True,False],p=[0.7,0.3]):
            #     indices = np.random.choice(np.arange(sensors.size), replace=False,size=int(sensors.size * 0.3))
            #     sensors[indices] = np.random.choice([0,1])               
            dataset.append([sensors, np.array([letter])])
    x = list()
    y = list()

    for i in range(num_samples * 24): #24 if we train fully instead of classes
            x.append(dataset[i][0])
            y.append(dataset[i][1])

    tensor_x = torch.Tensor(x)
    tensor_y = torch.Tensor(y)

    # train_split = int(split * len(x))
    # val_split = len(x) - train_split
    train_split = int(split * len(x))
    val_split = len(x) - train_split
    tensor_dataset = TensorDataset(tensor_x, tensor_y)
    train_dataset, val_dataset = random_split(tensor_dataset, [train_split, val_split])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader

class LinearAE(nn.Module):
    def __init__(self, in_size=16, latent_size=64):
        super(LinearAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.ReLU(),
            nn.Linear(64,latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, in_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

def get_min_and_max(dataloader, device) -> Tuple[float, float]:
    """
    Finds the min and max of the dataset.
    This is used for Normalization of the dataset.

    :param dataloader: dataloader to calculate it for
    :param device: device to run computations on
    :return: tuple of mean and std
    """
    min_val, max_val = torch.Tensor([999]).to(device), torch.Tensor([-999]).to(device)
    for data, _ in tqdm(dataloader):
        data = data.to(device)
        min_val = torch.min(min_val, torch.min(data))
        max_val = torch.max(max_val, torch.max(data))

    return min_val.item(), max_val.item()

set_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LinearAE(16, 256).cuda()
train_loader, val_loader = prepare_dataloader()
min_val, max_val = get_min_and_max(train_loader, device)

optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

for epoch in range(50):
  train_loss = 0
  for batch_idx, data in enumerate(train_loader):
      vector, _ = data
      vector = vector.to(device)
      vector = (vector - min_val) / (max_val - min_val)

      optimizer.zero_grad()
      y = model(vector)
      # recon_batch, mu, logvar = model(x)
      loss = F.mse_loss(y, vector)
      # loss = loss_function(recon_batch, x, mu, logvar)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()

  scheduler.step(train_loss/len(train_loader))
  print("Epoch: {:03d} || Train Loss: {:.20f} ".format(epoch, train_loss / len(train_loader)))


model_dir = 'models'
model_filename = 'domainA-encoder-complete.pt'
model_filepath = os.path.join(model_dir, model_filename)
torch.save(model.state_dict(), model_filepath)