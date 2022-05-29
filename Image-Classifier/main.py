import os
import random
# Testing sequence 1
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
import time
import copy
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from resnet import resnet50


def set_random_seeds(random_seed=1000):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

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

        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28))
        label = self.data.iloc[index, 0]
        if (label > 9):
          label = label - 1
          
        rgb = np.repeat(image[..., np.newaxis], 3, -1)
        if self.transform is not None:
    
            rgb = self.transform(rgb)
        return rgb, label



def prepare_dataloader(num_workers=8,
                       batch_size=1,
                      ):

    transform_train = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize((224)),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          # transforms.Normalize(mean=(0.4346, 0.2213, 0.0783),
                                          #                      std=(0.2439, 0.1311, 0.0703))            
                                        ])
    

    transform_test = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize((224)),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         #transforms.Normalize(mean=(0.4346, 0.2213, 0.0783),
                                         #                        std=(0.2439, 0.1311, 0.0703))
                                        ])
    
    train_data = DatasetMNIST('Image-Classifier/DomainB_dataset/sign_mnist_train.csv', transform = transform_train)
    test_data = DatasetMNIST('Image-Classifier/DomainB_dataset/sign_mnist_test.csv', transform = transform_test)
    
    valid_size = int(0.5 * len(test_data))
    test_size = len(test_data) - valid_size
    validation_dataset, test_dataset = torch.utils.data.random_split(test_data, [valid_size, test_size])
    
    train_loader = DataLoader(
                              train_data,
                              batch_size=batch_size,
                              shuffle=True
                             )
    valid_loader = test_loader = DataLoader(
                              validation_dataset,
                              batch_size=batch_size,
                              shuffle=False
                             )
    test_loader = DataLoader(
                              test_dataset,
                              batch_size=batch_size,
                              shuffle=False
                             )

    return train_loader, valid_loader, test_loader


def evaluate_model(model, test_loader, device, criterion=None):

    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0
    pred = []
    true = []
    for (inputs, labels) in tqdm(test_loader):

        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        pred.append(preds)
        true.append(labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy, pred, true


def train_model(model,
                train_loader,
                valid_loader,
                test_loader,
                device,
                learning_rate=1e-4,
                num_epochs=200):

    writer = SummaryWriter(log_dir = "Image-Classifier/events")

    criterion = nn.CrossEntropyLoss()

    model.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=0.9,
                          weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # Evaluation
    model.eval()
    eval_loss, eval_accuracy, _, _ = evaluate_model(model=model,
                                              test_loader=test_loader,
                                              device=device,
                                              criterion=criterion)

    print('Training Started:')
    test_acc_min = 0
    for epoch in range(num_epochs):

        # Training
        model.train()
        
        running_loss = 0
        running_corrects = 0
        
        print('\n')

        
        for (inputs, labels) in tqdm(train_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

  
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)
        
        
        # Evaluation
        model.eval()
        eval_loss, eval_accuracy, preds, true = evaluate_model(model=model,
                                                  test_loader=valid_loader,
                                                  device=device,
                                                  criterion=criterion)

        network_learned = ((100 * eval_accuracy) > test_acc_min)

        if network_learned:
            test_acc_min = (100 * eval_accuracy)
            preds = []
            true = []
            true.append(true)
            preds.append(preds)
            torch.save(model.state_dict(), 'Image-Classifier/models/mnist-resnet50.pt')
            print('Improvement-Detected, save-model')

        writer.add_scalar("Train Loss", train_loss, epoch)
        writer.add_scalar("Train Acc", 100.0 * train_accuracy, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("Test Loss", eval_loss, epoch)
        writer.add_scalar("Test Acc", 100.0 * eval_accuracy, epoch)
        scheduler.step()

        print(
            "Epoch: {:03d} || Train Loss: {:.3f} || Train Acc: {:.3f} || Eval Loss: {:.3f} || Eval Acc: {:.3f}"
            .format(epoch, train_loss, train_accuracy * 100, eval_loss,
                    eval_accuracy * 100))
    print('\nEvaluating on Test Set:')
    test_loss, test_accuracy, preds, true = evaluate_model(model=model,
                                                  test_loader=test_loader,
                                                  device=device,
                                                 criterion=criterion)
    print("\nTest Loss: {:.3f} || Test Acc: {:.3f}"
            .format(test_loss,test_accuracy * 100))

    writer.close()




def save_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)


def load_model(model, model_filepath, device):

    model.load_state_dict(torch.load(model_filepath, map_location=device))

    return model



def create_model():

    model = resnet50(num_classes=1000, pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 24)


    return model



def main():

    random_seed = 0
    num_classes = 24
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    model_dir = "Image-Classifier/models"
    model_filename = "mnist-resnet50.pt"
    model_filepath = os.path.join(model_dir, model_filename)


    set_random_seeds(random_seed=random_seed)

    # Create an untrained model.
    model = create_model()

    train_loader, valid_loader, test_loader = prepare_dataloader(num_workers=8,
                                                   batch_size=32,
                                                   )

    # Train model.
    
    train_model(model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                test_loader=test_loader,
                device=cuda_device,
                learning_rate=1e-4,
                num_epochs=5)



if __name__ == "__main__":

    main()

