import os
import random
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
from torch.utils.data.dataset import TensorDataset, random_split
from torch.utils.data.dataloader import DataLoader
from dnn import DNN


def set_random_seeds(random_seed=1000):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def prepare_dataloader(
    batch_size: int = 32,
    num_samples: int = 100,
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


def evaluate_model(model, test_loader, device, criterion=None, batch_size = 32):

    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0
    pred = []
    true = []
    for (inputs, labels) in tqdm(test_loader):

        inputs = inputs.to(device)       
        labels = labels.reshape(batch_size)
        labels = labels.type(torch.LongTensor)
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
                test_loader,
                device,
                learning_rate=1e-4,
                num_epochs=200,
                batch_size = 32):

    writer = SummaryWriter(log_dir = "/content/drive/MyDrive/Latent Transfer/Validation model/events")

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
                                              criterion=criterion,
                                              batch_size = batch_size)

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
            labels = labels.reshape(batch_size)
            labels = labels.type(torch.LongTensor)
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
                                                  test_loader=test_loader,
                                                  device=device,
                                                  criterion=criterion)

        network_learned = ((100 * eval_accuracy) > test_acc_min)

        if network_learned:
            test_acc_min = (100 * eval_accuracy)
            preds = []
            true = []
            true.append(true)
            preds.append(preds)
            torch.save(model.state_dict(), '/content/drive/MyDrive/Latent Transfer/Validation model/models/dnn.pt')
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

    model = DNN()
    return model



def main():

    random_seed = 0
    num_classes = 7
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    model_dir = "/content/drive/MyDrive/Latent Transfer/Validation model/models"
    model_filename = "dnn.pt"
    quantized_model_filename = "dnn.pt"
    model_filepath = os.path.join(model_dir, model_filename)
    quantized_model_filepath = os.path.join(model_dir,
                                            quantized_model_filename)

    set_random_seeds(random_seed=random_seed)

    # Create an untrained model.
    model = create_model()
    batch_size = 32

    train_loader, test_loader = prepare_dataloader(
    batch_size = batch_size,
    num_samples = 1000,
    spread = 0.1,
    split = 0.8,
    seed = 1234,
)

    # Train model.
    
    train_model(model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=cuda_device,
                learning_rate=1e-4,
                num_epochs=20,
                batch_size = batch_size)



if __name__ == "__main__":

    main()
