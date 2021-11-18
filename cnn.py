import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision import transforms, utils, datasets
from conv_net import ConvNet

import pandas as pd

#Wrap a dataloader to move data to a device 
class DeviceDataLoader():    
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        # Yield a batch of data after moving it to device
        for b in self.dl:
            yield to_device(b,self.device)
            
    def __len__(self):
        # Number of batches
        return len(self.dl)

# Move data to the device   
def to_device(data, device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]

    return data.to(device, non_blocking = True)

def cnn():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset, val_dataset, test_dataset = create_datasets()
    train_dataset = DeviceDataLoader(train_dataset, device)
    val_dataset = DeviceDataLoader(val_dataset, device)
    
    model = ConvNet()
    model.to(device)
    history = fit(model, 30, train_dataset, val_dataset)


def fit(model, epochs, train_dataset, val_dataset, lr = 0.001, optim = optim.SGD):
    history = []
    optimizer = optim(model.parameters(), lr, momentum = 0.9)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_dataset:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    
    return history

# Dataset for the pytorch cnn to read
def create_datasets(batch_size = 128):
    # Split dataset 85/5/10
    train_count = int(0.85 * 1560)
    val_count = int(0.05 * 1560)
    test_count = 1560 - train_count - val_count
    transform = transforms.ToTensor()

    dataset = datasets.ImageFolder(root = "./RoCoLe/Resized Photos/", transform = transform)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_count, val_count, test_count))
    
    train_load = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, pin_memory = True)
    val_load = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = True, pin_memory = True)
    test_load = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    
    return train_load, val_load, test_load

