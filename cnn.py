import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision import transforms, utils, datasets
from util_conv import ConvNet, DeviceDataLoader, EarlyStopping
import pandas as pd

def cnn():
    torch.manual_seed(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset, val_dataset, test_dataset = create_datasets()
    train_loader = DeviceDataLoader(train_dataset, device)
    val_loader = DeviceDataLoader(val_dataset, device)
    
    model = ConvNet()
    model.to(device)
    train_val_history = fit(model, 100, train_loader, val_loader)

    test_loader = DeviceDataLoader(test_dataset, device)
    test_outputs, test_labels, test_result = evaluate(model, test_loader, True)

    return train_val_history, test_outputs, test_labels, test_result

@torch.no_grad()
def evaluate(model, loader, test = False):
    model.eval()

    if test:
        out, labels, step = None, None, []
        for batch in loader:
            out, labels, s = model.validation_step(batch, test)
            step.append(s)

        _, preds = torch.max(out, dim=1)
        test_outputs = {'ground truth': labels, 'predicted': preds}
        return test_outputs, model.validation_epoch_end(step)

    outputs = [model.validation_step(batch, test) for batch in loader]
    return model.validation_epoch_end(outputs)

def fit(model, epochs, train_dataset, val_dataset, lr = 0.001, optim = optim.Adam, early = True):
    early_stopping = EarlyStopping()
    history = []
    optimizer = optim(model.parameters(), lr)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_accs = []
        for batch in train_dataset:
            loss, acc = model.training_step(batch)
            train_losses.append(loss)
            train_accs.append(acc)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_dataset)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['train_acc'] = torch.stack(train_accs).mean().item()
        
        # Early stopping
        if early:
            early_stopping(result['val_loss'])
            if early_stopping.early_stop:
                return history

        model.epoch_end(epoch, result)
        history.append(result)
    
    return history

# Dataset for the pytorch cnn to read
def create_datasets(batch_size = 128):
    # Split dataset 78/12/10
    train_count = int(0.78 * 1560)
    val_count = int(0.12 * 1560)
    test_count = 1560 - train_count - val_count
    transform = transforms.ToTensor()

    dataset = datasets.ImageFolder(root = "./RoCoLe/Resized Photos/", transform = transform)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_count, val_count, test_count))
    
    train_load = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, pin_memory = True)
    val_load = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = True, pin_memory = True)
    test_load = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size * 2, shuffle = False)
    
    return train_load, val_load, test_load

