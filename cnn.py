import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision import transforms, utils, datasets

def cnn():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    create_datasets()

def create_datasets(batch_size = 256):
    # Percent of training data used for validation
    train_count = int(0.8 * 1560)
    val_count = int(0.1 * 1560)
    test_count = 1560 - train_count - val_count
    transform = transforms.ToTensor()

    dataset = datasets.ImageFolder(root = "./RoCoLe/Resized Photos/", transform = transform)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_count, val_count, test_count))
    
    train_load = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_load = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    test_load = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    data_load = {"train": train_load, "val": val_load, "test": test_load}

