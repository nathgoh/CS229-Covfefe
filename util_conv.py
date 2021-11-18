import torch
import torch.nn as nn
import torch.nn.functional as F

# The custom nn.Module for image classification
class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch 

        # Generate predictions
        out = self(images)

        # Calculate loss          
        loss = F.cross_entropy(out, labels) 

        # Calculate accuracy
        acc = accuracy(out, labels)      

        return loss, acc
    
    def validation_step(self, batch):
        images, labels = batch 

        # Generate predictions
        out = self(images)

        # Calculate loss                   
        loss = F.cross_entropy(out, labels)   
        
        # Calculate accuracy
        acc = accuracy(out, labels)   

        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]

        # Combine losses
        epoch_loss = torch.stack(batch_losses).mean()   
        batch_accs = [x['val_acc'] for x in outputs]

        # Combine accuracies
        epoch_acc = torch.stack(batch_accs).mean()      
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch {}: train_acc: {:.4f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch + 1, result['train_acc'], result['train_loss'], result['val_loss'], result['val_acc']))

# Convolution Network - architecture is based on https://cs231n.github.io/convolutional-networks/ where we do layers
# of conv -> reLU -> conv -> reLU -> pool before condensing it down to a FC layer.
class ConvNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(), 
            nn.Conv2d(32, 64, kernel_size = 3, padding = 1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.ReLU(), 
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(65536, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 6)
        )

    def forward(self, x):
        return self.network(x)

# Early stopping to stop the training when the loss does not improve after certain epochs.
# patience: # of epochs before stopping when loss isn't improving
# min_delta: min difference between new loss and old loss for the new loss to be considered an improvement
class EarlyStopping():
    
    def __init__(self, patience = 5, min_delta = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss

            # Reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print("INFO: Early stopping {}/{}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                print('INFO: Early stopping complete')
                self.early_stop = True


# Wrap a dataloader to move data to a device 
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

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))