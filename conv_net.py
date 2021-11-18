from img_class_base import ImageClassificationBase
import torch.nn as nn
import torch.nn.functional as F

'''
Convolution Network - architecture is based on https://cs231n.github.io/convolutional-networks/ where we do layers
of conv -> reLU -> conv -> reLU -> pool before condensing it down to a FC layer.
'''
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
            nn.Conv2d(128, 256, kernel_size = 3, padding = 1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(4,4),
            
            nn.Flatten(),
            nn.Linear(20736, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        return self.network(x)