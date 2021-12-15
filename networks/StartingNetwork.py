import torch
import torch.nn as nn


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()
        """
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(224 * 224 * 3, 1)
        self.sigmoid = nn.Sigmoid() # First three lines are their code (remove)
        """

        self.c1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2.5, bias=True)
        self.c2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2.5, bias=True)
        self.m1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2.5, bias=True)
        self.m2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.f1 = nn.Linear(in_features=1024, out_features=512)
        self.f2 = nn.Linear(in_features=512, out_features=256)
        self.f3 = nn.Linear(in_features=256, out_features=64)
        self.f4 = nn.Linear(in_feautres=64, out_features=10)

        self.s1 = nn.Softmax(dim=5) # Four diseases + 1 healthy 


    def forward(self, x):
        """
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x) # First three lines are their code (remove)
        """

        x = self.c1(x) 
        x = self.c2(x) 
        x = self.m1(x) 
        x = self.c3(x) 
        x = self.m2(x) 
        return x

    # Conv layer, relu, conv layer, relu, maxpool, conv layer, relu, maxpool

    # 1st Conv Layer: 1 -> 16 (k:5, p:2.5, bias=True)
    # 2nd Conv Layer: 16 -> 32 (k:5, p:2.5, bias=True)
    # 1st Maxpool: (k:2, stride:2)
    # 3rd Conv Layer: 32 -> 64 (k:5, p:2.5, bias=True)
    # 2nd Maxpool: (k:2, stride:2)

    # Other optimizations: 
    # Dataset modifications: expand (blur, rotate) so that roughly # of images of each category is same 

    # Things to keep in mind: 
    # * Keep track of point where error starts increasing again
    # * Gradients don't dissappear 