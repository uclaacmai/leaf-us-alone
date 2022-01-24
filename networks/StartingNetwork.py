import torch
import torch.nn as nn


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=11, stride=4, padding=1, bias=True)
        model = torch.hub.load('pytorch/vision:v0.8.0', 'resnet18', pretrained=True)
        self.resnet = torch.nn.Sequential(*(list(model.children())[:-1]))

        self.f1 = nn.Linear(512, 512)
        self.f2 = nn.Linear(512, 5)

        # Should increase the kernel_size
        """
        self.c1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=1, bias=True)
        self.m1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2, bias=True)
        self.m2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1, bias=True)
        self.c4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, bias=True)
        self.c5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1, bias=True)
        self.m3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.f1 = nn.Linear(in_features=9216, out_features=4096) # Changed to 256 * 6 * 6 from 6400
        self.f2 = nn.Linear(in_features=4096, out_features=4096)
        self.f3 = nn.Linear(in_features=4096, out_features=5)
        # self.f4 = nn.Linear(in_features=64, out_features=5)

        # self.s1 = nn.Softmax(dim=5) # Four diseases + 1 healthy
        """

    def forward(self, x):
        x = self.c1(x)
        nn.ReLU()
        x = self.resnet(x)
        nn.ReLU()

        x = torch.flatten(x, 1)
        x = self.f1(x)
        nn.ReLU()

        nn.Dropout(p=0.5, inplace=True)
        x = self.f2(x)
        """
        x = self.c1(x) # [b, 96, 55, 55]
        nn.ReLU()

        x = self.m1(x) # [b, 96, 27, 27]

        x = self.c2(x) # [b, 256, 27, 27]
        nn.ReLU()

        x = self.m2(x) # [b, 256, 13, 13]

        x = self.c3(x) # [b, 384, 13, 13]
        nn.ReLU()

        x = self.c4(x) # [b, 384, 13, 13]
        nn.ReLU()

        x = self.c5(x) # [b, 256, 13, 13]
        nn.ReLU()

        x = self.m3(x) # [b, 256, 6, 6]

        x = torch.flatten(x, 1) # [50, 9216]

        x = self.f1(x)
        nn.ReLU()

        nn.Dropout(p=0.5, inplace=True)
        self.f2 = nn.Linear(in_features=4096, out_features=4096)
        nn.ReLU()

        nn.Dropout(p=0.5, inplace=True)
        self.f3 = nn.Linear(in_features=4096, out_features=5)
        """
        return x

    # Conv layer, relu, conv layer, relu, maxpool, conv layer, relu, maxpool

    # Reshape ouput size; #D ouput because of cov layers. We want our output to be one-d by the end.

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
