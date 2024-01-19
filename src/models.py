import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, input_shape, probabilistic=False):
        super(CNN,self).__init__()
        self.n_outputs = 2048
        self.probabilistic = probabilistic
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0],out_channels=16,kernel_size=5,padding=2),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),  # kernel_size, stride
            nn.Conv2d(in_channels=16,out_channels=64,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        if self.probabilistic:
            self.fc = nn.Linear(in_features=7*7*64,out_features=self.n_outputs * 2)
        else:
            self.fc = nn.Linear(in_features=7*7*64,out_features=self.n_outputs)
    def forward(self,x):
        feature=self.fc(self.conv(x).view(x.shape[0], -1))
        return feature