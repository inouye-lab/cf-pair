import torch.nn as nn
from torchvision.models import resnet18, resnet50, resnet101


class CNN(nn.Module):
    def __init__(self, input_shape, n_outputs=1024, probabilistic=False):
        super(CNN,self).__init__()
        self.n_outputs = n_outputs
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


class Classifier(nn.Module):
    def __init__(self, in_features, out_features, is_nonlinear=False):
        super(Classifier, self).__init__()
        if is_nonlinear:
            # self.model = SimpleTransformer(in_features=in_features, seq_len=16, model_dim=in_features//16, num_heads=2, num_layers=1, num_classes=out_features)    
            self.model = nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.Dropout(p=0.5),
                nn.ReLU(),
                nn.Linear(in_features // 2, in_features // 4),
                nn.Dropout(p=0.5),
                nn.ReLU(),
                nn.Linear(in_features // 4, in_features // 8),
                nn.Dropout(p=0.5),
                nn.ReLU(),
                nn.Linear(in_features // 8, out_features))
                # nn.Linear(in_features // 8, in_features // 16),
                # nn.Dropout(p=0.5),
                # nn.ReLU(),
                # nn.Linear(in_features // 16, out_features))
        else:
            self.model = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.model(x)


class ResNet18(nn.Module):
    def __init__(self, input_shape, n_outputs=1024, probabilistic=False):
        super(ResNet18, self).__init__()
        self.model = resnet18(num_classes=n_outputs, pretrained=False)
        self.model.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    def forward(self, x):
        return self.model(x)
    

class ResNet50(nn.Module):
    def __init__(self, input_shape, n_outputs=1024, probabilistic=False):
        super(ResNet50, self).__init__()
        self.model = resnet50(num_classes=n_outputs, pretrained=False)
        self.model.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    def forward(self, x):
        return self.model(x)


class ResNet101(nn.Module):
    def __init__(self, input_shape, n_outputs=1024, probabilistic=False):
        super(ResNet101, self).__init__()
        self.model = resnet101(num_classes=n_outputs, pretrained=False)
        self.model.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    def forward(self, x):
        return self.model(x)