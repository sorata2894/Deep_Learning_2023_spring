import numpy as np
import torch
from torch import nn
from torchinfo import summary

class DeepConvNet(nn.Module):
    def __init__(self, class_num, act_choose="ELU"):
        '''
            class_num : output total class
            act_choose : can input "ELU", "ReLU", "LeakyReLU" to control using which activation function
        '''
        super(DeepConvNet, self).__init__()
        self.class_num = class_num
        self.activation_func = {"ELU": nn.ELU(alpha=1.0), "ReLU": nn.ReLU(), "LeakyReLU": nn.LeakyReLU()}
        self.model = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1,5),bias=False),
            nn.Conv2d(25, 25, kernel_size=(2,1),bias=False),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            self.activation_func[act_choose],
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.4),

            nn.Conv2d(25, 50, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            self.activation_func[act_choose],
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.4),

            nn.Conv2d(50, 100, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            self.activation_func[act_choose], 
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.4),

            nn.Conv2d(100, 200, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            self.activation_func[act_choose],
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.4),

            nn.Flatten(),
            nn.Linear(8600, self.class_num, bias=True)
        )
    def forward(self, input):
        out = self.model(input)
        return out


class EEGNet(nn.Module):
    def __init__(self, class_num, act_choose="ELU"):
        '''
            class_num : output total class
            act_choose : can input "ELU", "ReLU", "LeakyReLU" to control using which activation function
        '''
        super(EEGNet, self).__init__()
        self.activation_func = {"ELU": nn.ELU(alpha=1.0), "ReLU": nn.ReLU(), "LeakyReLU": nn.LeakyReLU()}
        self.firstconv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.activation_func[act_choose],
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(0.35)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.activation_func[act_choose],
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(0.35)
        )
        self.classify = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(in_features=736, out_features=class_num, bias=True)
        )

    def forward(self, input):
        x = self.firstconv(input)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        out = self.classify(x.view(x.size(0), -1))
        return out


if __name__ == '__main__':
    device = torch.device("cuda", 0)
    # model = EEGNet() 
    model = DeepConvNet(class_num=2, act_choose="ELU")
    model.to(device)

    summary(model, (64, 1, 2, 750))
    pass