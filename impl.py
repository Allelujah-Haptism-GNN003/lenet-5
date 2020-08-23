import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, ):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1)
        self.pool1 = nn.AvgPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1)
        self.pool2 = nn.AvgPool2d(2, stride=2)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.pool1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.pool2(x))
        x = x.view(x.size()[0], -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        
        output = F.softmax(x, dim=1)

        return output