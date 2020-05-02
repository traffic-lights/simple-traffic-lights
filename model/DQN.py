import torch
import torch.nn as nn

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1),
            nn.MaxPool2d(3, 1, 1),
            nn.LeakyReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 2, 2),
            nn.MaxPool2d(3, 1, 1),
            nn.LeakyReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 2, 1),
            nn.MaxPool2d(3, 1, 1),
            nn.LeakyReLU()
        )

        self.value_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

        self.advantage_net = nn.Sequential(
             nn.Linear(128, 64),
             nn.LeakyReLU(),
             nn.Linear(64, 9)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(-1, 128)

        value = self.value_net(x)
        advantages = self.advantage_net(x)

        qvals = value + (advantages - advantages.mean())

        return qvals