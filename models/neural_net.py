from torch import nn

from models.utils import SerializableModel


class DQN(SerializableModel):

    def get_save_dict(self):
        return {
            'init_params': {'outputs': self.outputs},
            'state_dict': self.state_dict()
        }

    @classmethod
    def load_from_dict(cls, dict_to_load):
        dqn = DQN(**dict_to_load['init_params'])
        dqn.load_state_dict(dict_to_load['state_dict'])
        return dqn

    def __init__(self, outputs=9):
        super(DQN, self).__init__()

        self.outputs = outputs

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

        self.linear = nn.Linear(9 * 9 * 128, 128)

        self.value_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

        self.advantage_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.outputs)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(batch_size, -1)
        x = self.linear(x)

        value = self.value_net(x)
        advantages = self.advantage_net(x)

        qvals = value + (advantages - advantages.mean())

        return qvals


