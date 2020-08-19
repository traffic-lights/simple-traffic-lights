from torch import nn


class SerializableModel(nn.Module):
    def get_save_dict(self):
        return {
            'init_params': self._get_init_params(),
            'state_dict': self.state_dict()
        }

    def _get_init_params(self):
        raise NotImplementedError

    @classmethod
    def load_from_dict(cls, dict_to_load):
        model = cls(**dict_to_load['init_params'])
        model.load_state_dict(dict_to_load['state_dict'])
        return model


class DQN(SerializableModel):

    def _get_init_params(self):
        return {'outputs': self.outputs}

    # @classmethod
    # def load_from_dict(cls, dict_to_load):
    #     dqn = DQN(**dict_to_load['init_params'])
    #     dqn.load_state_dict(dict_to_load['state_dict'])
    #     return dqn

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


class SimpleLinear(SerializableModel):

    def _get_init_params(self):
        return {
            'input_params': self.input_params,
            'output_params': self.output_params,
            'linear_depth': self.linear_depth
        }

    def __init__(self, input_params, output_params, linear_depth=90):
        super().__init__()
        self.input_params = input_params
        self.output_params = output_params
        self.linear_depth = linear_depth

        self.linear = nn.Linear(input_params, linear_depth)

        self.value_net = nn.Sequential(
            nn.Linear(linear_depth, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

        self.advantage_net = nn.Sequential(
            nn.Linear(linear_depth, 32),
            nn.LeakyReLU(),
            nn.Linear(32, output_params)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.linear(x)

        value = self.value_net(x)
        advantages = self.advantage_net(x)

        qvals = value + (advantages - advantages.mean())

        return qvals


class Frap(SerializableModel):

    def _get_init_params(self):
        pass

    def __init__(self):
        super().__init__()
