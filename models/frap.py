from torch import nn
import torch.nn.functional as F
import torch

from models.neural_net import SerializableModel

NUM_PHASES = 8
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# from what traffic movements phases are made of
phases_movements = {
    0: (10, 4),
    1: (5, 4),
    2: (10, 11),
    3: (5, 11),
    4: (1, 7),
    5: (8, 7),
    6: (1, 2),
    7: (8, 2)
}

# 0 - light gray -> phases share one traffic movement [partial competing]
# 1 - gray -> competing, phases are totally conflict [competing]
competing_matrix = torch.tensor([
    [0, 0, 1, 1, 1, 1, 1],
    [0, 1, 0, 1, 1, 1, 1],
    [0, 1, 0, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 1],
    [1, 1, 1, 1, 0, 1, 0],
    [1, 1, 1, 1, 0, 1, 0],
    [1, 1, 1, 1, 1, 0, 0],
], device=device)

worth_movements = {1, 2, 4, 5, 7, 8, 10, 11}


class Frap(SerializableModel):

    def _get_init_params(self):
        return {
            'relation_embedding_size': self.relation_embedding_size,
            'demand_vec_size': self.demand_vec_size,
            'demand_hidden': self.demand_hidden,
            'num_conv_layers': self.num_conv_layers,
            'conv_channels_size': self.conv_channels_size,
            'output_mean': self.output_mean
        }

    def _create_conv(self, start_channels, num_layers, conv_channels_size):
        relation_conv = []

        for _ in range(num_layers):
            relation_conv.append(nn.Conv2d(start_channels, conv_channels_size, kernel_size=1, stride=1))
            relation_conv.append(nn.ReLU())

            start_channels = conv_channels_size

        return nn.Sequential(*relation_conv)

    def __init__(self, relation_embedding_size, demand_vec_size,
                 demand_hidden, num_conv_layers, conv_channels_size, output_mean=True):
        super().__init__()
        self.output_mean = output_mean
        self.conv_channels_size = conv_channels_size
        self.num_conv_layers = num_conv_layers
        self.demand_hidden = demand_hidden
        self.demand_vec_size = demand_vec_size
        self.relation_embedding_size = relation_embedding_size

        self.phase_v = nn.Linear(1, demand_hidden, bias=True)
        self.phase_s = nn.Linear(1, demand_hidden, bias=True)
        self.phase_d = nn.Linear(demand_hidden * 2, demand_vec_size, bias=True)

        self.rel_embedding = nn.Embedding(3, relation_embedding_size)

        self.relation_conv = self._create_conv(relation_embedding_size, num_conv_layers, conv_channels_size)
        self.demand_conv = self._create_conv(demand_vec_size * 2, num_conv_layers, conv_channels_size)

        self.last_conv = nn.Conv2d(conv_channels_size, 1, 1)

    def _calc_deamand(self, curr_phases, pressure):
        h_v = F.relu(self.phase_v(pressure))
        h_s = F.relu(self.phase_s(curr_phases))
        return F.relu(self.phase_d(torch.cat([h_v, h_s], dim=1)))

    def forward(self, pressures):
        curr_phases = pressures[:, 0].unsqueeze(-1)
        phases_pressures = pressures[:, 1:]
        saved_demands = dict()
        for m in worth_movements:
            my_d = self._calc_deamand(curr_phases, phases_pressures[:, m].unsqueeze(-1))
            saved_demands[m] = my_d
        phase_demands = []
        for i in range(NUM_PHASES):
            m_a, m_b = phases_movements[i]
            phase_demand = saved_demands[m_a] + saved_demands[m_b]
            phase_demands.append(phase_demand)

        demand_embedding = torch.cat([
            torch.cat([
                torch.cat([phase_demands[p], phase_demands[q]], dim=1).unsqueeze(1).unsqueeze(1)
                for q in range(NUM_PHASES) if q != p
            ], dim=2)
            for p in range(NUM_PHASES)
        ], dim=1)

        relation_embedding = self.rel_embedding(competing_matrix).unsqueeze(0)

        relation_conv_out = self.relation_conv(relation_embedding.permute(0, 3, 1, 2).contiguous())
        demand_conv_out = self.demand_conv(demand_embedding.permute(0, 3, 1, 2).contiguous())

        phase_competition = relation_conv_out * demand_conv_out

        phase_competition = self.last_conv(phase_competition).squeeze(1)

        if self.output_mean:
            return phase_competition.mean(dim=2)
        else:
            return phase_competition.sum(dim=2)