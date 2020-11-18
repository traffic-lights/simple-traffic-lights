from torch import nn
import torch.nn.functional as F
import torch

from models.neural_net import SerializableModel

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

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

    def __init__(self, relation_embedding_size=32,
                 demand_vec_size=16,
                 demand_hidden=16,
                 num_conv_layers=2,
                 conv_channels_size=16,
                 output_mean=True,
                 num_junctions=1,
                 traffic_lights_movements=12
                 ):
        super().__init__()
        self.num_junctions = num_junctions
        self.traffic_lights_movements = traffic_lights_movements
        self.output_mean = output_mean
        self.conv_channels_size = conv_channels_size
        self.num_conv_layers = num_conv_layers
        self.demand_hidden = demand_hidden
        self.demand_vec_size = demand_vec_size
        self.relation_embedding_size = relation_embedding_size

        self.phase_v = nn.Linear(1, demand_hidden, bias=True)
        self.phase_s = nn.Linear(1, demand_hidden, bias=True)
        self.phase_d = nn.Linear(demand_hidden * 2, demand_vec_size, bias=True)

        # 0 - light gray -> phases share one traffic movement [partial competing]
        # 1 - gray -> competing, phases are totally conflict [competing]
        self.competing_matrix = nn.Parameter(
            torch.tensor([
                [0, 0, 1, 1, 1, 1, 1],
                [0, 1, 0, 1, 1, 1, 1],
                [0, 1, 0, 1, 1, 1, 1],
                [1, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 1],
                [1, 1, 1, 1, 0, 1, 0],
                [1, 1, 1, 1, 0, 1, 0],
                [1, 1, 1, 1, 1, 0, 0],
            ]), False)

        self.rel_embedding = nn.Embedding(3, relation_embedding_size)

        self.relation_conv = self._create_conv(relation_embedding_size, num_conv_layers, conv_channels_size)
        self.demand_conv = self._create_conv(demand_vec_size * 2, num_conv_layers, conv_channels_size)

        self.last_conv = nn.Conv2d(conv_channels_size, 1, 1)

        # CONSTANTS
        self.worth_movements = [1, 2, 4, 5, 7, 8, 10, 11]

        self.num_phases = 8

        # from what traffic movements phases are made of
        self.phases_movements = {
            0: (10, 4),
            1: (5, 4),
            2: (10, 11),
            3: (5, 11),
            4: (1, 7),
            5: (8, 7),
            6: (1, 2),
            7: (8, 2)
        }

    def _calc_deamand(self, curr_phases, pressure):
        h_v = F.relu(self.phase_v(pressure))
        h_s = F.relu(self.phase_s(curr_phases))
        return F.relu(self.phase_d(torch.cat([h_v, h_s], dim=1)))

    def forward(self, pressures, prev_action=None, prev_reward=None):
        lead_dim, T, B, pressures_shape = infer_leading_dims(pressures, 1)
        #print("TOMSIA press before: ", pressures.shape, pressures)
        pressures = pressures.view(T * B * self.num_junctions, -1)
        #print("TOMSIA press after: ", pressures.shape, pressures)
        curr_phases = pressures[:, 0].unsqueeze(-1)
        phases_pressures = pressures[:, 1:]
        saved_demands = dict()
        for m in self.worth_movements:
            my_d = self._calc_deamand(curr_phases, phases_pressures[:, m].unsqueeze(-1))
            saved_demands[str(m)] = my_d
        phase_demands = []
        for i in range(self.num_phases):
            m_a, m_b = self.phases_movements[i]
            phase_demand = saved_demands[str(m_a)] + saved_demands[str(m_b)]
            phase_demands.append(phase_demand)

        tmp = []
        for p in range(self.num_phases):
            tmp2 = []
            for q in range(self.num_phases):
                if q != p:
                    tmp2.append(torch.cat([phase_demands[p], phase_demands[q]], dim=1).unsqueeze(1).unsqueeze(1))

            tmp.append(torch.cat(tmp2, dim=2))

        demand_embedding = torch.cat(tmp, dim=1)
        relation_embedding = self.rel_embedding(self.competing_matrix).unsqueeze(0)

        relation_conv_out = self.relation_conv(relation_embedding.permute(0, 3, 1, 2).contiguous())
        demand_conv_out = self.demand_conv(demand_embedding.permute(0, 3, 1, 2).contiguous())

        phase_competition = relation_conv_out * demand_conv_out

        phase_competition = self.last_conv(phase_competition).squeeze(1)

        if self.output_mean:
            phase_competition = phase_competition.mean(dim=2)
        else:
            phase_competition = phase_competition.sum(dim=2)

        #print('TOMSIA PHASE COMP before:', phase_competition.shape, phase_competition)
        phase_competition = phase_competition.reshape(T * B, self.num_junctions, -1)
        #print('TOMSIA PHASE COMP after:', phase_competition.shape, phase_competition)

        return restore_leading_dims(phase_competition, lead_dim, T, B)
