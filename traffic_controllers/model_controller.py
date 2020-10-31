import torch

from traffic_controllers.trafffic_controller import TrafficController


class ModelController(TrafficController):
    def __init__(self, model, device=torch.device('cpu')):
        self.model = model
        self.device = device

    def __call__(self, inputs):
        inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        if len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(0)

        return self.model(inputs).max(1)[1][0].cpu().detach().numpy().item()
