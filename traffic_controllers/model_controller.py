import torch


class ModelController:
    def __init__(self, model):
        self.model = model

    def __call__(self, inputs):
        inputs = torch.tensor(inputs, dtype=torch.float32)
        if len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(0)

        return self.model(inputs).max(1)[1][0].cpu().detach().numpy().item()
