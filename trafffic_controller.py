class TrafficController:
    def __init__(self, controller, controller_output_to_action_map_f=lambda x: x):
        self.controller = controller
        self.f = controller_output_to_action_map_f

    def __call__(self, state):
        return self.f(self.controller(state))
