from settings import init_sumo_tools

init_sumo_tools()
import traci


class CyclicSwitchControllerI:
    def __init__(self, actions):
        self.iter = 0
        self.actions = actions

    def _check_is_wait_period_over_and_update(self):
        raise NotImplementedError

    def _reset_period(self):
        raise NotImplementedError

    def __call__(self, state):
        if self._check_is_wait_period_over_and_update():
            self.iter = (self.iter + 1) % len(self.actions)
            self._reset_period()

        return self.actions[self.iter]


class TimedCyclicSwitchController(CyclicSwitchControllerI):
    def __init__(self, actions, switch_periods_in_s):
        super(TimedCyclicSwitchController, self).__init__(actions)
        self.switch_periods_in_s = switch_periods_in_s[-1:] + switch_periods_in_s[:-1]
        self.last_switch_time = 0

    def _check_is_wait_period_over_and_update(self):
        return traci.simulation.getTime() - self.last_switch_time >= self.switch_periods_in_s[self.iter]

    def _reset_period(self):
        self.last_switch_time = traci.simulation.getTime()


class CallCounterCyclicSwitchController(CyclicSwitchControllerI):
    def __init__(self, actions, nums_calls_before_reset):
        super(CallCounterCyclicSwitchController, self).__init__(actions)
        self.nums_calls_before_reset = nums_calls_before_reset
        self.cnt = -1

    def _check_is_wait_period_over_and_update(self):
        self.cnt += 1
        return self.cnt >= self.nums_calls_before_reset[self.iter]

    def _reset_period(self):
        self.cnt = 0
