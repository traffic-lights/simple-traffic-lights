from abc import ABC
from dataclasses import dataclass

from traffic_controllers.trafffic_controller import TrafficController


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


class TimedCyclicSwitchControllerRunner(CyclicSwitchControllerI):
    def __init__(self, connection, actions, switch_periods_in_s):
        super().__init__(actions)
        self.connection = connection
        self.switch_periods_in_s = switch_periods_in_s[-1:] + switch_periods_in_s[:-1]
        self.last_switch_time = 0

    def _check_is_wait_period_over_and_update(self):
        return self.connection.simulation.getTime() - self.last_switch_time >= self.switch_periods_in_s[self.iter]

    def _reset_period(self):
        self.last_switch_time = self.connection.simulation.getTime()


@dataclass
class TimedCyclicSwitchController(TrafficController):
    actions: list
    switch_periods_in_s: list

    def with_connection(self, connection):
        return TimedCyclicSwitchControllerRunner(connection, self.actions, self.switch_periods_in_s)


class CallCounterCyclicSwitchController(CyclicSwitchControllerI, TrafficController):
    def __init__(self, actions, nums_calls_before_reset):
        super(CallCounterCyclicSwitchController, self).__init__(actions)
        self.nums_calls_before_reset = nums_calls_before_reset
        self.cnt = -1

    def _check_is_wait_period_over_and_update(self):
        self.cnt += 1
        return self.cnt >= self.nums_calls_before_reset[self.iter]

    def _reset_period(self):
        self.cnt = 0
