import time
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
    def __init__(self, actions, switch_periods_in_s, current_time_in_s_f):
        super(TimedCyclicSwitchController, self).__init__(actions)
        self.switch_periods_in_s = switch_periods_in_s[-1:] + switch_periods_in_s[:-1]
        self.time_f = current_time_in_s_f
        self.last_check_time = 0

    def _check_is_wait_period_over_and_update(self):
        ret_val = False

        check_time = self.time_f()
        if check_time - self.last_check_time >= self.switch_periods_in_s[self.iter]:
            ret_val = True

        self.last_check_time = check_time
        return ret_val

    def _reset_period(self):
        self.last_check_time = self.time_f()


class BlockingTimedCyclicSwitchController(TimedCyclicSwitchController):
    def __init__(self, actions, switch_periods_in_s, current_time_in_s_f):
        super(BlockingTimedCyclicSwitchController, self).__init__(actions, switch_periods_in_s, current_time_in_s_f)
        self.switch_periods_in_s = switch_periods_in_s[-1:] + switch_periods_in_s[:-1]
        self.last_check_time = 0

    def _check_is_wait_period_over_and_update(self):
        time_since_last_change = self.time_f() - self.last_check_time
        if time_since_last_change <= self.switch_periods_in_s[self.iter]:
            time.sleep(self.switch_periods_in_s[self.iter] - time_since_last_change)

        self.last_check_time = self.time_f()
        return True


class CallCounterCyclicSwitchController(CyclicSwitchControllerI):
    def __init__(self, actions, nums_calls_before_reset):
        super(CallCounterCyclicSwitchController, self).__init__(actions)
        self.nums_calls_before_reset = nums_calls_before_reset
        self.cnt = -1

    def _check_is_wait_period_over_and_update(self):
        #print('gneE19_0', traci.lane.getLastStepVehicleNumber('gneE19_0'))
        #print('gneE19_1', traci.lane.getLastStepVehicleNumber('gneE19_1'))
        #print('gneE19_2', traci.lane.getLastStepVehicleNumber('gneE19_2'))
        #input('waiting on phase %d...' % self.iter)
        self.cnt += 1
        return self.cnt >= self.nums_calls_before_reset[self.iter]

    def _reset_period(self):
        self.cnt = 0
