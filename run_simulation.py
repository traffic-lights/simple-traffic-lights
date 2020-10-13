from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from pathlib import Path

from functools import partial

import torch

from environment.aaai_env import AaaiEnv
from environment.simple_env import SimpleEnv
from trainings.training_parameters import TrainingState

MIN_SLIDER_VAL = 0
MAX_SLIDER_VAL = 50
SLIDER_TICK_INTERVAL = 5


class SumoWorker(QRunnable):
    def __init__(self, period_dict, active_lanes, states_path):
        super(SumoWorker, self).__init__()
        self.period_dict = period_dict
        self.active_lanes = active_lanes
        self.state = TrainingState.from_path(states_path)
        self.model = self.state.model

        self.env = AaaiEnv(render=True, save_replay=False, conf_file="aaai_qt.json")

        ret = self.env.vehicle_generator.get_periods()

        for key in ret:
            self.period_dict[key] = ret[key]

        ret = self.env.vehicle_generator.get_active_lanes()
        
        for key in ret:
            self.active_lanes[key] = ret[key]

    @pyqtSlot()
    def run(self):
        state = self.env.reset()
        ep_len = 0

        while not False:
            ep_len += 1
            tensor_state = torch.tensor([state], dtype=torch.float32)
            action = self.model(tensor_state).max(1)[1][0].cpu().detach().numpy().item()

            try:
                next_state, reward, _, info = self.env.step(action)
            except Exception:
                break

            state = next_state

            self.env.vehicle_generator.update(self.period_dict, self.active_lanes)

        print(f"simulation done")
        self.env.close()


# Subclass QMainWindow to customise your application's main window
class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        filenames = []

        if dlg.exec_():
            filenames = dlg.selectedFiles()

        layout = QVBoxLayout()
        label = QLabel("SUMO period controller")

        layout.addWidget(label)

        self.lanes_dict = {}
        self.active_lanes = {}

        self.threadpool = QThreadPool()
        worker = SumoWorker(self.lanes_dict, self.active_lanes, filenames[0])

        self.labels = {}
        self.sliders = []
        self.checkboxes = []

        for key in self.lanes_dict:
            period = self.lanes_dict[key]
            self.labels[key] = QLabel(f"{key} period: {period}s")
            layout.addWidget(self.labels[key])

            lane_box = QHBoxLayout()

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(MIN_SLIDER_VAL)
            slider.setMaximum(MAX_SLIDER_VAL)
            slider.setValue(period)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(SLIDER_TICK_INTERVAL)

            slider.valueChanged.connect(partial(self.update_period, key, slider))

            lane_box.addWidget(slider)

            self.sliders.append(slider)

            checkbox = QCheckBox("active")
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(partial(self.update_active, key, checkbox))

            lane_box.addWidget(checkbox)
            layout.addLayout(lane_box)

            self.checkboxes.append(checkbox)

        self.setLayout(layout)

        self.threadpool.start(worker)

    def update_period(self, lane, slider):
        period = slider.value()
        self.lanes_dict[lane] = period
        self.labels[lane].setText(f"{lane} period: {period}s")

    def update_active(self, lane, checkbox):
        self.active_lanes[lane] = checkbox.isChecked()


app = QApplication([])

window = MainWindow()
window.show()

app.exec_()
