from volsurfs_py.callbacks.callback import *
import os
import torch
import time


class StateCallback(Callback):
    iter_start_time = 0.0
    iter_end_time = 0.0

    def __init__(self):
        pass

    def iter_started(self, phase, **kwargs):
        phase.iter_start_time = time.time()

    def iter_ended(self, phase, **kwargs):
        phase.iter_end_time = time.time()
        phase.iters_per_second = 1.0 / (phase.iter_end_time - phase.iter_start_time)

    def after_forward_pass(self, phase, train_losses, **kwargs):
        pass
