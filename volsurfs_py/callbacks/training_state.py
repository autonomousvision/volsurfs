class TrainingState:
    """
    Stores information about the current training iteration.
    """

    def __init__(self):
        self.iter_nr = 0
        self.is_first_iter = True

        self.iter_start_time = 0.0
        self.iter_end_time = 0.0
        self.iters_per_second = 0.0

        self.test_cam_idx = 0
