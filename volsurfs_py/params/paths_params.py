import os
from rich import print
from volsurfs_py.params.params import Params


class PathsParams(Params):

    def __init__(self, args):
        # load config file
        super().__init__(cfg_path=args["paths_config"])

        # read from config file
        self.runs_root = self.cfg["paths"]["runs"]
        self.datasets = self.cfg["paths"]["datasets"]

        self.root = args["root"]
        self.train_config = os.path.join(self.root, "config", "train_config.cfg")
        self.data_config = os.path.join(self.root, "config", "data_config.cfg")
        self.hyper_params = os.path.join(
            self.root, "config", args["method_name"], args["exp_name"] + ".cfg"
        )
        self.runs = os.path.join(
            self.runs_root, args["method_name"], args["exp_name"], args["scene"]
        )

        # teacher model (for distillation)
        if args["teacher_exp_config"] is not None:

            if args["teacher_exp_config"] is None:
                print("[bold red]ERROR[/bold red]: teacher_exp_config not provided")
                exit(1)

            self.teacher_runs = os.path.join(
                self.runs_root, "nerf", args["teacher_exp_config"], args["scene"]
            )
            # assert it exists
            if not os.path.exists(self.teacher_runs):
                print(
                    f"[bold red]ERROR[/bold red]: teacher_runs path ({self.teacher_runs}) not found"
                )
                exit(1)
        else:
            self.teacher_runs = None

        if args["teacher_run_id"] is not None:

            if args["teacher_exp_config"] is None:
                print("[bold red]ERROR[/bold red]: teacher_exp_config not provided")
                exit(1)

            # check if checkpoints folder exists
            self.teacher_checkpoints = os.path.join(
                self.teacher_runs, args["teacher_run_id"]
            )
            if not os.path.exists(self.teacher_checkpoints):
                print(
                    f"[bold red]ERROR[/bold red]: teacher_checkpoints ({self.teacher_checkpoints}) not found"
                )
                exit(1)

            # teacher config file path
            self.teacher_hyper_params = os.path.join(
                self.teacher_checkpoints, "config", "hyper_params.cfg"
            )
            if not os.path.exists(self.teacher_hyper_params):
                print(
                    f"[bold red]ERROR[/bold red]: teacher_hyper_params ({self.teacher_hyper_params}) not found"
                )
                exit(1)
