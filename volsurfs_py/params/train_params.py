import hjson

from volsurfs_py.params.params import Params


class TrainParams(Params):

    method_name = ""
    with_wandb = False
    # TODO: add wandb params
    # wandb_project: str = "volsurfs"
    # wandb_entity: str = "permuto"
    # wandb_run_name: str = "default"
    # wandb_tags: list = ["default"]
    # wandb_notes: str = "default"

    save_checkpoints = True
    checkpoint_freq = 25000

    compute_test_loss = True
    compute_test_loss_freq = 500

    eval_test = True
    eval_test_freq = 5000

    eval_train = True
    eval_train_freq = 5000

    render_freq = 5000

    def __init__(self, method_name, cfg_path):
        self.method_name = method_name

        # load config file
        super().__init__(cfg_path)

        for key, cfg in self.cfg.items():
            # general params

            if "with_wandb" in cfg:
                self.with_wandb = bool(cfg["with_wandb"])

            if "save_checkpoints" in cfg:
                self.save_checkpoints = bool(cfg["save_checkpoints"])

            if "compute_test_loss" in cfg:
                self.compute_test_loss = bool(cfg["compute_test_loss"])

            if "eval_test" in cfg:
                self.eval_test = bool(cfg["eval_test"])

            if "eval_train" in cfg:
                self.eval_train = bool(cfg["eval_train"])

            # method specific params

            if key == method_name:

                if "checkpoint_freq" in cfg:
                    self.checkpoint_freq = int(cfg["checkpoint_freq"])

                if "eval_train_freq" in cfg:
                    self.eval_train_freq = int(cfg["eval_train_freq"])

                if "eval_test_freq" in cfg:
                    self.eval_test_freq = int(cfg["eval_test_freq"])

                if "compute_test_loss_freq" in cfg:
                    self.compute_test_loss_freq = int(cfg["compute_test_loss_freq"])

                if "render_freq" in cfg:
                    self.render_freq = int(cfg["render_freq"])
