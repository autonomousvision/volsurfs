from volsurfs_py.callbacks.callback import Callback
import wandb
import numpy as np


class WandBCallback(Callback):
    """WandB callback for logging training and testing metrics."""

    def __init__(
        self,
        run_name,
        run_id,
        entity,
        models=None,
        config_dict=None,
    ):
        self.run_name = run_name

        assert wandb.run is None
        self.run = wandb.init(
            project="volsurfs",
            id=run_id,
            name=run_name,
            entity=entity,
            config=config_dict,
            resume="allow",
        )
        assert wandb.run is not None

        # wandb watch models
        if models is not None:
            for _, model in models.items():
                if model is not None:
                    wandb.watch(model, log="all", log_freq=300)

    def training_started(self, **kwargs):
        print("training started")
        pass

    def training_ended(self, **kwargs):
        print("training ended")
        pass

    def after_forward_pass(self, phase, train_losses, additional_info_to_log, **kwargs):
        # iterate over train losses and add them to wandb
        for key, value in train_losses.items():
            if value != 0.0:
                if key == "loss":
                    wandb.log({f"train/{key}": value}, step=phase.iter_nr)
                else:
                    wandb.log({f"train/{key}_loss": value}, step=phase.iter_nr)

        # iterate over additional info and add them to wandb
        for key, value in additional_info_to_log.items():
            if value != 0.0:
                wandb.log({f"train/{key}": value}, step=phase.iter_nr)

        # iterate over kwargs and add them to wandb
        for key, value in kwargs.items():
            if value != 0.0:  # exclude 0 values
                wandb.log({f"train/{key}": value}, step=phase.iter_nr)

    def after_backward_pass(self, phase, models_grad_norms):
        # iterate over models_grad_norms and add them to wandb
        for model_grad_norm in models_grad_norms:
            if model_grad_norm["grad_norm"] != 0.0:
                wandb.log(
                    {
                        f"models_grad_norm/{model_grad_norm['name']}": model_grad_norm[
                            "grad_norm"
                        ]
                    },
                    step=phase.iter_nr,
                )

    def iter_ended(
        self, phase, imgs, test_losses, eval_metrics, occupancy_grid_stats, **kwargs
    ):
        # iteration time
        wandb.log({"train/iters_per_sec": phase.iters_per_second}, step=phase.iter_nr)

        # iterate over test losses and add them to wandb
        for key, value in test_losses.items():
            if value != 0.0:
                if key == "loss":
                    wandb.log({f"test/{key}": value}, step=phase.iter_nr)
                else:
                    wandb.log({f"test/{key}_loss": value}, step=phase.iter_nr)

        # log images
        for render_mode, renders_dict in imgs.items():
            for key, img_float in renders_dict.items():
                if img_float is not None:
                    # convert to uint8
                    img_int8 = np.uint8(np.clip(img_float, 0, 1) * 255)
                    wandb.log(
                        {f"test/{render_mode}/{key}": wandb.Image(img_int8)},
                        step=phase.iter_nr,
                    )

        # iterate over eval metrics and add them to wandb
        for set_key, set_metrics in eval_metrics.items():
            for metric_key, metric_val in set_metrics.items():
                if metric_val != 0.0:
                    wandb.log(
                        {f"{set_key}/{metric_key}": metric_val}, step=phase.iter_nr
                    )

        # iterate over occupancy grid stats and add them to wandb
        for metric_key, metric_val in occupancy_grid_stats.items():
            if metric_val != 0.0:
                wandb.log(
                    {f"occupancy_grid/{metric_key}": metric_val}, step=phase.iter_nr
                )
