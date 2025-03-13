from volsurfs_py.callbacks.callback import CallbacksGroup
from volsurfs_py.callbacks.wandb_callback import WandBCallback
from volsurfs_py.callbacks.state_callback import StateCallback


def create_callbacks(
    method,
    scene_name,
    exp_name,
    train_params,
    data_params,
    run_id,
    profiler=None,
):
    # state callback
    cb_list = []
    cb_list.append(StateCallback())

    # wandb callback
    if train_params.with_wandb:
        entity_name = "stefanoesp"  # your username on wandb
        wandb_callback = WandBCallback(
            run_name=method.method_name + "_" + exp_name,
            run_id=run_id,
            entity=entity_name,
            models=method.models,
            config_dict={
                "scene": scene_name,
                "exp_name": exp_name,
                "hyper_params": method.hyper_params.dict(),
                "data_params": data_params.dict(),
                "train_params": train_params.dict(),
            },
        )
        cb_list.append(wandb_callback)

    # group callbacks
    cb = CallbacksGroup(cb_list, profiler=profiler)

    return cb
