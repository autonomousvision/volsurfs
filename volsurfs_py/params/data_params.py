import os
from volsurfs_py.params.params import Params


def merge_and_override(cfg, scene_cfg):
    return dict(list(cfg.items()) + list(scene_cfg.items()))


class DataParams(Params):

    def __init__(self, datasets_path, dataset_name, scene_name, cfg_path):

        # make sure folder exists
        assert os.path.exists(
            datasets_path
        ), f"data path: {datasets_path} does not exist"

        self.datasets_path = datasets_path
        self.dataset_name = dataset_name
        self.scene_name = scene_name
        self.bg_color = None

        # load config file
        super().__init__(cfg_path)

        for key, cfg in self.cfg.items():

            # dataset specific params

            if key == dataset_name:

                print(f"loading {dataset_name} data params")

                #
                if "bg_color" in cfg:
                    print("LOADED BG COLOR")
                    self.bg_color = str(cfg["bg_color"])

                # check if scene specific params are present
                if "scenes" in cfg:
                    if self.scene_name in cfg["scenes"]:
                        scene_cfg = cfg["scenes"][self.scene_name]
                        cfg = merge_and_override(cfg, scene_cfg)

                if "subsample_factor" in cfg:
                    self.subsample_factor = float(cfg["subsample_factor"])

                # DTU, Blender
                if "scene_radius_mult" in cfg:
                    self.scene_radius_mult = float(cfg["scene_radius_mult"])

                # DTU, Blender
                if "load_mask" in cfg:
                    self.load_mask = bool(cfg["load_mask"])

                # DTU, Blender
                if "target_cameras_max_distance" in cfg:
                    self.target_cameras_max_distance = float(
                        cfg["target_cameras_max_distance"]
                    )

                # DTU, Blender
                if "rotate_scene_x_axis_deg" in cfg:
                    self.rotate_scene_x_axis_deg = float(cfg["rotate_scene_x_axis_deg"])

                if "translate_scene_x" in cfg:
                    self.translate_scene_x = float(cfg["translate_scene_x"])

                if "translate_scene_y" in cfg:
                    self.translate_scene_y = float(cfg["translate_scene_y"])

                if "translate_scene_z" in cfg:
                    self.translate_scene_z = float(cfg["translate_scene_z"])

                # DTU
                if "train_test_overlap" in cfg:
                    self.train_test_overlap = bool(cfg["train_test_overlap"])

                # DTU
                if "test_camera_freq" in cfg:
                    self.test_camera_freq = int(cfg["test_camera_freq"])

                # Blender
                if "white_bg" in cfg:
                    self.white_bg = bool(cfg["white_bg"])

                # Blender
                if "test_skip" in cfg:
                    self.test_skip = int(cfg["test_skip"])

                if "init_sphere_scale" in cfg:
                    self.init_sphere_scale = float(cfg["init_sphere_scale"])

                if "scene_type" in cfg:
                    self.scene_type = str(cfg["scene_type"])
