import hjson
from rich import print
from volsurfs_py.params.params import Params


class HyperParams(Params):
    # lr schedule
    lr = 1e-3
    lr_milestones = [100000, 150000, 180000, 190000]
    training_end_iter = 200000
    nr_warmup_iters = 3000

    # appearance
    geom_feat_size = 0
    rgb_mlp_layers_dims = [128, 128, 64]
    appearance_predict_sh_coeffs = False
    sh_degree = 3
    rgb_mlp_output_dims = 3
    # rgb_pos_dep = True
    rgb_view_dep = True
    rgb_normal_dep = True
    rgb_geom_feat_dep = True
    rgb_use_lipshitz_mlp = False
    rgb_pos_encoder_type = "permutohash"
    rgb_dir_encoder_type = "spherical_harmonics"

    # background
    bg_pos_encoder_type = "permutohash"
    bg_dir_encoder_type = "spherical_harmonics"

    # color calibration
    use_color_calibration = False

    # coarse to fine
    rgb_nr_iters_for_c2f = 0
    bg_nr_iters_for_c2f = 0

    # losses weights
    is_training_masked = False
    is_testing_masked = False
    mask_weight = 0.0

    # occupancy grid
    use_occupancy_grid = True

    # grad scaling
    use_grad_scaler = False

    # sampling
    training_rays_batch_size = 512
    test_rays_batch_size = 512
    nr_training_rays_per_pixel = 1  # if > 1, performs super-sampling
    nr_test_rays_per_pixel = 1  # if > 1, performs super-sampling
    jitter_training_rays = True
    jitter_test_rays = False
    # nr of rays can be dynamically changed so that the number of samples per iteration is approximately the same
    # reduce or disable this for faster training or if your GPU has little VRAM
    is_nr_training_rays_dynamic = True
    target_nr_of_training_samples = 512 * (64 + 16 + 16)
    do_importance_sampling = False
    max_nr_imp_samples_per_ray = 32
    min_dist_between_samples = 1e-4
    min_nr_samples_per_ray = 1
    max_nr_samples_per_ray = 64  # for the foreground
    nr_samples_bg = 32

    def __init__(self, cfg_path):
        # load config file
        super().__init__(cfg_path)

        cfg_hp = self.cfg["hyper_params"]

        if "lr" in cfg_hp:
            self.lr = float(cfg_hp["lr"])
        if "training_end_iter" in cfg_hp:
            self.training_end_iter = int(cfg_hp["training_end_iter"])
        if "lr_milestones" in cfg_hp:
            self.lr_milestones = cfg_hp["lr_milestones"]
        if "nr_warmup_iters" in cfg_hp:
            self.nr_warmup_iters = int(cfg_hp["nr_warmup_iters"])

        # appearance
        if "rgb_nr_iters_for_c2f" in cfg_hp:
            self.rgb_nr_iters_for_c2f = int(cfg_hp["rgb_nr_iters_for_c2f"])
        if "bg_nr_iters_for_c2f" in cfg_hp:
            self.bg_nr_iters_for_c2f = int(cfg_hp["bg_nr_iters_for_c2f"])
        if "geom_feat_size" in cfg_hp:
            self.geom_feat_size = int(cfg_hp["geom_feat_size"])
        if "rgb_mlp_layers_dims" in cfg_hp:
            self.rgb_mlp_layers_dims = cfg_hp["rgb_mlp_layers_dims"]
        if "appearance_predict_sh_coeffs" in cfg_hp:
            self.appearance_predict_sh_coeffs = bool(
                cfg_hp["appearance_predict_sh_coeffs"]
            )
        if "sh_degree" in cfg_hp:
            self.sh_degree = int(cfg_hp["sh_degree"])
        if "rgb_pos_encoder_type" in cfg_hp:
            self.rgb_pos_encoder_type = cfg_hp["rgb_pos_encoder_type"]
        if "rgb_dir_encoder_type" in cfg_hp:
            self.rgb_dir_encoder_type = cfg_hp["rgb_dir_encoder_type"]
        # if "rgb_pos_dep" in cfg_hp:
        #     self.rgb_pos_dep = bool(cfg_hp["rgb_pos_dep"])
        if "rgb_view_dep" in cfg_hp:
            self.rgb_view_dep = bool(cfg_hp["rgb_view_dep"])
        if "rgb_normal_dep" in cfg_hp:
            self.rgb_normal_dep = bool(cfg_hp["rgb_normal_dep"])
        if "rgb_geom_feat_dep" in cfg_hp:
            self.rgb_geom_feat_dep = bool(cfg_hp["rgb_geom_feat_dep"])
            if self.rgb_geom_feat_dep:
                # check if self.geom_feat_size > 0
                if self.geom_feat_size == 0:
                    print(
                        "\n[bold red]ERROR[/bold red]: rgb_geom_feat_dep can't be true if geom_feat_size is 0"
                    )
                    exit(1)
        if "rgb_use_lipshitz_mlp" in cfg_hp:
            self.rgb_use_lipshitz_mlp = bool(cfg_hp["rgb_use_lipshitz_mlp"])

        # background
        if "bg_pos_encoder_type" in cfg_hp:
            self.bg_pos_encoder_type = cfg_hp["bg_pos_encoder_type"]
        if "bg_dir_encoder_type" in cfg_hp:
            self.bg_dir_encoder_type = cfg_hp["bg_dir_encoder_type"]

        # color calibration
        if "use_color_calibration" in cfg_hp:
            self.use_color_calibration = bool(cfg_hp["use_color_calibration"])

        # occupancy grid
        if "use_occupancy_grid" in cfg_hp:
            self.use_occupancy_grid = bool(cfg_hp["use_occupancy_grid"])

        if "do_importance_sampling" in cfg_hp:
            self.do_importance_sampling = bool(cfg_hp["do_importance_sampling"])
        if "max_nr_imp_samples_per_ray" in cfg_hp:
            self.max_nr_imp_samples_per_ray = int(cfg_hp["max_nr_imp_samples_per_ray"])
        if "training_rays_batch_size" in cfg_hp:
            self.training_rays_batch_size = int(cfg_hp["training_rays_batch_size"])
        if "test_rays_batch_size" in cfg_hp:
            self.test_rays_batch_size = int(cfg_hp["test_rays_batch_size"])
            # do not batchify test rendering if batch size is 1
            if self.test_rays_batch_size < 1:
                self.test_rays_batch_size = None
        if "nr_training_rays_per_pixel" in cfg_hp:
            self.nr_training_rays_per_pixel = int(cfg_hp["nr_training_rays_per_pixel"])
        if "nr_test_rays_per_pixel" in cfg_hp:
            self.nr_test_rays_per_pixel = int(cfg_hp["nr_test_rays_per_pixel"])
        if "jitter_training_rays" in cfg_hp:
            self.jitter_training_rays = bool(cfg_hp["jitter_training_rays"])
        if "jitter_test_rays" in cfg_hp:
            self.jitter_test_rays = bool(cfg_hp["jitter_test_rays"])
        if "is_nr_training_rays_dynamic" in cfg_hp:
            self.is_nr_training_rays_dynamic = bool(
                cfg_hp["is_nr_training_rays_dynamic"]
            )
        if "min_dist_between_samples" in cfg_hp:
            self.min_dist_between_samples = float(cfg_hp["min_dist_between_samples"])
        if "min_nr_samples_per_ray" in cfg_hp:
            self.min_nr_samples_per_ray = int(cfg_hp["min_nr_samples_per_ray"])
        if "max_nr_samples_per_ray" in cfg_hp:
            self.max_nr_samples_per_ray = int(cfg_hp["max_nr_samples_per_ray"])
        if "nr_samples_bg" in cfg_hp:
            self.nr_samples_bg = int(cfg_hp["nr_samples_bg"])

        if "is_training_masked" in cfg_hp:
            self.is_training_masked = bool(cfg_hp["is_training_masked"])
        if "is_testing_masked" in cfg_hp:
            self.is_testing_masked = bool(cfg_hp["is_testing_masked"])
        if "mask_weight" in cfg_hp:
            self.mask_weight = float(cfg_hp["mask_weight"])

        # if importance sampling is on, we need at least 3 samples per ray to compute the cdf
        if self.do_importance_sampling:
            if self.min_nr_samples_per_ray < 3:
                print(
                    "[bold yellow]WARNING[/bold yellow]: importance sampling is on, setting min_nr_samples_per_ray to 3"
                )
                self.min_nr_samples_per_ray = 3

        if self.nr_test_rays_per_pixel > 1:
            if not self.jitter_test_rays:
                print(
                    "[bold yellow]WARNING[/bold yellow]: nr_test_rays_per_pixel > 1, setting jitter_test_rays to True"
                )
                self.jitter_test_rays = True

        if self.nr_training_rays_per_pixel > 1:
            if not self.jitter_training_rays:
                print(
                    "[bold yellow]WARNING[/bold yellow]: nr_training_rays_per_pixel > 1, setting jitter_training_rays to True"
                )
                self.jitter_training_rays = True


class HyperParamsSuRF(HyperParams):
    # geometry
    sdf_mlp_layers_dims = [32, 32]
    sdf_mlp_output_dims = 1
    sdf_encoding_type = "permutohash"

    # nr iters of sphere init
    init_phase_end_iter = 4000

    # coarse to fine
    sdf_nr_iters_for_c2f = 10000

    # variance
    first_phase_variance_start_value = 0.3
    first_phase_variance_end_value = 0.8
    # nr iters until the SDF to density transform is sharp
    first_phase_end_iter = 35000

    # nr iters when we reduce the curvature loss
    reduce_curv_start_iter = None
    reduce_curv_end_iter = None

    # loss weights
    eikonal_weight = 0.0
    curvature_weight = 0.0
    lipshitz_weight = 0.0
    offsurface_weight = 0.0

    def __init__(self, cfg_path):
        # load config file
        super().__init__(cfg_path)

        cfg_hp = self.cfg["hyper_params"]

        # SDF
        if "sdf_nr_iters_for_c2f" in cfg_hp:
            self.sdf_nr_iters_for_c2f = int(cfg_hp["sdf_nr_iters_for_c2f"])

        if "sdf_mlp_layers_dims" in cfg_hp:
            self.sdf_mlp_layers_dims = cfg_hp["sdf_mlp_layers_dims"]
        if "sdf_encoding_type" in cfg_hp:
            self.sdf_encoding_type = cfg_hp["sdf_encoding_type"]

        if "init_phase_end_iter" in cfg_hp:
            self.init_phase_end_iter = int(cfg_hp["init_phase_end_iter"])
        if "first_phase_end_iter" in cfg_hp:
            self.first_phase_end_iter = int(cfg_hp["first_phase_end_iter"])

        if "first_phase_variance_start_value" in cfg_hp:
            self.first_phase_variance_start_value = float(
                cfg_hp["first_phase_variance_start_value"]
            )
        if "first_phase_variance_end_value" in cfg_hp:
            self.first_phase_variance_end_value = float(
                cfg_hp["first_phase_variance_end_value"]
            )

        if "eikonal_weight" in cfg_hp:
            self.eikonal_weight = float(cfg_hp["eikonal_weight"])
        if "curvature_weight" in cfg_hp:
            self.curvature_weight = float(cfg_hp["curvature_weight"])
        if "lipshitz_weight" in cfg_hp:
            self.lipshitz_weight = float(cfg_hp["lipshitz_weight"])
        if "offsurface_weight" in cfg_hp:
            self.offsurface_weight = float(cfg_hp["offsurface_weight"])

        if "reduce_curv_start_iter" in cfg_hp:
            self.reduce_curv_start_iter = int(cfg_hp["reduce_curv_start_iter"])
        if "reduce_curv_end_iter" in cfg_hp:
            self.reduce_curv_end_iter = int(cfg_hp["reduce_curv_end_iter"])


class HyperParamsOffsetsSuRFs(HyperParamsSuRF):

    # surfaces
    nr_inner_surfs = 1
    nr_outer_surfs = 1
    delta_surfs_multiplier = 1.0

    color_init_phase_end_iter = 6000

    are_surfs_colors_indep = False
    are_surfs_transparency_indep = False
    is_inner_surf_solid = False

    transp_view_dep = True
    transp_normal_dep = True
    transp_geom_feat_dep = True

    # loss weights
    offsets_weight = 0.0
    support_surfs_eikonal_weight = 0.0

    with_alpha_decay = True

    # first_phase_stop_main_surf = False
    # second_phase_stop_main_surf = False
    # second_phase_stop_offsets = False

    def __init__(self, cfg_path):
        # load config file
        super().__init__(cfg_path)

        cfg_hp = self.cfg["hyper_params"]

        if "nr_inner_surfs" in cfg_hp:
            self.nr_inner_surfs = int(cfg_hp["nr_inner_surfs"])
        if "nr_outer_surfs" in cfg_hp:
            self.nr_outer_surfs = int(cfg_hp["nr_outer_surfs"])

        if "color_init_phase_end_iter" in cfg_hp:
            self.color_init_phase_end_iter = int(cfg_hp["color_init_phase_end_iter"])

        if "delta_surfs_multiplier" in cfg_hp:
            self.delta_surfs_multiplier = float(cfg_hp["delta_surfs_multiplier"])

        if "are_surfs_colors_indep" in cfg_hp:
            self.are_surfs_colors_indep = bool(cfg_hp["are_surfs_colors_indep"])
        if "are_surfs_transparency_indep" in cfg_hp:
            self.are_surfs_transparency_indep = bool(
                cfg_hp["are_surfs_transparency_indep"]
            )
        if "is_inner_surf_solid" in cfg_hp:
            self.is_inner_surf_solid = bool(cfg_hp["is_inner_surf_solid"])

        if "transp_view_dep" in cfg_hp:
            self.transp_view_dep = bool(cfg_hp["transp_view_dep"])
        if "transp_normal_dep" in cfg_hp:
            self.transp_normal_dep = bool(cfg_hp["transp_normal_dep"])
        if "transp_geom_feat_dep" in cfg_hp:
            # check if self.geom_feat_size > 0
            if self.geom_feat_size == 0:
                print(
                    "\n[bold red]ERROR[/bold red]: transp_geom_feat_dep can't be true if geom_feat_size is 0"
                )
                exit(1)
            self.transp_geom_feat_dep = bool(cfg_hp["transp_geom_feat_dep"])

        if "offsets_weight" in cfg_hp:
            self.offsets_weight = float(cfg_hp["offsets_weight"])
        if "support_surfs_eikonal_weight" in cfg_hp:
            self.support_surfs_eikonal_weight = float(
                cfg_hp["support_surfs_eikonal_weight"]
            )

        if "with_alpha_decay" in cfg_hp:
            self.with_alpha_decay = bool(cfg_hp["with_alpha_decay"])

        # if "first_phase_stop_main_surf" in cfg_hp:
        #     self.first_phase_stop_main_surf = bool(cfg_hp["first_phase_stop_main_surf"])
        # if "second_phase_stop_main_surf" in cfg_hp:
        #     self.second_phase_stop_main_surf = bool(cfg_hp["second_phase_stop_main_surf"])
        # if "second_phase_stop_offsets" in cfg_hp:
        #     self.second_phase_stop_offsets = bool(cfg_hp["second_phase_stop_offsets"])


class HyperParamsNeRF(HyperParams):
    # density
    density_mlp_layers_dims = [32, 32]
    density_mlp_output_dims = 1
    density_encoding_type = "permutohash"

    # coarse to fine
    density_nr_iters_for_c2f = 10000

    # losses (do not use when mask is used)
    sparsity_weight = 0.0

    def __init__(self, cfg_path):
        # load config file
        super().__init__(cfg_path)

        cfg_hp = self.cfg["hyper_params"]

        if "density_nr_iters_for_c2f" in cfg_hp:
            self.density_nr_iters_for_c2f = int(cfg_hp["density_nr_iters_for_c2f"])
        if "sparsity_weight" in cfg_hp:
            self.sparsity_weight = float(cfg_hp["sparsity_weight"])

        # density
        if "density_mlp_layers_dims" in cfg_hp:
            self.density_mlp_layers_dims = cfg_hp["density_mlp_layers_dims"]
        if "density_encoding_type" in cfg_hp:
            self.density_encoding_type = cfg_hp["density_encoding_type"]


class HyperParamsVolSurfs(HyperParams):
    """
    hyper_params for finetuning an appearance model on meshes
    """

    meshes_indices = None
    are_volsurfs_colors_indep = True
    are_volsurfs_alphas_indep = True
    is_inner_mesh_solid = True
    using_neural_textures = False
    using_neural_textures_anchor = False
    using_neural_textures_lerp = False
    using_sh_quantization = False
    using_sh_squeezing = False

    transp_view_dep = True
    transp_normal_dep = True

    sh_range = [1.0, 5.0, 10.0, 20.0]
    textures_res = [2048, 1024, 512, 256]

    with_alpha_decay = True

    def __init__(self, cfg_path):
        # load config file
        super().__init__(cfg_path)

        cfg_hp = self.cfg["hyper_params"]

        if "meshes_indices" in cfg_hp:
            self.meshes_indices = cfg_hp["meshes_indices"]
        if "is_inner_mesh_solid" in cfg_hp:
            self.is_inner_mesh_solid = bool(cfg_hp["is_inner_mesh_solid"])
        if "using_neural_textures" in cfg_hp:
            self.using_neural_textures = bool(cfg_hp["using_neural_textures"])
        if "using_neural_textures_anchor" in cfg_hp:
            self.using_neural_textures_anchor = bool(
                cfg_hp["using_neural_textures_anchor"]
            )
        if "using_neural_textures_lerp" in cfg_hp:
            self.using_neural_textures_lerp = bool(cfg_hp["using_neural_textures_lerp"])
        if "using_sh_quantization" in cfg_hp:
            self.using_sh_quantization = bool(cfg_hp["using_sh_quantization"])
        if "using_sh_squeezing" in cfg_hp:
            self.using_sh_squeezing = bool(cfg_hp["using_sh_squeezing"])

        if "are_volsurfs_colors_indep" in cfg_hp:
            self.are_volsurfs_colors_indep = bool(cfg_hp["are_volsurfs_colors_indep"])
        if "are_volsurfs_alphas_indep" in cfg_hp:
            self.are_volsurfs_alphas_indep = bool(cfg_hp["are_volsurfs_alphas_indep"])

        if "transp_view_dep" in cfg_hp:
            self.transp_view_dep = bool(cfg_hp["transp_view_dep"])
        if "transp_normal_dep" in cfg_hp:
            self.transp_normal_dep = bool(cfg_hp["transp_normal_dep"])

        if "sh_range" in cfg_hp:
            self.sh_range = cfg_hp["sh_range"]
            if len(self.sh_range) != 4:
                print("\n[bold red]ERROR[/bold red]: sh_range should have 4 elements")
                exit(1)
        if "textures_res" in cfg_hp:
            self.textures_res = cfg_hp["textures_res"]
            if len(self.textures_res) != 4:
                print(
                    "\n[bold red]ERROR[/bold red]: textures_res should have 4 elements"
                )
                exit(1)

        if "with_alpha_decay" in cfg_hp:
            self.with_alpha_decay = bool(cfg_hp["with_alpha_decay"])

        if self.using_neural_textures:
            # neural textures assumes sh predictions
            if not self.appearance_predict_sh_coeffs:
                print(
                    "\n[bold red]ERROR[/bold red]: when using neural textures, appearance_predict_sh_coeffs should be True"
                )
                exit(1)
            # neural textures assumes color view dependency
            if not self.rgb_view_dep:
                print(
                    "\n[bold red]ERROR[/bold red]: when using neural textures, rgb_view_dep should be True"
                )
                exit(1)
            # neural textures can't be conditioned on normals
            if self.transp_normal_dep or self.rgb_normal_dep:
                print(
                    "\n[bold red]ERROR[/bold red]: when using neural textures, transp_normal_dep and rgb_normal_dep should be False"
                )
                exit(1)

        if self.rgb_geom_feat_dep or self.geom_feat_size > 0:
            # appearance on meshes cant be conditioned on geom features
            print(
                "\n[bold red]ERROR[/bold red]: rgb_geom_feat_dep should be False, geom_feat_size should be 0"
            )
            exit(1)


def get_method_hyper_params(method_name, cfg_path):
    if method_name == "surf":
        return HyperParamsSuRF(cfg_path)
    # if method_name == "adaptive_surf":
    #     return HyperParamsSuRF(cfg_path)
    elif method_name == "nerf":
        return HyperParamsNeRF(cfg_path)
    elif method_name == "volsurfs":
        return HyperParamsVolSurfs(cfg_path)
    elif method_name == "offsets_surfs":
        return HyperParamsOffsetsSuRFs(cfg_path)
    else:
        print(f"[bold red]ERROR[/bold red]: unknown method name {method_name}")
        exit(1)
