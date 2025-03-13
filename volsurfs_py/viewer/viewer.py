import numpy as np
import dearpygui.dearpygui as dpg
import time
from rich import print

from volsurfs_py.viewer.orbit_camera import OrbitCamera
from volsurfs_py.renderers.base_renderer import BaseRenderer
from mvdatasets import MVDataset


class Viewer:

    def __init__(
        self,
        renderer: BaseRenderer,
        mv_data: MVDataset,
        radius=1.0,
        fovy=45.0,
        # continuous_update=True,
        profiler=None,
    ):
        """
        TODO
        """
        print("\ninitializing viewer")

        # storing renderer reference
        self.renderer = renderer

        # dataset (assuming all cameras have the same resolution)
        self.mv_data = mv_data
        camera = self.mv_data[list(self.mv_data.data.keys())[0]][0]
        self.width, self.height = camera.width, camera.height
        print("width", self.width, "height", self.height)

        # cameras
        self.orbit_camera = OrbitCamera(
            width=self.width, height=self.height, r=radius, fovy=fovy
        )

        # profiler
        self.profiler = profiler
        self.frame_time = 1.0

        # set active camera
        self.active_camera = self.orbit_camera
        # self.active_camera = camera

        # self.is_using_mv_data_dims = is_using_mv_data_dims
        # self.is_showing_mv_camera_gt = False

        # create render buffer
        self.render_buffer = np.zeros((self.height, self.width, 3), dtype=np.float32)
        print("render buffer shape", self.render_buffer.shape)

        # shaders available
        self.renders_shaders = ["rgb", "alpha", "normals", "uvs", "view_dirs", "is_hit"]

        # utils
        # self.continuous_update = continuous_update  # continuous rendering
        # self.need_update = True  # when state changes, update frame

    def run(self):
        dpg.create_context()
        self.register_dpg()
        self.render()

    def __del__(self):
        dpg.destroy_context()

    def register_dpg(self):

        # register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.width,
                self.height,
                self.render_buffer,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        # register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.width, height=self.height):
            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=250, height=200):
            # button theme
            with dpg.theme():
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # stats
            with dpg.group(horizontal=True):
                dpg.add_text("rays generation time: ")
                dpg.add_text("no data", tag="_ray_gen_time")

            with dpg.group(horizontal=True):
                dpg.add_text("render time: ")
                dpg.add_text("no data", tag="_render_time")

            with dpg.group(horizontal=True):
                dpg.add_text("FPS: ")
                dpg.add_text("no data", tag="_frame_time")

            # options
            with dpg.group():  # dpg.collapsing_header(label="Options", default_open=True):

                # shader selection
                def callback_change_mode(sender, app_data):
                    self.renderer.active_shader = app_data
                    # self.need_update = True
                    print(f"[INFO] rendering mode: {app_data}")

                dpg.add_combo(
                    self.renders_shaders,
                    label="shader",
                    default_value=self.renderer.active_shader,
                    callback=callback_change_mode,
                )

                # # which hit should be rendered selection
                # def callback_change_which_hit(sender, app_data):
                #     if app_data == "first":
                #         self.renderer.renders_options["render_second_hit"] = False
                #     else:
                #         self.renderer.renders_options["render_second_hit"] = True
                #     self.need_update = True
                #     print(f"[INFO] showing {app_data} hit")

                # if "render_second_hit" in self.renderer.renders_options:
                #     dpg.add_combo(
                #         ("first", "second"),
                #         label="which hit",
                #         default_value="first",
                #         callback=callback_change_which_hit,
                #     )

                # which camera
                def callback_change_which_camera(sender, app_data):

                    if app_data == "orbit_camera":
                        self.active_camera = self.orbit_camera
                        # TODO: disable gt visualization
                        # TODO: disable error computation
                        # TODO: enable fov slider

                    else:
                        camera_name = app_data.split(" ")[0]
                        split, cam_idx = camera_name.split("_")
                        self.active_camera = self.mv_data[split][int(cam_idx)]
                        # TODO: disable fov slider
                        # TODO: enable error computation
                        # TODO: allow gt visualization

                    # self.need_update = True
                    print(f"[INFO] camera {app_data} selected")

                # camera selection list
                available_cameras = ["orbit_camera"]
                if self.mv_data is not None:
                    if "test" in self.mv_data.data:
                        available_cameras += [
                            f"test_{i} ({camera.camera_idx})"
                            for i, camera in enumerate(self.mv_data["test"])
                        ]
                    if "train" in self.mv_data.data:
                        available_cameras += [
                            f"train_{i} ({camera.camera_idx})"
                            for i, camera in enumerate(self.mv_data["train"])
                        ]

                dpg.add_combo(
                    available_cameras,
                    label="which camera",
                    default_value=available_cameras[0],
                    callback=callback_change_which_camera,
                )

                # fov selection
                def callback_set_fovy(sender, app_data):
                    self.orbit_camera.fovy = app_data
                    # self.need_update = True

                dpg.add_slider_int(
                    label="vfov",
                    min_value=15,
                    max_value=120,
                    format="%d deg",
                    default_value=int(self.orbit_camera.fovy),
                    callback=callback_set_fovy,
                )

                # # mesh selection slider
                # def callback_set_mesh_id(sender, app_data):
                #     self.renderer.mesh_id = app_data
                #     self.need_update = True

                # dpg.add_slider_int(
                #     label="which mesh",
                #     min_value=0,
                #     max_value=len(self.renderer.tensor_meshes)-1,
                #     format="%d",
                #     default_value=self.renderer.mesh_id,
                #     callback=callback_set_mesh_id,
                # )

        # register camera handler

        def callback_camera_drag_rotate(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.orbit_camera.orbit(dx, dy)
            # self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.orbit_camera.scale(delta)
            # self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.orbit_camera.pan(dx, dy)
            # self.need_update = True

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="viewer", width=self.width, height=self.height, resizable=False
        )

        # global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)
        dpg.setup_dearpygui()
        # dpg.show_metrics()
        dpg.show_viewport()

    def draw(self):

        #
        # if self.need_update:

        # start frame time
        frame_time_start = time.time()

        # render
        renders = self.renderer.render(self.active_camera, verbose=False)
        # print(renders)

        active_render_mode = self.renderer.active_render_mode
        active_shader_key = self.renderer.active_shader

        # # TODO: reactivate which hit for specific methods
        # if "render_second_hit" in self.renderer.renders_options:
        #     if self.renderer.renders_options["render_second_hit"]:
        #         active_shader_key = "sh_" + active_shader_key

        render = renders[active_render_mode][active_shader_key]
        # TODO: if more than one render per shader is available, allow selection

        # if render.ndim == 3:
        #     render = render[:, 0, :]
        # # print("render shape", render.shape)
        # render = render.reshape(self.height, self.width, -1)

        # scale to [0, 1] if needed
        if np.max(render) > 1:
            render = render / np.max(render)

        # repeat third channel 3 times
        if render.shape[-1] == 1:
            render = np.repeat(render, 3, axis=-1)

        # end frame time
        frame_time_end = time.time()

        self.frame_time = frame_time_end - frame_time_start

        # # if continuos update is disabled, set need_update to False
        # if not self.continuous_update:
        #     self.need_update = False

        # TODO: update list of available shaders
        # if len(self.renders_shaders) == 0:
        #     self.renders_shaders = list(renders[active_render_mode].keys())

        # update gui render buffer
        self.render_buffer = render

    def render(self):
        while dpg.is_dearpygui_running():

            # draw on render buffer
            self.draw()

            # update gui texture with render buffer
            dpg.set_value("_texture", self.render_buffer)

            # collect stats from renderer profiler
            stats = {}
            if self.renderer.profiler is not None:

                ray_gen_time = self.renderer.profiler.get_last_time("ray_gen") * 1000
                # avg_ray_gen_time = profiler.get_avg_time("ray_gen") * 1000
                stats["ray_gen_time"] = f"{ray_gen_time:.4f}ms"

                render_time = (
                    self.renderer.profiler.get_last_time("render_frame") * 1000
                )
                # avg_render_time = profiler.get_avg_time("render_frame") * 1000
                stats["render_time"] = f"{render_time:.4f}ms"

                stats["frame_time"] = f"{int(1/self.frame_time)} FPS"

            # update gui stats
            for k, v in stats.items():
                dpg.set_value(f"_{k}", v)

            dpg.render_dearpygui_frame()
