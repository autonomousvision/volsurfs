#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

from rich import print
import torch
import numpy as np
import os
import math
from tqdm import tqdm
from functools import partial
import open3d as o3d
import trimesh
from skimage import measure
import imageio
import json
import re

from mvdatasets.geometry.contraction import uncontract_points


# modified from here https://github.com/autonomousvision/sdfstudio/blob/370902a10dbef08cb3fe4391bd3ed1e227b5c165/nerfstudio/utils/marching_cubes.py#L201
def marching_cubes_with_contraction(
    sdf,
    resolution=512,
    bounding_box_min=(-1.0, -1.0, -1.0),
    bounding_box_max=(1.0, 1.0, 1.0),
    return_mesh=False,
    level=0,
    simplify_mesh=True,
    inv_contraction=None,
    max_range=32.0,
):
    assert resolution % 512 == 0

    resN = resolution
    cropN = 512
    level = 0
    N = resN // cropN

    grid_min = bounding_box_min
    grid_max = bounding_box_max
    xs = np.linspace(grid_min[0], grid_max[0], N + 1)
    ys = np.linspace(grid_min[1], grid_max[1], N + 1)
    zs = np.linspace(grid_min[2], grid_max[2], N + 1)

    meshes = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                print(i, j, k)
                x_min, x_max = xs[i], xs[i + 1]
                y_min, y_max = ys[j], ys[j + 1]
                z_min, z_max = zs[k], zs[k + 1]

                x = torch.linspace(x_min, x_max, cropN).cuda()
                y = torch.linspace(y_min, y_max, cropN).cuda()
                z = torch.linspace(z_min, z_max, cropN).cuda()

                xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
                points = (
                    torch.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T.float().cuda()
                )

                @torch.no_grad()
                def evaluate(points):
                    z = []
                    for _, pnts in enumerate(torch.split(points, 256**3, dim=0)):
                        z.append(sdf(pnts))
                    z = torch.cat(z, axis=0)
                    return z

                # construct point pyramids
                points = points.reshape(cropN, cropN, cropN, 3)
                points = points.reshape(-1, 3)
                pts_sdf = evaluate(points.contiguous())
                z = pts_sdf.detach().cpu().numpy()
                if not (np.min(z) > level or np.max(z) < level):
                    z = z.astype(np.float32)
                    verts, faces, normals, _ = measure.marching_cubes(
                        volume=z.reshape(cropN, cropN, cropN),
                        level=level,
                        spacing=(
                            (x_max - x_min) / (cropN - 1),
                            (y_max - y_min) / (cropN - 1),
                            (z_max - z_min) / (cropN - 1),
                        ),
                    )
                    verts = verts + np.array([x_min, y_min, z_min])
                    meshcrop = trimesh.Trimesh(verts, faces, normals)
                    meshes.append(meshcrop)

                print("finished one block")

    if len(meshes) == 0:
        # print error and exit
        print("[bold red]ERROR[/bold red]: no mesh extracted")
        exit(1)

    combined = trimesh.util.concatenate(meshes)
    combined.merge_vertices(digits_vertex=6)

    # inverse contraction and clipping the points range
    if inv_contraction is not None:
        combined.vertices = (
            inv_contraction(torch.from_numpy(combined.vertices).float().cuda())
            .cpu()
            .numpy()
        )
        combined.vertices = np.clip(combined.vertices, -max_range, max_range)

    return combined


def to_cam_open3d(c2ws, intrinsics, width, height):
    camera_traj, full_proj_transform = [], []
    for i, (c2w, intrins) in enumerate(zip(c2ws, intrinsics)):

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            cx=intrins[0, 2].item(),
            cy=intrins[1, 2].item(),
            fx=intrins[0, 0].item(),
            fy=intrins[1, 1].item(),
        )

        w2c = np.linalg.inv(c2w)
        extrinsic = np.asarray(w2c)
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

        fov_x, fov_y = intrinsic_to_fov(intrins)
        intrins_expand = getProjectionMatrix(0.1, 100, fov_x, fov_y)

        full_proj_transform.append(torch.from_numpy(intrins_expand @ w2c))

    return camera_traj, torch.stack(full_proj_transform).float()


class MeshExtractor(object):
    def __init__(self, depthmaps, rgbs, c2ws, intrinsics, with_vertex_colors=False):
        """ """

        self.depthmaps = torch.stack(depthmaps)
        self.rgbmaps = torch.stack(rgbs)

        height, width = rgbs[0].shape[-2:]
        self.with_vertex_colors = with_vertex_colors
        self.cam_o3d, self.full_proj_transform = to_cam_open3d(
            c2ws, intrinsics, width, height
        )

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.

        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f"voxel_size: {voxel_size}")
        print(f"sdf_trunc: {sdf_trunc}")
        print(f"depth_truc: {depth_trunc}")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        for i, cam_o3d in tqdm(
            enumerate(self.cam_o3d), desc="TSDF integration progress"
        ):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(
                    np.asarray(
                        rgb.permute(1, 2, 0).numpy() * 255, order="C", dtype=np.uint8
                    )
                ),  # hw3 255
                o3d.geometry.Image(
                    np.asarray(depth.permute(1, 2, 0).cpu().numpy(), order="C")
                ),  # hw1
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
                depth_scale=1.0,
            )

            volume.integrate(
                rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic
            )

        mesh = volume.extract_triangle_mesh()
        return mesh

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=512):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets.
        return o3d.mesh
        """

        def compute_sdf_perframe(i, points, depthmap, rgbmap, full_proj_transform):
            """
            compute per frame sdf
            """
            new_points = torch.cat(
                [points, torch.ones_like(points[..., :1])], dim=-1
            ) @ full_proj_transform.t().to(points.device)
            z = new_points[..., -1:]
            pix_coords = new_points[..., :2] / new_points[..., -1:]
            mask_proj = ((pix_coords > -1.0) & (pix_coords < 1.0) & (z > 0)).all(dim=-1)

            sampled_depth = torch.nn.functional.grid_sample(
                depthmap.cuda()[None],
                pix_coords[None, None],
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).reshape(-1, 1)
            sampled_rgb = (
                torch.nn.functional.grid_sample(
                    rgbmap.cuda()[None],
                    pix_coords[None, None],
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                .reshape(3, -1)
                .T
            )
            sdf = sampled_depth - z
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(
            samples, inv_contraction, voxel_size, return_rgb=False
        ):
            """
            Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                # mask = torch.linalg.norm(samples, dim=-1) > 1
                # # adaptive sdf_truncation
                # sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                # sdf_trunc[mask] *= 1/(2-torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
                sdf_trunc = 5 * voxel_size
                # samples = inv_contraction(samples)
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:, 0]) * 1
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:, 0])
            for i, full_proj_transform in tqdm(
                enumerate(self.full_proj_transform), desc="TSDF integration progress"
            ):

                sdf, rgb, mask_proj = compute_sdf_perframe(
                    i,
                    samples,
                    depthmap=self.depthmaps[i],
                    rgbmap=self.rgbmaps[i],
                    full_proj_transform=full_proj_transform,
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:, None] + rgb[mask_proj]) / wp[
                    :, None
                ]
                # update weight
                weights[mask_proj] = wp

            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        inv_contraction = lambda x: uncontract_points(x)

        uncontraded_radius = 1.0
        N = resolution
        voxel_size = uncontraded_radius * 2 / N
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(
                -uncontraded_radius,
                -uncontraded_radius,
                -uncontraded_radius,
            ),
            bounding_box_max=(
                uncontraded_radius,
                uncontraded_radius,
                uncontraded_radius,
            ),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )

        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d

        if self.with_vertex_colors:
            print("texturing mesh ... ")
            _, rgbs = compute_unbounded_tsdf(
                torch.tensor(np.asarray(mesh.vertices)).float().cuda(),
                inv_contraction=None,
                voxel_size=voxel_size,
                return_rgb=True,
            )
            mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())

        return mesh


def getProjectionMatrix(znear, zfar, fovX, fovY):

    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = np.zeros((4, 4))

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def intrinsic_to_fov(K, w=None, h=None):
    # Extract the focal lengths from the intrinsic matrix
    fx = K[0, 0]
    fy = K[1, 1]

    w = K[0, 2] * 2 if w is None else w
    h = K[1, 2] * 2 if h is None else h

    # Calculating field of view
    fov_x = 2 * np.arctan2(w, 2 * fx)
    fov_y = 2 * np.arctan2(h, 2 * fy)

    return fov_x, fov_y


#################################################################################3


def read_pfm(filename):
    file = open(filename, "rb")
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode("utf-8").rstrip()
    if header == "PF":
        color = True
    elif header == "Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def load_scene(path):

    json_info = json.load(open(os.path.join(path, f"transforms.json")))
    b2c = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    scene_info = {"ixts": [], "c2ws": [], "w2cs": [], "rgbs": [], "depthes": []}

    for idx, frame in enumerate(json_info["frames"]):

        c2w = np.array(frame["transform_matrix"])
        c2w = c2w @ b2c

        ixt = np.array(frame["intrinsic_matrix"])

        scene_info["ixts"].append(ixt.astype(np.float32))
        scene_info["c2ws"].append(c2w.astype(np.float32))
        scene_info["w2cs"].append(np.linalg.inv(c2w.astype(np.float32)))

        img_path = os.path.join(path, f"r_{idx:03d}.png")
        depth_path = os.path.join(path, f"depth/r_{idx:03d}.pfm")

        img = imageio.imread(img_path)
        img = img.astype(np.float32) / 255.0
        img = (img[..., :3] * img[..., -1:] + (1 - img[..., -1:])).astype(np.float32)
        depth, _ = read_pfm(depth_path)

        scene_info["rgbs"].append(torch.from_numpy(img).permute(2, 0, 1))
        scene_info["depthes"].append(torch.from_numpy(depth.copy())[None])

    return scene_info


# def post_process_mesh(mesh, cluster_to_keep=1000):
#     """
#     Post-process a mesh to filter out floaters and disconnected parts
#     """
#     import copy
#     print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
#     mesh_0 = copy.deepcopy(mesh)
#     with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
#             triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

#     triangle_clusters = np.asarray(triangle_clusters)
#     cluster_n_triangles = np.asarray(cluster_n_triangles)
#     cluster_area = np.asarray(cluster_area)
#     n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
#     n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
#     triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
#     mesh_0.remove_triangles_by_mask(triangles_to_remove)
#     mesh_0.remove_unreferenced_vertices()
#     mesh_0.remove_degenerate_triangles()
#     print("num vertices raw {}".format(len(mesh.vertices)))
#     print("num vertices post {}".format(len(mesh_0.vertices)))
#     return mesh_0
