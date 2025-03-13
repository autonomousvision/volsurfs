import torch
import open3d as o3d
from skimage import measure
import math
import numpy as np
import os
import pymeshlab
from tqdm import tqdm

from volsurfs_py.methods.surf import Surf
from volsurfs_py.methods.nerf import NeRF
from volsurfs_py.methods.offsets_surfs import OffsetsSurfs
from mvdatasets.geometry.primitives.bounding_sphere import BoundingSphere
from mvdatasets.geometry.primitives.bounding_box import BoundingBox
from mvdatasets.geometry.contraction import contract_points, uncontract_points


def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy

    print(
        "post processing the mesh to have {} clusterscluster_to_kep".format(
            cluster_to_keep
        )
    )
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh_0.cluster_connected_triangles()
        )

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def to_cam_open3d(c2ws, intrinsics, width, height):

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


@torch.no_grad()
def extract_o3d_mesh_bg_fn(
    fn,
    nr_points_per_dim,
    threshold=0,
    iter_nr=None,
    verbose=True,
    contraction_fn=None,
    inv_contraction_fn=None,
):
    scene_radius = 1.0

    if contraction_fn is None:
        contraction_fn = lambda x: x
    if inv_contraction_fn is None:
        inv_contraction_fn = lambda x: x

    min_val = -scene_radius
    max_val = scene_radius

    chunk_size = 64
    X = torch.linspace(
        -scene_radius, scene_radius, nr_points_per_dim, dtype=torch.float32
    )
    Y = torch.linspace(
        -scene_radius, scene_radius, nr_points_per_dim, dtype=torch.float32
    )
    Z = torch.linspace(
        -scene_radius, scene_radius, nr_points_per_dim, dtype=torch.float32
    )

    X = X.split(chunk_size)
    Y = Y.split(chunk_size)
    Z = Z.split(chunk_size)

    out_all = np.zeros(
        [nr_points_per_dim, nr_points_per_dim, nr_points_per_dim], dtype=np.float32
    )
    with tqdm(
        total=len(X) * len(Y) * len(Z), desc="getting regular grid values"
    ) as pbar:
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")

                    pts = torch.cat(
                        [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                        dim=-1,
                    ).cuda()
                    pts_norm = torch.norm(pts, dim=-1).cpu().numpy()

                    density = fn(pts, iter_nr)

                    step_size = 1.0 / 256
                    alpha = 1.0 - torch.exp(-density.view(-1, 1) * step_size)
                    alpha[pts_norm < 0.5] = 0.0

                    out_curr = alpha.reshape(len(xs), len(ys), len(zs))
                    out_curr = out_curr.detach().cpu().numpy()

                    out_all[
                        xi * chunk_size : xi * chunk_size + len(xs),
                        yi * chunk_size : yi * chunk_size + len(ys),
                        zi * chunk_size : zi * chunk_size + len(zs),
                    ] = out_curr

                    pbar.update(1)

    print("output values in range: ", (out_all.min(), out_all.max()))

    # check if the surface is in the volume
    if threshold >= out_all.min() and threshold <= out_all.max():

        print("running marching cubes")
        vertices, faces, _, _ = measure.marching_cubes(out_all, threshold)
        if verbose:
            print("marching cubes result:", vertices.shape, faces.shape)

        # center vertices
        vertices = vertices / (nr_points_per_dim - 1.0) * (max_val - min_val) + min_val
    else:

        print("no intersection with level set")
        vertices = np.zeros((0, 3), dtype=np.float32)
        faces = np.zeros((0, 3), dtype=np.int32)

    # vertices = inv_contraction_fn(vertices)
    # breakpoint()
    # make open3d mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    nr_vertices = vertices.shape[0]

    # # filter out points outside the bounding_primitive
    # if bounding_primitive is not None and nr_vertices > 0:
    #     vertices = torch.tensor(mesh.vertices).cuda().float()
    #     faces = torch.tensor(mesh.triangles).cuda().long()
    #     valid_vertices = bounding_primitive.check_points_inside(vertices).squeeze()
    #     invalid_faces = (
    #         valid_vertices[faces.flatten()].reshape(-1, 3).all(axis=-1).squeeze()
    #     )
    #     valid_faces = ~invalid_faces
    #     triangle_indices = valid_faces.nonzero().squeeze().cpu().numpy()
    #     print(f"filtered out {len(triangle_indices)} faces outside the bounding_primitive")

    #     mesh.remove_triangles_by_index(triangle_indices)
    #     mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()

    return mesh


@torch.no_grad()
def extract_o3d_mesh_from_fn(
    fn,
    nr_points_per_dim,
    bounding_primitive=None,
    out_idx=None,
    level_set=0.0,
    threshold=0,
    occupancy_grid=None,
    iter_nr=None,
    verbose=True,
    contraction_fn=None,
    inv_contraction_fn=None,
):
    if bounding_primitive is not None:
        scene_radius = bounding_primitive.get_radius()
    else:
        scene_radius = 1.0

    if contraction_fn is None:
        contraction_fn = lambda x: x
    if inv_contraction_fn is None:
        inv_contraction_fn = lambda x: x

    min_val = -scene_radius
    max_val = scene_radius

    chunk_size = 64
    X = torch.linspace(
        -scene_radius, scene_radius, nr_points_per_dim, dtype=torch.float32
    )
    Y = torch.linspace(
        -scene_radius, scene_radius, nr_points_per_dim, dtype=torch.float32
    )
    Z = torch.linspace(
        -scene_radius, scene_radius, nr_points_per_dim, dtype=torch.float32
    )

    X = X.split(chunk_size)
    Y = Y.split(chunk_size)
    Z = Z.split(chunk_size)

    out_all = np.zeros(
        [nr_points_per_dim, nr_points_per_dim, nr_points_per_dim], dtype=np.float32
    )
    with tqdm(
        total=len(X) * len(Y) * len(Z), desc="getting regular grid values"
    ) as pbar:
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")

                    pts = torch.cat(
                        [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                        dim=-1,
                    ).cuda()
                    pts = inv_contraction_fn(pts)

                    # not all fn take iter_nr as input
                    if iter_nr is None:
                        pred = fn(pts)  # [(chunk_size), ...]
                    else:
                        pred = fn(pts, iter_nr=iter_nr)  # [(chunk_size), ...]

                    # unpack if tuple
                    if isinstance(pred, tuple):
                        pred = pred[0]

                    # select output if index is given
                    if out_idx is not None:
                        pred = pred[:, out_idx]  # e.g. to select a specific sdf

                    # shift level set
                    pred -= level_set

                    out_curr = pred.reshape(len(xs), len(ys), len(zs))
                    out_curr = out_curr.detach().cpu().numpy()

                    out_all[
                        xi * chunk_size : xi * chunk_size + len(xs),
                        yi * chunk_size : yi * chunk_size + len(ys),
                        zi * chunk_size : zi * chunk_size + len(zs),
                    ] = out_curr

                    pbar.update(1)

    print("output values in range: ", (out_all.min(), out_all.max()))

    # check if the surface is in the volume
    if threshold >= out_all.min() and threshold <= out_all.max():

        print("running marching cubes")
        vertices, faces, _, _ = measure.marching_cubes(out_all, threshold)
        if verbose:
            print("marching cubes result:", vertices.shape, faces.shape)

        # center vertices
        vertices = vertices / (nr_points_per_dim - 1.0) * (max_val - min_val) + min_val

    else:

        print("no intersection with level set")
        vertices = np.zeros((0, 3), dtype=np.float32)
        faces = np.zeros((0, 3), dtype=np.int32)

    # make open3d mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    nr_vertices = vertices.shape[0]

    # if occupancy_grid is not None and nr_vertices > 0:
    #     vertices = torch.tensor(mesh.vertices).cuda().float()
    #     faces = torch.tensor(mesh.triangles).cuda().long()
    #     is_in_occupied_voxel, _ = occupancy_grid.check_occupancy(vertices)
    #     is_in_occupied_voxel = is_in_occupied_voxel.squeeze()
    #     valid_vertices = is_in_occupied_voxel
    #     invalid_faces = (
    #         valid_vertices[faces.flatten()].reshape(-1, 3).all(axis=-1).squeeze()
    #     )
    #     valid_faces = ~invalid_faces
    #     triangle_indices = valid_faces.nonzero().squeeze().cpu().numpy()
    #     print(f"filtered out {len(triangle_indices)} faces outside the occupancy grid")

    #     mesh.remove_triangles_by_index(triangle_indices)
    #     mesh.remove_unreferenced_vertices()
    #     mesh.compute_vertex_normals()

    # filter out points outside the bounding_primitive
    if bounding_primitive is not None and nr_vertices > 0:
        vertices = torch.tensor(mesh.vertices).cuda().float()
        faces = torch.tensor(mesh.triangles).cuda().long()
        valid_vertices = bounding_primitive.check_points_inside(vertices).squeeze()
        invalid_faces = (
            valid_vertices[faces.flatten()].reshape(-1, 3).all(axis=-1).squeeze()
        )
        valid_faces = ~invalid_faces
        triangle_indices = valid_faces.nonzero().squeeze().cpu().numpy()
        print(
            f"filtered out {len(triangle_indices)} faces outside the bounding_primitive"
        )

        mesh.remove_triangles_by_index(triangle_indices)
        mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()

    return mesh


@torch.no_grad()
def extract_o3d_meshes_from_offsets_surfs(
    method, nr_points_per_dim, bounding_primitive, iter_nr=None
):

    assert isinstance(method, OffsetsSurfs), "method must be OffsetsSurfs"

    o3d_meshes = []
    model = method.models["sdfs"]

    for surf_idx in range(method.nr_surfs):

        print(f"\nmesh {surf_idx+1}/{method.nr_surfs}")

        occ_grid = method.occupancy_grid

        o3d_mesh = extract_o3d_mesh_from_fn(
            fn=model,
            nr_points_per_dim=nr_points_per_dim,
            out_idx=surf_idx,
            level_set=0.0,
            bounding_primitive=bounding_primitive,
            occupancy_grid=occ_grid,
            iter_nr=iter_nr,
        )
        print("number of vertices", len(o3d_mesh.vertices))
        print("number of faces", len(o3d_mesh.triangles))

        o3d_meshes.append(o3d_mesh)

    return o3d_meshes


@torch.no_grad()
def extract_o3e_mesh_from_background(
    method, nr_points_per_dim, bounding_primitive, iter_nr=None
):

    bg_model = method.models["bg"]
    o3d_mesh = extract_o3d_mesh_from_fn(
        fn=bg_model,
        nr_points_per_dim=nr_points_per_dim,
        level_set=0.0,
        threshold=0.0,
        bounding_primitive=bounding_primitive,
        iter_nr=iter_nr,
    )

    return o3d_mesh


@torch.no_grad()
def extract_o3d_mesh_from_method(
    method,
    nr_points_per_dim,
    bounding_primitive,
    levelset,
    extract_main_surf=True,
    iter_nr=None,
    is_bg=False,
):

    # TODO: make main surf extraction optional

    if isinstance(method, Surf) or isinstance(method, OffsetsSurfs):
        if not is_bg:
            levelset, threshold = 0.0, levelset

            if isinstance(method, Surf):
                model = method.models["sdf"]
            if isinstance(method, OffsetsSurfs):
                model = method.models["sdfs"].main_sdf

            contraction_fn = lambda x: x
            inv_contraction_fn = lambda x: x

            o3d_mesh = extract_o3d_mesh_from_fn(
                model,
                nr_points_per_dim=nr_points_per_dim,
                level_set=levelset,
                threshold=threshold,
                bounding_primitive=bounding_primitive,
                occupancy_grid=method.occupancy_grid,
                iter_nr=iter_nr,
                contraction_fn=contraction_fn,
                inv_contraction_fn=inv_contraction_fn,
            )
        else:
            model = method.models["bg"].get_only_density
            contraction_fn = contract_points
            inv_contraction_fn = uncontract_points
            o3d_mesh = extract_o3d_mesh_bg_fn(
                model,
                nr_points_per_dim=nr_points_per_dim,
                threshold=levelset,
                iter_nr=iter_nr,
                contraction_fn=contraction_fn,
                inv_contraction_fn=inv_contraction_fn,
            )

    elif isinstance(method, NeRF):
        # NB: not tested yet
        model = method.models["density"]
        o3d_mesh = extract_o3d_mesh_from_fn(
            model,
            nr_points_per_dim=nr_points_per_dim,
            level_set=0.5,
            threshold=levelset,
            bounding_primitive=bounding_primitive,
            occupancy_grid=method.occupancy_grid,
            iter_nr=iter_nr,
        )
    else:
        raise NotImplementedError("Method mesh extraction not implemented")

    return o3d_mesh


def simplify_o3d_mesh(o3d_mesh, target_nr_faces_ratio=0.1, verbose=False):
    """
    TODO
    """

    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)

    # create a new Mesh with the two arrays
    m = pymeshlab.Mesh(vertices, faces)

    nr_faces = m.face_number()

    if verbose:
        print("current number of faces", nr_faces)

    # create a new MeshSet
    ms = pymeshlab.MeshSet()

    # add the mesh to the MeshSet
    ms.add_mesh(m, "mesh")

    nr_target_faces = int(nr_faces * target_nr_faces_ratio)
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=nr_target_faces)

    m_simplified = ms.current_mesh()

    newfacenum = m_simplified.face_number()

    if verbose:
        print("new number of faces", newfacenum)

    # convert back to open3d mesh

    # get numpy arrays of vertices and faces of the current mesh
    vertices_simplified = m_simplified.vertex_matrix()
    faces_simplified = m_simplified.face_matrix()

    # create a new Mesh with the two arrays
    o3d_mesh_simplified = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices_simplified),
        o3d.utility.Vector3iVector(faces_simplified),
    )
    o3d_mesh_simplified.compute_vertex_normals()

    return o3d_mesh_simplified


# def simplify_mesh(mesh_path, target_nr_faces_ratio=0.1):
#     """Loads a mesh from file, simplifies it with MeshLab decimation quadric edge
#     collapse and stores it to file.
#     Simplified mesh is stored in folder "meshes_simplified" in the same folder as the
#     original mesh and with the same name.

#     args:
#         mesh_path (str): path to mesh file
#         target_nr_faces_ratio (float): target number of faces in percentage

#     out:
#         new_mesh_path (str): path to simplified mesh file
#     """

#     assert os.path.exists(mesh_path), "mesh_path does not exist"

#     print(f"loading mesh to simplify from {mesh_path}")

#     ms = pymeshlab.MeshSet()
#     ms.load_new_mesh(mesh_path)

#     facenum = ms.current_mesh().face_number()
#     print("current number of faces", facenum)

#     targetfacenum = int(facenum * target_nr_faces_ratio)
#     ms.meshing_decimation_quadric_edge_collapse(targetfacenum=targetfacenum)

#     newfacenum = ms.current_mesh().face_number()
#     print("new number of faces", newfacenum)

#     # convert back to open3d mesh

#     # save simplified mesh
#     mesh_name = os.path.basename(mesh_path)
#     mesh_folder = os.path.dirname(mesh_path)
#     simplified_mesh_folder = mesh_folder + "_simplified"
#     # create folder if it does not exist
#     os.makedirs(simplified_mesh_folder, exist_ok=True)
#     new_mesh_path = os.path.join(simplified_mesh_folder, mesh_name)
#     ms.save_current_mesh(new_mesh_path)

#     print(f"simplified mesh saved to {new_mesh_path}")

#     return new_mesh_path


def load_o3d_mesh(mesh_path):
    """Loads mesh from file.

    args:
        mesh_path (str): path to mesh file

    out:
        mesh (open3d.geometry.TriangleMesh): loaded mesh
    """
    print(f"loading mesh from {mesh_path}")

    if os.path.exists(mesh_path):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
    else:
        print(f"mesh_path does not exist: {mesh_path}")
        mesh = None

    return mesh


def save_o3d_mesh(mesh, mesh_path, write_uvs=False):
    """Saves mesh to file.

    args:
        mesh (open3d.geometry.TriangleMesh): mesh to save
        mesh_path (str): path to save mesh to
    """

    print(f"saving mesh to {mesh_path}")

    mesh_folder = os.path.dirname(mesh_path)
    # create folder if it does not exist
    os.makedirs(mesh_folder, exist_ok=True)
    o3d.io.write_triangle_mesh(
        mesh_path,
        mesh,
        write_ascii=False,
        compressed=True,
        write_vertex_normals=False,
        write_vertex_colors=False,
        write_triangle_uvs=write_uvs,
        print_progress=True,
    )
