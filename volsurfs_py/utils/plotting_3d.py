from volsurfs_py.utils.mesh_extraction import extract_o3d_mesh_from_fn
import open3d as o3d
import numpy as np
import open3d.visualization as vis


def view_3d_sdf(
    sdf_fn,
    nr_points_per_dim,
    bounding_primitive=None,
    width=500,
    height=500,
    mesh_show_wireframe=False,
):

    o3d_mesh = extract_o3d_mesh_from_fn(
        fn=sdf_fn,
        nr_points_per_dim=nr_points_per_dim,
        level_set=0.0,
        bounding_primitive=bounding_primitive,
    )
    o3d_mesh.paint_uniform_color(np.random.rand(3, 1))

    # use open3d to visualize the mesh
    vis.draw_geometries(
        [o3d_mesh],
        width=width,
        height=height,
        mesh_show_wireframe=mesh_show_wireframe,
        mesh_show_back_face=True,
    )


def view_3d_sdfs(
    sdfs_fn,
    nr_sdfs,
    nr_points_per_dim,
    bounding_primitive=None,
    width=500,
    height=500,
    mesh_show_wireframe=True,
):

    o3d_meshes = []
    for nr_sdf in range(nr_sdfs):
        o3d_mesh = extract_o3d_mesh_from_fn(
            fn=sdfs_fn,
            out_idx=nr_sdf,
            nr_points_per_dim=nr_points_per_dim,
            level_set=0.0,
            bounding_primitive=bounding_primitive,
        )
        # assign a random color to the mesh

        o3d_mesh.paint_uniform_color(np.random.rand(3, 1))
        o3d_meshes.append(o3d_mesh)

    # use open3d to visualize the mesh
    vis.draw_geometries(
        o3d_meshes,
        width=width,
        height=height,
        mesh_show_wireframe=mesh_show_wireframe,
        mesh_show_back_face=True,
    )
