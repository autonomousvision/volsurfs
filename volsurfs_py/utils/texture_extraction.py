import torch
from tqdm import tqdm
import math
import xatlas
from copy import deepcopy
import open3d as o3d
import numpy as np
import torch.nn.functional as F


def barycentric_coordinates(points, triangle):
    """Returns the barycentric coordinates of 2D points relative to a 2D triangle

    Args:
        points (torch.tensor): (N, 2) tensor of points
        triangle (torch.tensor): (3, 2) tensor of triangle vertices

    Returns:
        bar_coords (torch.tensor): (N, 3) tensor of barycentric coordinates
        is_inside (torch.tensor): (N,) tensor of booleans indicating whether the point is inside the triangle
    """

    # Extract vertices of the triangle
    p1, p2, p3 = triangle[0], triangle[1], triangle[2]

    # Calculate vectors for the edges of the triangle
    v0 = (p3 - p1).unsqueeze(0).expand(points.shape[0], -1)
    v1 = (p2 - p1).unsqueeze(0).expand(points.shape[0], -1)
    v2 = points - p1

    # Calculate dot products and cross product
    dot00 = torch.sum(v0 * v0, dim=1)
    dot01 = torch.sum(v0 * v1, dim=1)
    dot02 = torch.sum(v0 * v2, dim=1)
    dot11 = torch.sum(v1 * v1, dim=1)
    dot12 = torch.sum(v1 * v2, dim=1)

    # Calculate barycentric coordinates
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    bar_coords = torch.zeros(
        (points.shape[0], 3), dtype=points.dtype, device=points.device
    )

    # new
    bar_coords[:, 2] = (dot11 * dot02 - dot01 * dot12) * inv_denom  # u
    bar_coords[:, 1] = (dot00 * dot12 - dot01 * dot02) * inv_denom  # v
    bar_coords[:, 0] = 1 - bar_coords[:, 1] - bar_coords[:, 2]  # w

    is_inside = torch.all(bar_coords >= 0, dim=1)
    is_inside &= torch.abs(torch.sum(bar_coords, dim=1) - 1) < 1e-6

    return bar_coords, is_inside


@torch.no_grad()
def extract_texture_from_color_model(
    appearance_model,
    vertices,
    faces,
    uvs,
    texture_res=512,
    nr_samples_per_texel=12,
    export_rgb=False,
    iter_nr=None,
):
    """Extracts a texture from a color model given a mesh with uv atlas

    Iterate over all 2d triangles in the atlas, sample points inside the triangle,
    compute 3d points from barycentric coordinates and 3d vertices,
    get model output for 3d points,
    write output to texture.

    Args:
        appearance_model (nn.Model): _description_
        vertices (torch.tensor): _description_
        faces (torch.tensor): _description_
        uvs (torch.tensor): _description_
        texture_res (int, optional): _description_. Defaults to 512.
        nr_samples_per_texel (int, optional): How many random samples are generated for each texel. Defaults to 12.
        iter_nr (int, optional): _description_. Defaults to None.

    Returns:
        texture (torch.tensor): _description_
    """

    # get model out dimension
    if export_rgb:
        out_dims = 3
    else:
        out_dims = appearance_model.out_channels

    # create texture
    texture = torch.zeros(
        (texture_res, texture_res, out_dims), dtype=torch.float32, device=uvs.device
    )

    # iterate over all triangles
    # visualize_every = 10000
    pbar = tqdm(
        range(faces.shape[0]),
        desc="Extracting texture",
        unit="iter",
        # ncols=100,
        leave=False,
    )
    for triangle_idx in pbar:

        # get 2d triangle coordinates
        triangle_2d = uvs[faces[triangle_idx]]

        # get 3d triangle coordinates
        triangle_3d = vertices[faces[triangle_idx]]

        # get triangle 2d bounding box
        min_x, max_x = torch.min(triangle_2d[:, 0]), torch.max(triangle_2d[:, 0])
        min_y, max_y = torch.min(triangle_2d[:, 1]), torch.max(triangle_2d[:, 1])
        min_x *= texture_res
        max_x *= texture_res
        min_y *= texture_res
        max_y *= texture_res
        min_x, max_x = math.floor(min_x), math.ceil(max_x)
        min_y, max_y = math.floor(min_y), math.ceil(max_y)

        # iterate over all texels in the bounding box
        # get texel centers
        texel_indices = torch.stack(
            torch.meshgrid(
                torch.arange(min_x, max_x), torch.arange(min_y, max_y), indexing="ij"
            ),
            dim=-1,
        )
        texel_centers = texel_indices.float() / texture_res
        texel_centers += 1 / (2 * texture_res)
        u, v = texel_centers[..., 0].flatten(), texel_centers[..., 1].flatten()
        texel_centers = torch.stack([u, v], dim=-1)  # (N, 2)
        texel_indices = texel_indices.reshape(-1, 2)  # (N, 2)

        # supersampling (multiple samples per texel, one in the center and the rest jittered around it)
        if nr_samples_per_texel > 1:
            # repeat interleaved
            texel_centers = texel_centers.repeat_interleave(
                nr_samples_per_texel, dim=0
            )  # (N * S, 2)

            # add random jitter
            jitter = (torch.rand_like(texel_centers) - 0.5 - 1e-6) / texture_res
            # don't jitter first sample, so that it is always in the center
            mask = torch.ones(nr_samples_per_texel).bool()
            mask[0] = False
            mask = mask.repeat(texel_indices.shape[0])
            texel_centers[mask] += jitter[mask]

        points_2d = texel_centers
        bar_coords, is_inside = barycentric_coordinates(points_2d, triangle_2d)
        # bar_coords is (N * S, 3)
        # is_inside is (N * S,)

        # use 2d_points barycentric coords to compute 3d points
        points_3d = bar_coords.unsqueeze(-1) * triangle_3d.unsqueeze(0)
        points_3d = torch.sum(points_3d, dim=1)

        # check if there is at least one point
        if points_3d.shape[0] == 0:
            continue

        # get triangle normal
        e1 = triangle_3d[1] - triangle_3d[0]
        e2 = triangle_3d[2] - triangle_3d[0]
        normal = torch.cross(e1, e2)
        normal = F.normalize(normal, p=2, dim=-1)
        normal = normal.unsqueeze(0).repeat(points_3d.shape[0], 1)
        # print("normal", normal.shape)

        if export_rgb:
            vals = appearance_model(
                points_3d, samples_dirs=-normal, normals=normal, iter_nr=iter_nr
            )
        else:
            # get model output for 3d points
            vals = appearance_model(
                points_3d, samples_dirs=None, normals=normal, iter_nr=iter_nr
            )
        # vals is (N * S, out_dims)

        # average same texel samples
        if nr_samples_per_texel > 1:
            # only average samples inside the triangle
            mask = is_inside.reshape(-1, nr_samples_per_texel, 1)
            vals = vals.reshape(-1, nr_samples_per_texel, out_dims)
            vals = vals * mask
            vals_sum = torch.sum(vals, dim=1)
            mask_sum = torch.sum(mask, dim=1)
            # this can generate nans if all samples of a texel are outside the triangle
            vals = vals_sum / mask_sum
            # nans are filtered out by is_inside
            is_inside = is_inside.reshape(-1, nr_samples_per_texel, 1)
            is_inside = is_inside.any(dim=1).squeeze()

        # write output to texture
        # if only_write_when_inside:
        texel_indices_x = texel_indices[is_inside, 0]
        texel_indices_y = texel_indices[is_inside, 1]
        texture[texel_indices_x, texel_indices_y] = vals[is_inside]

    return texture


@torch.no_grad()
def extract_textures(
    mesh_idx, method, texture_res=512, nr_samples_per_texel=12, iter_nr=None
):
    print(f"extracting texture for mesh {mesh_idx}")

    print("vertices", method.tensor_meshes[mesh_idx].vertices.shape)
    print("faces", method.tensor_meshes[mesh_idx].faces.shape)

    if method.tensor_meshes[mesh_idx] is None:
        print("\n[bold red]ERROR[/bold red]: mesh UVs not found")
        exit(1)
    else:
        print("uvs", method.tensor_meshes[mesh_idx].vertices_uvs.shape)

    # extract transparency texture
    if method.models[f"alpha_{mesh_idx}"] is not None:
        texture_alpha = extract_texture_from_color_model(
            method.models[f"alpha_{mesh_idx}"],
            method.tensor_meshes[mesh_idx].vertices,
            method.tensor_meshes[mesh_idx].faces,
            method.tensor_meshes[mesh_idx].vertices_uvs,
            texture_res=texture_res,
            nr_samples_per_texel=nr_samples_per_texel,
            iter_nr=iter_nr,
        )
    else:
        texture_alpha = None

    # extract rgb texture
    texture_rgb = extract_texture_from_color_model(
        method.models[f"rgb_{mesh_idx}"],
        method.tensor_meshes[mesh_idx].vertices,
        method.tensor_meshes[mesh_idx].faces,
        method.tensor_meshes[mesh_idx].vertices_uvs,
        texture_res=texture_res,
        nr_samples_per_texel=nr_samples_per_texel,
        iter_nr=iter_nr,
    )

    # texture_alpha shape is (texture_res, texture_res, 1 * nr_sh_coeffs)
    # texture_rgb shape is (texture_res, texture_res, nr_channels * nr_sh_coeffs)

    if texture_alpha is not None:
        # concat rgb and alpha
        texture = torch.cat([texture_rgb, texture_alpha], dim=2)
    else:
        texture = texture_rgb

    texture_rgb = texture

    # TODO: test new texture extraction (separating rgb and alpha)
    return texture_rgb, None


def compute_o3d_mesh_atlas(mesh, bilinear=False, padding=5, verbose=False):
    """Compute uv atlas for mesh

    Args:
        mesh (o3d.geometry.TriangleMesh): Mesh to compute uv atlas for

    Returns:
        new_mesh (o3d.geometry.TriangleMesh): Mesh with uv atlas
        atlas_img (np.ndarray): Atlas image
    """

    # compute uv atlas

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    mesh.compute_vertex_normals()
    vertex_normals = np.asarray(mesh.vertex_normals)

    if verbose:
        print("computing atlas for mesh with:")
        print("vertices", vertices.shape)
        print("faces", faces.shape)
        print("vertex_normals", vertex_normals.shape)

    # compute atlas
    atlas = xatlas.Atlas()

    # TODO: add padding around charts, run a dilate filter on the baked texture

    pack_options = xatlas.PackOptions()
    pack_options.create_image = True
    pack_options.bilinear = bilinear
    pack_options.padding = padding

    atlas.add_mesh(vertices, faces, vertex_normals)
    atlas.generate(pack_options=pack_options)

    if verbose:
        print("atlas.width, atlas.height", atlas.width, atlas.height)
        print("atlas.utilization", atlas.utilization)

    # The parametrization potentially duplicates vertices.
    # `vmapping` contains the original vertex index for each new vertex (shape N, type uint32).
    # `indices` contains the vertex indices of the new triangles (shape Fx3, type uint32)
    # `uvs` contains texture coordinates of the new vertices (shape Nx2, type float32)
    vmapping, indices, uvs = atlas.get_mesh(0)

    # convert vertices uvs to triangle uvs (3 * num_faces, 2)
    uvs = uvs[indices.flatten()]

    # save new mesh with uvs
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(vertices[vmapping])
    new_mesh.triangles = o3d.utility.Vector3iVector(indices)
    # new_mesh.compute_vertex_normals()
    new_mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)

    atlas_img = atlas.chart_image

    return new_mesh, atlas_img


def dilate_texture(img, nr_iterations):
    """Dilate texture by filling empty pixels with the color of their occupied neighbors

    Args:
        img (np.ndarray): texture image
        nr_iterations (int, optional): Number of dilation iterations. Stops early when no empty pixels are left.

    Returns:
        dilated_img: dilated texture
    """
    print("\ndilating texture of shape", img.shape)

    # copy image
    img_dilated = deepcopy(img)

    # pixel neighbors offsets
    neighbors_offsets = np.array(
        [
            -1,
            -1,
            0,
            -1,
            1,
            -1,
            -1,
            0,
            1,
            0,
            -1,
            1,
            0,
            1,
            1,
            1,
        ]
    ).reshape(8, 2)

    last_iter = 0
    # get occupied pixels
    full_pixels = np.stack(np.where((img_dilated != 0).all(axis=2))).T[:, :2]
    for i in range(nr_iterations):
        print(f"iteration {i}")

        # get neighbor pixels
        full_pixels_repeated = full_pixels.repeat(8, axis=0)
        neightbors_offsets_repeated = np.tile(neighbors_offsets, (len(full_pixels), 1))
        neighbor_pixels = full_pixels_repeated + neightbors_offsets_repeated

        # filter out pixels outside the image
        mask_1 = np.logical_and(
            neighbor_pixels[:, 0] >= 0, neighbor_pixels[:, 0] < img_dilated.shape[0]
        )
        mask_2 = np.logical_and(
            neighbor_pixels[:, 1] >= 0, neighbor_pixels[:, 1] < img_dilated.shape[1]
        )
        mask = np.logical_and(mask_1, mask_2)
        neighbor_pixels = neighbor_pixels[mask]
        full_pixels_repeated = full_pixels_repeated[mask]

        # remove duplicates
        _, unique_idxs = np.unique(neighbor_pixels, axis=0, return_index=True)
        neighbor_pixels = neighbor_pixels[unique_idxs]
        full_pixels_repeated = full_pixels_repeated[unique_idxs]

        # check which neighbors are empty
        mask = img_dilated[neighbor_pixels[:, 0], neighbor_pixels[:, 1]] == 0
        mask = mask.all(axis=1)
        dest_pixels = neighbor_pixels[mask]
        src_pixels = full_pixels_repeated[mask]

        last_iter = i + 1

        if dest_pixels.shape[0] == 0:
            print("no empty pixels left to fill")
            break

        img_dilated[dest_pixels[:, 0], dest_pixels[:, 1]] = img_dilated[
            src_pixels[:, 0], src_pixels[:, 1]
        ]
        # new full pixels are the ones that were empty and got filled
        full_pixels = dest_pixels

    return img_dilated
