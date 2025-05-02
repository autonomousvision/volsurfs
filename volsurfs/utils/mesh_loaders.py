import os
from rich import print
from mvdatasets.utils.mesh import Mesh


# def get_mesh_file_from_filename(mesh_filename, meshes_filenames):
#     # get mesh filename from meshes_filenames
#     mesh_filename = ""
#     # mesh_isolevel_ = str(float(mesh_filename))  # make sure it is a string with a dot
#     for filename in meshes_filenames:
#         if mesh_filename == filename.split("_")[-1].replace(
#             ".obj", ""
#         ).replace(".ply", ""):
#             mesh_filename = filename
#     if mesh_filename == "":
#         print(f"\n[bold red]ERROR[/bold red]: mesh for {mesh_isolevel_} not found")
#         exit(1)

#     return mesh_filename


def get_meshes_filenames_in_path(meshes_path):
    meshes_filenames = []
    print(f"\nmeshes found in {meshes_path}")
    for filename in os.listdir(meshes_path):
        if filename.endswith(".obj") or filename.endswith(".ply"):
            meshes_filenames.append(filename)
    meshes_filenames = sorted(
        meshes_filenames, key=lambda x: float(x.replace(".obj", "").replace(".ply", ""))
    )
    return meshes_filenames


def load_meshes_indexed_from_path(
    meshes_indices, meshes_path, require_uvs=False, return_paths=False
):
    """handles meshes loading and initialization
    args:
        meshes_filenames: list of meshes filenames (without extension) to load (e.g. ["-0.01", "0.0", "0.01"] or ["1", "2", "3"])
        meshes_path: path to the meshes
    """

    # make sure the path exists
    if not os.path.exists(meshes_path):
        print(f"\n[bold red]ERROR[/bold red]: mesh file {meshes_path} does not exist")
        exit(1)

    # get files in meshes_path
    meshes_filenames_in_path = get_meshes_filenames_in_path(meshes_path)
    print(f"meshes_filenames_in_path: {meshes_filenames_in_path}")

    nr_meshes_in_path = len(meshes_filenames_in_path)

    if nr_meshes_in_path == 0:
        print(f"\n[bold red]ERROR[/bold red]: no meshes found in {meshes_path}")
        exit(1)

    if meshes_indices is not None:

        # select meshes in path based on indices

        # validate meshes_filenames
        if len(meshes_indices) == 0:
            print("\n[bold red]ERROR[/bold red]: no meshes indices set in config file")
            exit(1)

        # mesh_indices_int = [int(meshes_index) + nr_meshes_in_path//2 for meshes_index in meshes_indices]
        mesh_indices_int = [int(meshes_index) for meshes_index in meshes_indices]
        mesh_indices_int.sort()
        print(f"mesh_indices: {mesh_indices_int}")

        selected_meshes_filenames = []
        for mesh_index in mesh_indices_int:
            if mesh_index < 0 or mesh_index >= nr_meshes_in_path:
                print(
                    f"\n[bold red]ERROR[/bold red]: mesh index {mesh_index} out of range"
                )
                exit(1)
            selected_meshes_filenames.append(meshes_filenames_in_path[mesh_index])

    else:

        # select all
        selected_meshes_filenames = meshes_filenames_in_path

    print(f"selected_meshes_filenames: {selected_meshes_filenames}")

    # load the mesh and add it to the scene
    meshes = []
    meshes_paths = []

    for i, mesh_filename in enumerate(selected_meshes_filenames):
        # load mesh
        mesh_path = os.path.join(meshes_path, mesh_filename)
        meshes_paths.append(mesh_path)
        print(f"loading mesh {i+1}/{len(selected_meshes_filenames)} from {mesh_path}")
        mesh = Mesh(mesh_meta={"mesh_path": mesh_path, "textures": []})

        if require_uvs and not mesh.has_uvs:
            print(
                f"\n[bold red]ERROR[/bold red]: mesh {mesh_filename} does not have UVs"
            )
            exit(1)

        meshes.append(mesh)

    if return_paths:
        return meshes, meshes_paths

    return meshes
