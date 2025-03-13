import torch


@torch.no_grad()
def intersect_bounding_primitive(bounding_primitive, rays_o, rays_d):
    """
    Args:
        bounding_privitive (mvdataset.BoundingBox or mvdataset.BoundingSphere)
        rays_o (torch.tensor): (N, 3) in world space
        rays_d (torch.tensor): (N, 3) in world space
    Out:
        is_hit (torch.tensor): (N,) boolean
        t_near (torch.tensor): (N, 1) distance to near intersection
        t_far (torch.tensor): (N, 1) distance to far intersection
        p_near (torch.tensor): (N, 3) intersection point near
        p_far (torch.tensor): (N, 3) intersection point far
    """
    nr_rays = rays_o.shape[0]

    (is_hit, t_near, t_far, p_near, p_far) = bounding_primitive.intersect(
        rays_o, rays_d
    )

    t_near = t_near.unsqueeze(-1)
    t_far = t_far.unsqueeze(-1)

    return {
        "rays_o": rays_o,
        "rays_d": rays_d,
        "nr_rays": nr_rays,
        "points_near": p_near,
        "points_far": p_far,
        "t_near": t_near,
        "t_far": t_far,
        "is_hit": is_hit,
    }


@torch.no_grad()
def reflect_rays(rays_dirs, normals_dirs):
    """
    Args:
        rays_dirs (torch.tensor): (N, 3)
        normals_dirs (torch.tensor): (N, 3)
    Out:
        reflected_dirs (torch.tensor): (N, 3)
    """
    # make sure the input is normalized
    # rays_dirs = rays_dirs / torch.norm(rays_dirs, dim=-1, keepdim=True)
    # normals_dirs = normals_dirs / torch.norm(normals_dirs, dim=-1, keepdim=True)

    # r = d - 2(d . n)n

    # samples_dirs = 2*(torch.sum(eye_dirs * normals_dirs, dim=1, keepdim=True))*normals_dirs - eye_dirs

    reflected_dirs = (
        rays_dirs
        - 2 * torch.sum(rays_dirs * normals_dirs, dim=-1, keepdim=True) * normals_dirs
    )
    return reflected_dirs
