import torch
from volsurfs_py.utils.sdf_utils import sdf_loss_sphere
from volsurfs_py.utils.fields_utils import get_field_gradients


def loss_l2(gt, pred, mask=None):
    if mask is not None:
        loss = ((gt - pred) ** 2 * mask).mean()
    else:
        loss = ((gt - pred) ** 2).mean()
    return loss


def loss_l1(gt, pred, mask=None):
    if mask is not None:
        loss = ((gt - pred).abs() * mask).mean()
    else:
        loss = (gt - pred).abs().mean()
    return loss


def sparsity_loss(densities, lambda_sparsity=1.0):
    loss_sparsity = (1 - torch.exp(-lambda_sparsity * densities)).mean()
    loss_sparsity = torch.clamp(loss_sparsity, min=0.0)
    return loss_sparsity


def eikonal_loss(sdf_gradients, distance_scale=1.0):
    gradient_error = (
        (torch.linalg.norm(sdf_gradients, ord=2, dim=-1)) - distance_scale
    ) ** 2
    gradient_error = gradient_error.mean()
    return gradient_error


def entropy_loss(values):
    entropy = -values * torch.log(values + 1e-6) - (1 - values) * torch.log(
        1 - values + 1e-6
    )
    loss = entropy.mean()
    return loss


# TODO: update with best one
# def mask_loss(gt_mask, pred_mask):
#     # loss = torch.nn.functional.binary_cross_entropy(pred_mask, gt_mask)
#     # loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_mask, gt_mask)
#     loss = torch.nn.functional.binary_cross_entropy(pred_mask.clip(1e-3, 1.0 - 1e-3), gt_mask)
#     return loss


def sphere_init_loss(nr_points, bounding_primitive, model_sdf, iter_nr):
    points = bounding_primitive.get_random_points_inside(nr_points=nr_points)
    sdf, _ = model_sdf(points, iter_nr)
    sdf_gradients = get_field_gradients(model_sdf.forward, points, iter_nr)
    loss, loss_sdf, gradient_error = sdf_loss_sphere(
        points,
        sdf,
        sdf_gradients,
        scene_radius=bounding_primitive.get_radius() / 2.0,
        sphere_center=[0, 0, 0],
        distance_scale=1.0,
    )
    return loss, loss_sdf, gradient_error


# def loss_multi_spheres_init(dataset_name, nr_points, nr_sdfs, aabb, model, iter_nr):
#     points = aabb.get_random_points_inside(nr_points=nr_points)
#     sdfs, feat = model.forward(points, iter_nr)
#     sdfs_gradients = model.get_sdfs_gradients(points, iter_nr)
#     sdfs_gradients = sdfs_gradients.view(-1, 3)  # [N * K, 3]

#     # TODO: spheres sizes depending on dataset_name
#     scene_radius = 0.3
#     spheres_radius = np.flip(np.linspace(scene_radius - 1e-2, scene_radius, nr_sdfs)).copy()
#     # spheres_radius = np.flip(np.linspace(0.2, aabb.m_radius - 0.1, nr_sdfs)).copy()
#     spheres_center = [0, 0, 0]

#     points_in_sphere_coord = points - torch.as_tensor(spheres_center)
#     point_dist_to_center = points_in_sphere_coord.norm(dim=-1, keepdim=True)

#     distance_scale = 1.0
#     dists = (point_dist_to_center - torch.as_tensor(spheres_radius)) * distance_scale

#     loss_sdf = ((sdfs - dists) ** 2).mean()
#     eikonal_loss = ((sdfs_gradients.norm(dim=-1) - distance_scale) ** 2).mean()
#     loss = loss_sdf * 3e3 + eikonal_loss * 5e1

#     return loss, loss_sdf, eikonal_loss
