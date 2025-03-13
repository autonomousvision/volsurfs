#pragma once

#include "torch/torch.h"
#include <Eigen/Core>

#include "volsurfs/pcg32.h"
#include "volsurfs/RaySamplesPacked.cuh"

class RaySampler
{
public:
    RaySampler();
    ~RaySampler();

    static RaySamplesPacked compute_samples_bg(
        const torch::Tensor &rays_o,
        const torch::Tensor &rays_d,
        const torch::Tensor &ray_t_exit,
        const float ray_t_far,
        const int nr_samples,
        const bool jitter_samples
    );
    static RaySamplesPacked compute_samples_fg(
        const torch::Tensor &rays_o,
        const torch::Tensor &rays_d,
        const torch::Tensor &ray_t_entry,
        const torch::Tensor &ray_t_exit,
        const float min_dist_between_samples,
        const int min_nr_samples_per_ray,
        const int max_nr_samples_per_ray,
        const bool jitter_samples,
        const int values_dim
    );
    static RaySamplesPacked compute_samples_fg_in_grid_occupied_regions(
        const torch::Tensor &rays_o,
        const torch::Tensor &rays_d,
        const torch::Tensor &ray_t_entry,
        const torch::Tensor &ray_t_exit,
        const float min_dist_between_samples,
        const int min_nr_samples_per_ray,
        const int max_nr_samples_per_ray,
        const bool jitter_samples,
        const int nr_voxels_per_dim,
        const Eigen::Vector3f grid_extent,
        const torch::Tensor &grid_occupancy,
        const torch::Tensor &grid_roi,
        const int values_dim
    );
    static RaySamplesPacked init_with_one_sample_per_ray(
        const torch::Tensor samples_3d,
        const torch::Tensor samples_dirs
    );
    static RaySamplesPacked contract_samples(
        const RaySamplesPacked &uncontracted_ray_samples_packed
    );
    static RaySamplesPacked uncontract_samples(
        const RaySamplesPacked &contracted_ray_samples_packed
    );

    static pcg32 m_rng;
    static const bool verbose = false;

private:
};
