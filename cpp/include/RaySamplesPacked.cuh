#pragma once

#include "torch/torch.h"
#include <Eigen/Core>


class RaySamplesPacked{
public:
    
    RaySamplesPacked(const int nr_rays, const int max_nr_samples, const int first_sample_idx, const int values_dim);

    RaySamplesPacked compact_to_valid_samples();

    void update_dt(const bool is_background);
    // void update_dt_in_grid_occupied_regions(
    //     const int nr_voxels_per_dim,
    //     const Eigen::Vector3f grid_extent,
    //     const torch::Tensor &grid_occupancy,
    //     const torch::Tensor &grid_roi
    // );
    void set_samples_values(const torch::Tensor& values);
    void remove_samples_values();

    // RaySamplesPacked filter_samples_in_range(
    //     const torch::Tensor &samples_3d,
    //     const torch::Tensor &samples_dirs,
    //     const torch::Tensor &t_near,
    //     const torch::Tensor &t_far
    // );

    RaySamplesPacked copy() const;
    
    int get_nr_rays() const;
    int get_max_nr_samples() const;
    torch::Tensor get_nr_samples_per_ray() const;
    int get_total_nr_samples() const;
    int get_values_dim() const;
    bool is_empty() const;
    bool are_samples_values_set() const;

    torch::Tensor get_samples_values() const;

    torch::Tensor get_ray_samples_idx(int ray_idx) const;
    torch::Tensor get_ray_samples_3d(int ray_idx) const;
    torch::Tensor get_ray_samples_dirs(int ray_idx) const;
    torch::Tensor get_ray_samples_z(int ray_idx) const;
    torch::Tensor get_ray_samples_dt(int ray_idx) const;
    torch::Tensor get_ray_samples_values(int ray_idx) const;
    torch::Tensor get_ray_start_end_idx(int ray_idx) const;
    torch::Tensor get_ray_o(int ray_idx) const;
    torch::Tensor get_ray_d(int ray_idx) const;
    torch::Tensor get_ray_enter(int ray_idx) const;
    torch::Tensor get_ray_exit(int ray_idx) const;
    float get_ray_max_dt(int ray_idx) const;

    // Attributes
    
    // per sample
    torch::Tensor samples_idx; // nr_samples x 1
    torch::Tensor samples_3d; // nr_samples x 3
    torch::Tensor samples_dirs; // nr_samples x 3
    torch::Tensor samples_z; // nr_samples x 1
    torch::Tensor samples_dt; // nr_samples x 1
    torch::Tensor samples_values; // nr_samples x values_dim
    
    // per ray
    torch::Tensor ray_start_end_idx; // nr_rays x 2 for each ray, we store the idx of the first sample and the end sample
    torch::Tensor ray_o; // nr_rays x 3
    torch::Tensor ray_d; // nr_rays x 3
    torch::Tensor ray_enter; // nr_rays x 1
    torch::Tensor ray_exit; // nr_rays x 1

    //if it has sdf it should also be combined properly with another raysamples packed when needed
    bool has_samples_values;
    bool has_dt;
    bool is_compacted;

    torch::Tensor ray_max_dt; // nr_rays x 1

    static const bool verbose = false;
};