#pragma once

#include "torch/torch.h"
#include <Eigen/Core>

#include "volsurfs/pcg32.h"
#include "volsurfs/RaySamplesPacked.cuh"
// #include "volsurfs/Sphere.cuh"
class OccupancyGrid
{
public:
    OccupancyGrid(const int nr_voxels_per_dim, const Eigen::Vector3f grid_extent);
    ~OccupancyGrid();

    static torch::Tensor make_grid_values(const int nr_voxels_per_dim);
    static torch::Tensor make_grid_occupancy(const int nr_voxels_per_dim);
    torch::Tensor get_grid_values();
    torch::Tensor get_grid_occupancy();
    torch::Tensor get_grid_roi();
    torch::Tensor get_grid_occupancy_in_roi();
    void set_grid_values(const torch::Tensor &grid_values);
    void set_grid_occupancy(const torch::Tensor &grid_occupancy);
    void set_grid_occupancy_full();
    void set_grid_occupancy_empty();
    void init_sphere_roi(const float radius, const float padding); // sets the region of interest to be a sphere
    int get_nr_voxels();
    int get_nr_voxels_per_dim();
    int get_nr_voxels_in_roi();
    int get_nr_occupied_voxels();
    int get_nr_occupied_voxels_in_roi();
    float get_grid_max_value();
    float get_grid_min_value();
    float get_grid_max_value_in_roi();
    float get_grid_min_value_in_roi();
    Eigen::Vector3f get_grid_extent();
    // torch::Tensor xyz_to_morton_indices(const torch::Tensor &xyz_indices);
    // torch::Tensor linear_to_morton_indices(const torch::Tensor &linear_indices);
    
    torch::Tensor get_grid_all_voxels_vertices(const torch::Tensor &ll_vertices);
    std::tuple<torch::Tensor, torch::Tensor> get_grid_lower_left_voxels_vertices();
    std::tuple<torch::Tensor, torch::Tensor> get_grid_samples(const bool jitter_samples);
    std::tuple<torch::Tensor, torch::Tensor> get_random_grid_samples(const int nr_of_voxels_to_select, const bool jitter_samples);
    std::tuple<torch::Tensor, torch::Tensor> get_random_grid_samples_in_roi(const int nr_of_voxels_to_select, const bool jitter_samples);
    std::tuple<torch::Tensor, torch::Tensor> get_rays_t_near_t_far(
        const torch::Tensor &rays_o, const torch::Tensor &rays_d,
        const torch::Tensor &ray_t_entry, const torch::Tensor &ray_t_exit
    );
    std::tuple<torch::Tensor, torch::Tensor> check_occupancy(const torch::Tensor &points);
    RaySamplesPacked get_first_rays_sample_start_of_grid_occupied_regions(const torch::Tensor &rays_o, const torch::Tensor &rays_d, const torch::Tensor &ray_t_entry, const torch::Tensor &ray_t_exit);
    std::tuple<torch::Tensor, torch::Tensor> advance_ray_sample_to_next_occupied_voxel(const torch::Tensor &samples_dirs, const torch::Tensor &samples_3d);

    void update_grid_values(const torch::Tensor &point_indices, const torch::Tensor &values, const float decay);
    void update_grid_occupancy_with_density_values(const torch::Tensor &point_indices, const float occupancy_tresh, const bool check_neighbours);
    void update_grid_occupancy_with_sdf_values(const torch::Tensor &point_indices, const torch::Tensor &logistic_beta, const float occupancy_thresh, const bool check_neighbours);

    int m_nr_voxels_per_dim;
    Eigen::Vector3f m_grid_extent; // lenghts of the cuboid in each dimension
    torch::Tensor m_grid_extent_tensor;

    torch::Tensor m_grid_roi;       // voxels mask, 1 if voxel is in the region of interest, 0 otherwise
    torch::Tensor m_grid_values;    // for each element in the grid,  we store the full value, either density or sdf
    torch::Tensor m_grid_occupancy; // just booleans saying if the voxels are occupied or not

    static pcg32 m_rng;
    static const bool verbose = false;

private:
};
