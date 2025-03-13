#include "volsurfs/PyBridge.h"

#include "torch/torch.h"
#include <torch/extension.h>

// my stuff
// #include "volsurfs/Sphere.cuh"
#include "volsurfs/OccupancyGrid.cuh"
#include "volsurfs/VolumeRendering.cuh"
#include "volsurfs/RaySampler.cuh"
#include "volsurfs/RaySamplesPacked.cuh"

// https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
// PYBIND11_MAKE_OPAQUE(std::vector<int>); //to be able to pass vectors by reference to functions and have things like push back actually work
// PYBIND11_MAKE_OPAQUE(std::vector<float>, std::allocator<float> >);

namespace py = pybind11;

PYBIND11_MODULE(volsurfs, m)
{
    // py::class_<Sphere>(m, "Sphere")
    //     .def(py::init<const float, const Eigen::Vector3f>())
    //     .def("ray_intersection", &Sphere::ray_intersection)
    //     .def("get_random_points_inside", &Sphere::get_random_points_inside, py::arg("nr_points"))
    //     .def("check_points_inside", &Sphere::check_points_inside)
    //     .def("check_points_inside_with_padding", &Sphere::check_points_inside_with_padding)
    //     .def("get_radius", &Sphere::get_radius)
    //     .def("get_center", &Sphere::get_center)
    //     .def_readwrite("m_center_tensor", &Sphere::m_center_tensor)
    //     .def_readwrite("m_center", &Sphere::m_center)
    //     .def_readwrite("m_radius", &Sphere::m_radius);

    py::class_<OccupancyGrid>(m, "OccupancyGrid")
        .def(py::init<const int, const Eigen::Vector3f>())
        .def_static("make_grid_values", &OccupancyGrid::make_grid_values)
        .def_static("make_grid_occupancy", &OccupancyGrid::make_grid_occupancy)
        // .def("xyz_to_morton_indices", &OccupancyGrid::xyz_to_morton_indices)
        // .def("linear_to_morton_indices", &OccupancyGrid::linear_to_morton_indices)
        .def("set_grid_values", &OccupancyGrid::set_grid_values)
        .def("set_grid_occupancy", &OccupancyGrid::set_grid_occupancy)
        .def("set_grid_occupancy_full", &OccupancyGrid::set_grid_occupancy_full)
        .def("set_grid_occupancy_empty", &OccupancyGrid::set_grid_occupancy_empty)
        .def("init_sphere_roi", &OccupancyGrid::init_sphere_roi)
        .def("get_grid_values", &OccupancyGrid::get_grid_values)
        .def("get_grid_occupancy", &OccupancyGrid::get_grid_occupancy)
        .def("get_grid_roi", &OccupancyGrid::get_grid_roi)
        .def("get_grid_occupancy_in_roi", &OccupancyGrid::get_grid_occupancy_in_roi)
        .def("get_nr_voxels", &OccupancyGrid::get_nr_voxels)
        .def("get_grid_extent", &OccupancyGrid::get_grid_extent)
        .def("get_nr_voxels_per_dim", &OccupancyGrid::get_nr_voxels_per_dim)
        .def("get_nr_occupied_voxels", &OccupancyGrid::get_nr_occupied_voxels)
        .def("get_nr_voxels_in_roi", &OccupancyGrid::get_nr_voxels_in_roi)
        .def("get_nr_occupied_voxels_in_roi", &OccupancyGrid::get_nr_occupied_voxels_in_roi)
        .def("get_grid_max_value", &OccupancyGrid::get_grid_max_value)
        .def("get_grid_min_value", &OccupancyGrid::get_grid_min_value)
        .def("get_grid_max_value_in_roi", &OccupancyGrid::get_grid_max_value_in_roi)
        .def("get_grid_min_value_in_roi", &OccupancyGrid::get_grid_min_value_in_roi)
        .def("get_grid_lower_left_voxels_vertices", &OccupancyGrid::get_grid_lower_left_voxels_vertices)
        .def("get_grid_samples", &OccupancyGrid::get_grid_samples)
        .def("get_random_grid_samples", &OccupancyGrid::get_random_grid_samples)
        .def("get_random_grid_samples_in_roi", &OccupancyGrid::get_random_grid_samples_in_roi)
        .def("check_occupancy", &OccupancyGrid::check_occupancy)
        .def("update_grid_values", &OccupancyGrid::update_grid_values)
        .def("update_grid_occupancy_with_density_values", &OccupancyGrid::update_grid_occupancy_with_density_values)
        .def("update_grid_occupancy_with_sdf_values", &OccupancyGrid::update_grid_occupancy_with_sdf_values)
        .def("get_rays_t_near_t_far", &OccupancyGrid::get_rays_t_near_t_far)
        .def("get_first_rays_sample_start_of_grid_occupied_regions", &OccupancyGrid::get_first_rays_sample_start_of_grid_occupied_regions)
        .def("advance_ray_sample_to_next_occupied_voxel", &OccupancyGrid::advance_ray_sample_to_next_occupied_voxel);

    py::class_<RaySamplesPacked>(m, "RaySamplesPacked")
        .def(py::init<const int, const int, const int, const int>())
        .def("compact_to_valid_samples", &RaySamplesPacked::compact_to_valid_samples)
        // .def("filter_samples_in_range", &RaySamplesPacked::filter_samples_in_range)
        .def("get_nr_rays", &RaySamplesPacked::get_nr_rays)
        .def("get_ray_max_dt", &RaySamplesPacked::get_ray_max_dt)
        .def("get_samples_values", &RaySamplesPacked::get_samples_values)
        .def("get_ray_samples_idx", &RaySamplesPacked::get_ray_samples_idx)
        .def("get_ray_samples_3d", &RaySamplesPacked::get_ray_samples_3d)
        .def("get_ray_samples_dirs", &RaySamplesPacked::get_ray_samples_dirs)
        .def("get_ray_samples_z", &RaySamplesPacked::get_ray_samples_z)
        .def("get_ray_samples_dt", &RaySamplesPacked::get_ray_samples_dt)
        .def("get_ray_samples_values", &RaySamplesPacked::get_ray_samples_values)
        .def("get_ray_start_end_idx", &RaySamplesPacked::get_ray_start_end_idx)
        .def("get_ray_o", &RaySamplesPacked::get_ray_o)
        .def("get_ray_d", &RaySamplesPacked::get_ray_d)
        .def("get_ray_enter", &RaySamplesPacked::get_ray_enter)
        .def("get_ray_exit", &RaySamplesPacked::get_ray_exit)
        .def("is_empty", &RaySamplesPacked::is_empty)
        .def("copy", &RaySamplesPacked::copy)
        .def("get_values_dim", &RaySamplesPacked::get_values_dim)
        .def("get_max_nr_samples", &RaySamplesPacked::get_max_nr_samples)
        .def("get_nr_samples_per_ray", &RaySamplesPacked::get_nr_samples_per_ray)
        .def("get_total_nr_samples", &RaySamplesPacked::get_total_nr_samples)
        .def("get_values_dim", &RaySamplesPacked::get_values_dim)
        .def("set_samples_values", &RaySamplesPacked::set_samples_values)
        .def("remove_samples_values", &RaySamplesPacked::remove_samples_values)
        .def("are_samples_values_set", &RaySamplesPacked::are_samples_values_set)
        .def("update_dt", &RaySamplesPacked::update_dt)
        // .def("update_dt_in_grid_occupied_regions", &RaySamplesPacked::update_dt_in_grid_occupied_regions)
        // .def_static("compute_per_sample_ray_idx", &RaySamplesPacked::compute_per_sample_ray_idx)
        .def_readwrite("samples_idx", &RaySamplesPacked::samples_idx)
        .def_readwrite("samples_3d", &RaySamplesPacked::samples_3d)
        .def_readwrite("samples_dirs", &RaySamplesPacked::samples_dirs)
        .def_readwrite("samples_z", &RaySamplesPacked::samples_z)
        .def_readwrite("samples_dt", &RaySamplesPacked::samples_dt)
        .def_readwrite("samples_values", &RaySamplesPacked::samples_values)
        .def_readwrite("ray_o", &RaySamplesPacked::ray_o)
        .def_readwrite("ray_d", &RaySamplesPacked::ray_d)
        .def_readwrite("ray_enter", &RaySamplesPacked::ray_enter)
        .def_readwrite("ray_exit", &RaySamplesPacked::ray_exit)
        .def_readwrite("ray_start_end_idx", &RaySamplesPacked::ray_start_end_idx);

    py::class_<VolumeRendering>(m, "VolumeRendering")
        .def(py::init<>())
        .def_static("cumprod_one_minus_alpha_to_transmittance", &VolumeRendering::cumprod_one_minus_alpha_to_transmittance)
        .def_static("integrate_with_weights_1d", &VolumeRendering::integrate_with_weights_1d)
        .def_static("integrate_with_weights_3d", &VolumeRendering::integrate_with_weights_3d)
        .def_static("sdf2alpha", &VolumeRendering::sdf2alpha)
        .def_static("sum_over_rays", &VolumeRendering::sum_over_rays)
        .def_static("median_depth_over_rays", &VolumeRendering::median_depth_over_rays)
        .def_static("cumsum_over_rays", &VolumeRendering::cumsum_over_rays)
        .def_static("compute_cdf", &VolumeRendering::compute_cdf)
        .def_static("importance_sample", &VolumeRendering::importance_sample)
        .def_static("combine_ray_samples_packets", &VolumeRendering::combine_ray_samples_packets)
        // backward passes
        .def_static("cumprod_one_minus_alpha_to_transmittance_backward", &VolumeRendering::cumprod_one_minus_alpha_to_transmittance_backward)
        .def_static("integrate_with_weights_1d_backward", &VolumeRendering::integrate_with_weights_1d_backward)
        .def_static("integrate_with_weights_3d_backward", &VolumeRendering::integrate_with_weights_3d_backward)
        .def_static("sum_over_rays_backward", &VolumeRendering::sum_over_rays_backward);

    py::class_<RaySampler>(m, "RaySampler")
        .def(py::init<>())
        .def_static("compute_samples_fg", &RaySampler::compute_samples_fg)
        .def_static("compute_samples_fg_in_grid_occupied_regions", &RaySampler::compute_samples_fg_in_grid_occupied_regions)
        .def_static("compute_samples_bg", &RaySampler::compute_samples_bg)
        .def_static("init_with_one_sample_per_ray", &RaySampler::init_with_one_sample_per_ray)
        .def_static("contract_samples", &RaySampler::contract_samples)
        .def_static("uncontract_samples", &RaySampler::uncontract_samples);
}
