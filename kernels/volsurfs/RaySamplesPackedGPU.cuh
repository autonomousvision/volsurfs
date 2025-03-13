#pragma once

#include <torch/torch.h>
#include "volsurfs/helper_math.h"

// import pos_to_lin_idx and distance_to_next_voxel
#include "volsurfs/occ_grid_helpers.h"

#define BLOCK_SIZE 256

namespace RaySamplesPackedGPU
{

    __global__ void
    update_dt_gpu(
        const int nr_rays,
        const int max_nr_samples,
        // input
        const bool is_background,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_max_dt,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_t_exit,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_z,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> ray_start_end_idx,
        // output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_samples_dt
    ) {
        // each thread will deal with a new value
        int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (ray_idx >= nr_rays)
        {
            return;
        }

        // get the indexes where this ray starts and end
        int idx_start = ray_start_end_idx[ray_idx][0];
        int idx_end = ray_start_end_idx[ray_idx][1];
        int nr_samples = idx_end - idx_start;

        if (nr_samples == 0)
            return;

        float max_dt = ray_max_dt[ray_idx][0];

        // nr_samples == 1: do not enter for loop
        // nr_samples >= 2: enter for loop

        for (int i = 0; i <= nr_samples - 2; i++)
        {
            int cur_sample_idx = idx_start + i;
            if (cur_sample_idx >= max_nr_samples)
            {
                printf("cur_sample_idx is out of bounds");
                return;
            }

            int next_sample_idx = idx_start + i + 1;
            if (next_sample_idx >= max_nr_samples)
            {
                printf("next_sample_idx is out of bounds");
                return;
            }

            float cur_t = samples_z[cur_sample_idx][0];
            float next_t = samples_z[next_sample_idx][0];
            float dt = clamp(next_t - cur_t, 0.0, max_dt);
            out_samples_dt[cur_sample_idx][0] = dt;
        }
        // last sample
        int last_sample_idx = idx_start + nr_samples - 1;
        if (last_sample_idx >= max_nr_samples)
        {
            printf("last_sample_idx is out of bounds");
            return;
        }

        if (is_background) {
            out_samples_dt[last_sample_idx][0] = 1e10;
        }
        else{
            float t_exit = ray_t_exit[ray_idx][0];
            float last_sample_t = samples_z[last_sample_idx][0];
            float remaining_dist_until_border = t_exit - last_sample_t;
            float last_sample_dt = clamp(remaining_dist_until_border, 0.0, max_dt);
            out_samples_dt[last_sample_idx][0] = last_sample_dt;
        }
    }

    // __global__ void
    // update_dt_in_grid_occupied_regions_gpu(
    //     const int nr_rays,
    //     // input
    //     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_max_dt,
    //     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_o,
    //     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_d,
    //     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_t_exit,
    //     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_z,
    //     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_3d,
    //     const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> ray_start_end_idx,
    //     // occupancy grid
    //     const int nr_voxels_per_dim,
    //     const Eigen::Vector3f grid_extent,
    //     const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> grid_occupancy_tensor,
    //     const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> grid_roi_tensor,
    //     // output
    //     torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_samples_dt
    // ) {
    //     // each thread will deal with a new value
    //     int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //     // don't go out of bounds
    //     if (idx >= nr_rays)
    //     {
    //         return;
    //     }

    //     // get the indexes where this ray starts and end
    //     int idx_start = ray_start_end_idx[idx][0];
    //     int idx_end = ray_start_end_idx[idx][1];
    //     int nr_samples = idx_end - idx_start;

    //     for (int i = 0; i < nr_samples - 1; i++)
    //     {   
    //         float cur_t = samples_z[idx_start + i][0];
    //         float cur_x = samples_3d[idx_start + i][0];
    //         float cur_y = samples_3d[idx_start + i][1];
    //         float cur_z = samples_3d[idx_start + i][2];
    //         float3 cur_pos = make_float3(cur_x, cur_y, cur_z);
    //         int cur_idx_voxel = pos_to_lin_idx(cur_pos, nr_voxels_per_dim, grid_extent);
            
    //         float next_t = samples_z[idx_start + i + 1][0];
    //         float next_x = samples_3d[idx_start + i + 1][0];
    //         float next_y = samples_3d[idx_start + i + 1][1];
    //         float next_z = samples_3d[idx_start + i + 1][2];
    //         float3 next_pos = make_float3(next_x, next_y, next_z);
    //         int next_idx_voxel = pos_to_lin_idx(next_pos, nr_voxels_per_dim, grid_extent);

    //         float dt = next_t - cur_t;

    //         if (dt < ray_max_dt) 
    //         {
    //             // keep dt as is
    //         }
    //         else 
    //         {
                
    //         }

    //         // if we are in an occupied voxel, accumulate the distance that we traversed through occupied space
    //         if (grid_roi_tensor[cur_idx_voxel] && grid_occupancy_tensor[cur_idx_voxel])
    //         {
    //             dt = clamp(next_t - cur_t, 0.0, ray_max_dt);
    //         }
    //         else 
    //         {
    //             // if a sample is empty space, set dt to 0
    //             dt = 0.0f;
    //         }
            
    //         out_samples_dt[idx_start + i][0] = dt;
    //         // ray_max_dt = max(ray_max_dt, dt);
    //     }
    //     // last sample
    //     float t_exit = ray_t_exit[idx][0];
    //     float last_sample_t = samples_z[idx_start + nr_samples - 1][0];
    //     float remaining_dist_until_border = t_exit - last_sample_t;
    //     float last_sample_dt = clamp(remaining_dist_until_border, 0.0, ray_max_dt);
    //     out_samples_dt[idx_start + nr_samples - 1][0] = last_sample_dt;
    // }

    __global__ void
    compact_to_valid_samples_gpu(
        const int nr_rays,
        const int max_nr_samples_input,
        const int max_nr_samples_output,
        const int values_dim,
        // input
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> samples_idx,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_3d,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_dirs,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_z,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_dt,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_values,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> ray_start_end_idx,
        // 
        const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> out_indices_start,
        // output
        torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> out_samples_idx,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_samples_3d,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_samples_dirs,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_samples_z,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_samples_dt,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_samples_values,
        torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> out_ray_start_end_idx
    ) {
        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_rays)
        { 
            return;
        }

        // get the indexes where this ray starts and end
        int in_idx_start = ray_start_end_idx[idx][0];
        int in_idx_end = ray_start_end_idx[idx][1];
        int nr_samples = in_idx_end - in_idx_start;
        
        if (nr_samples == 0)
            return;

        int out_idx_start = out_indices_start[idx];
        
        // copy all
        for (int i = 0; i < nr_samples; i++)
        {

            if (in_idx_start + i >= max_nr_samples_input)
            {
                printf("we are reading outside of bounds!");
                return;
            }

            if (out_idx_start + i >= max_nr_samples_output)
            {
                printf("we are writing outside of bounds!");
                return;
            }

            // samples_idx
            out_samples_idx[out_idx_start + i][0] = samples_idx[in_idx_start + i][0];
            // samples_3d
            out_samples_3d[out_idx_start + i][0] = samples_3d[in_idx_start + i][0];
            out_samples_3d[out_idx_start + i][1] = samples_3d[in_idx_start + i][1];
            out_samples_3d[out_idx_start + i][2] = samples_3d[in_idx_start + i][2];
            // samples_dirs
            out_samples_dirs[out_idx_start + i][0] = samples_dirs[in_idx_start + i][0];
            out_samples_dirs[out_idx_start + i][1] = samples_dirs[in_idx_start + i][1];
            out_samples_dirs[out_idx_start + i][2] = samples_dirs[in_idx_start + i][2];
            // samples_z
            out_samples_z[out_idx_start + i][0] = samples_z[in_idx_start + i][0];
            // samples_dt
            out_samples_dt[out_idx_start + i][0] = samples_dt[in_idx_start + i][0];
            
            // samples_values
            for (int j = 0; j < values_dim; j++)
            {
                out_samples_values[out_idx_start + i][j] = samples_values[in_idx_start + i][j];
            }
        }

        // rays with samples
        out_ray_start_end_idx[idx][0] = out_idx_start;
        out_ray_start_end_idx[idx][1] = out_idx_start + nr_samples;
    }

    // __global__ void filter_samples_in_range_gpu(
    //     const int nr_rays,
    //     const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> samples_idx,
    //     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_3d,
    //     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_dirs,
    //     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_z,
    //     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_dt,
    //     const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> ray_start_end_idx,
    //     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_o,
    //     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_d,
    //     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_enter,
    //     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_exit,
    //     //
    //     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> t_near,
    //     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> t_far,
    //     // output
    //     torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> out_samples_idx,
    //     torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_samples_3d,
    //     torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_samples_dirs,
    //     torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_samples_z,
    //     torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_samples_dt,
    //     torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> out_ray_start_end_idx,
    //     torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_ray_o,
    //     torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_ray_d,
    //     torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_ray_enter,
    //     torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_ray_exit
    // ) {
    //     int idx = blockIdx.x * blockDim.x + threadIdx.x; // each thread will deal with a new value

    //     if (idx >= nr_rays)
    //     { // don't go out of bounds
    //         return;
    //     }

    //     // get the indexes where this ray starts and end
    //     int in_idx_start = ray_start_end_idx[idx][0];
    //     int in_idx_end = ray_start_end_idx[idx][1];
    //     int nr_samples = in_idx_end - in_idx_start;

    //     // retrieve ray specific t- interesection values from tensor
    //     float t_near_val = t_near[idx][0];
    //     float t_far_val = t_far[idx][0];

    //     // calculate real number of samples to copy
    //     int nr_samples_to_copy = 0;
    //     for (int i = 0; i < nr_samples; i++)
    //     {
    //         // obtain z-value of current sample and check if it is within intersection bounds
    //         float sample_z = samples_z[in_idx_start + i][0];
    //         if (sample_z >= t_near_val && sample_z <= t_far_val)
    //         {
    //             nr_samples_to_copy += 1;
    //         }
    //     }

    //     int out_idx_start = in_idx_start;

    //     // copy all the samples
    //     int nr_samples_copied = 0;
    //     for (int i = 0; i < nr_samples; i++)
    //     {
    //         // obtain z-value of current sample and check if it is within intersection bounds
    //         float sample_z = samples_z[in_idx_start + i][0];
            
    //         if (sample_z >= t_near_val && sample_z <= t_far_val)
    //         {
    //             // samples_idx
    //             out_samples_idx[out_idx_start + nr_samples_copied][0] = samples_idx[in_idx_start + i][0];
    //             // samples_3d
    //             out_samples_3d[out_idx_start + nr_samples_copied][0] = samples_3d[in_idx_start + i][0];
    //             out_samples_3d[out_idx_start + nr_samples_copied][1] = samples_3d[in_idx_start + i][1];
    //             out_samples_3d[out_idx_start + nr_samples_copied][2] = samples_3d[in_idx_start + i][2];
    //             // samples_dirs
    //             out_samples_dirs[out_idx_start + nr_samples_copied][0] = samples_dirs[in_idx_start + i][0];
    //             out_samples_dirs[out_idx_start + nr_samples_copied][1] = samples_dirs[in_idx_start + i][1];
    //             out_samples_dirs[out_idx_start + nr_samples_copied][2] = samples_dirs[in_idx_start + i][2];
    //             // samples_z
    //             out_samples_z[out_idx_start + nr_samples_copied][0] = sample_z;
    //             // samples_dt
    //             out_samples_dt[out_idx_start + nr_samples_copied][0] = samples_dt[in_idx_start + i][0];
    //             nr_samples_copied += 1;
    //         }
    //     }

    //     // ray specific values
    //     out_ray_o[idx][0] = ray_o[idx][0];
    //     out_ray_o[idx][1] = ray_o[idx][1];
    //     out_ray_o[idx][2] = ray_o[idx][2];

    //     out_ray_d[idx][0] = ray_d[idx][0];
    //     out_ray_d[idx][1] = ray_d[idx][1];
    //     out_ray_d[idx][2] = ray_d[idx][2];

    //     // ray enter
    //     out_ray_enter[idx][0] = t_near[idx][0];
    //     // ray exit
    //     out_ray_exit[idx][0] = t_far[idx][0];

    //     // per ray values
    //     out_ray_start_end_idx[idx][0] = out_idx_start;
    //     out_ray_start_end_idx[idx][1] = out_idx_start + nr_samples_to_copy;

    //     // zero out remaining positions
    //     for (int i = nr_samples_copied; i < nr_samples_to_copy; i++) {
    //         // samples_idx
    //         out_samples_idx[out_idx_start + i][0] = -1;
    //         // samples_3d
    //         out_samples_3d[out_idx_start + i][0] = 0;
    //         out_samples_3d[out_idx_start + i][1] = 0;
    //         out_samples_3d[out_idx_start + i][2] = 0;
    //         // samples_dirs
    //         out_samples_dirs[out_idx_start + i][0] = 0;
    //         out_samples_dirs[out_idx_start + i][1] = 0;
    //         out_samples_dirs[out_idx_start + i][2] = 0;
    //         // samples_z
    //         out_samples_z[out_idx_start + i][0] = -1; // just a sentinel value that we can easily detect in the volumetric rendering and discard these samples
    //         // samples_dt
    //         out_samples_dt[out_idx_start + i][0] = 0;
    //     }
    // }

    // __global__ void
    // compute_per_sample_ray_idx_gpu(
    //     const int nr_rays,
    //     const int max_nr_samples,
    //     const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> ray_start_end_idx,
    //     // output
    //     torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> per_sample_ray_idx)
    // {

    //     int idx = blockIdx.x * blockDim.x + threadIdx.x; // each thread will deal with a new value

    //     if (idx >= nr_rays)
    //     { // don't go out of bounds
    //         return;
    //     }

    //     // get the indexes where this ray starts and end
    //     int in_idx_start = ray_start_end_idx[idx][0];
    //     int in_idx_end = ray_start_end_idx[idx][1];
    //     int nr_samples = in_idx_end - in_idx_start;

    //     if (nr_samples > 0)
    //     {
    //         for (int i = 0; i < nr_samples; i++)
    //         {
    //             per_sample_ray_idx[in_idx_start + i] = idx;
    //             if (in_idx_start + i >= nr_samples)
    //             {
    //                 printf("we are writing outside of bounds!");
    //             }
    //         }
    //     }
    // }

} // namespace
