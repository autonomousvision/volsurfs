#pragma once

#include <torch/torch.h>
#include "volsurfs/helper_math.h"

// matrices
#include "volsurfs/mat3.h"
#include "volsurfs/mat4.h"

#include "volsurfs/pcg32.h"

// import pos_to_lin_idx and distance_to_next_voxel
#include "volsurfs/occ_grid_helpers.h"

#define BLOCK_SIZE 256

namespace RaySamplerGPU
{
    
    inline constexpr __device__ uint32_t MAX_STEPS() { return 2048 * 2; } // finest number of steps per unit length

    __device__ float clamp_min(float x, float a)
    {
        return max(a, x);
    }

    __device__ float clamp_gpu(float x, float a, float b)
    {
        return max(a, min(b, x));
    }

    __device__ float map_range_gpu(const float input, const float input_start, const float input_end, const float output_start, const float output_end)
    {
        // we clamp the input between the start and the end
        float input_clamped = clamp_gpu(input, input_start, input_end);
        return output_start + ((output_end - output_start) / (input_end - input_start)) * (input_clamped - input_start);
    }

    __global__ void
    compute_samples_bg_gpu(
        const int nr_rays,
        const int nr_samples_per_ray,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_o,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_d,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_t_start,
        const float ray_t_far,
        pcg32 rng,
        const bool jitter_samples,
        // const bool contract_3d_samples,
        // output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_max_dt,
        torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> samples_3d,
        torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> samples_dirs,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_z,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_dt,
        torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> ray_start_end_idx
    ) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x; // each thread will deal with a new value

        // don't go out of bounds
        if (idx >= nr_rays)
        {
            return;
        }

        float eps = 1e-6;
        float t_start = ray_t_start[idx][0];
        float t_end = ray_t_far;
        float3 ray_o = make_float3(rays_o[idx][0], rays_o[idx][1], rays_o[idx][2]);
        float3 ray_d = make_float3(rays_d[idx][0], rays_d[idx][1], rays_d[idx][2]);
        float max_dt = 0.0;

        // 
        float delta_s = 1.0 / (nr_samples_per_ray - 1);

        float dt = 0.0;
        float s = 1.0;
        // first t is always equal to t_start
        float t_prec = t_start;
        bool is_first_sample = true;
        bool is_last_sample = false;
        for (int i = 0; i < nr_samples_per_ray; i++)
        {
            if (i != 0) 
            {
                is_first_sample = false;
            }

            if (i == nr_samples_per_ray - 1)
            {
                is_last_sample = true;
            }

            float t = (1.0 / (s + eps)) - 1.0;
            
            t += t_start;
            t = clamp(t, t_start, t_end);

            // first and last samples should not be jittered
            if (jitter_samples && !is_first_sample && !is_last_sample)
            {
                rng.advance(idx);
                float interp = rng.next_float();
                t = lerp(t_prec, t, interp);
                // s -= delta_s * rng.next_float();
                // s = clamp(s, 0.0, 1.0);
            }

            // store the 3d point depth
            samples_z[idx][i] = t;

            // store the 3d point
            float3 sample_3d = ray_o + t * ray_d;
            samples_3d[idx][i][0] = sample_3d.x;
            samples_3d[idx][i][1] = sample_3d.y;
            samples_3d[idx][i][2] = sample_3d.z;

            // store the dir
            samples_dirs[idx][i][0] = ray_d.x;
            samples_dirs[idx][i][1] = ray_d.y;
            samples_dirs[idx][i][2] = ray_d.z;

            // advance s
            s -= delta_s;

            dt = t - t_prec;
            max_dt = max(max_dt, dt);

            t_prec = t;
        }

        // set the max distance between samples
        ray_max_dt[idx][0] = max_dt;

        // ray_start_end_idx
        ray_start_end_idx[idx][0] = idx * nr_samples_per_ray;
        ray_start_end_idx[idx][1] = idx * nr_samples_per_ray + nr_samples_per_ray;
    }

    __global__ void
    compute_samples_fg_gpu(
        const int nr_rays,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_o,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_d,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_t_entry,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_t_exit,
        const float min_dist_between_samples,
        const int min_nr_samples_per_ray,
        const int max_nr_samples_per_ray,
        const int max_nr_samples,
        pcg32 rng,
        const bool jitter_samples,
        // output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_max_dt,
        torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> samples_idx,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_3d,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_dirs,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_z,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_dt,
        torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> ray_start_end_idx
    ) {

        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_rays)
        {
            return;
        }

        float eps = 1e-6;
        float t_start = ray_t_entry[idx][0];
        float t_exit = ray_t_exit[idx][0];
        float3 ray_o = make_float3(rays_o[idx][0], rays_o[idx][1], rays_o[idx][2]);
        float3 ray_d = make_float3(rays_d[idx][0], rays_d[idx][1], rays_d[idx][2]);

        //  calculate how many samples to create
        float distance_through_occupied_space = t_exit - t_start;
        int nr_samples_to_create = 0;
        float const_dist_between_samples = 0.0;

        if (distance_through_occupied_space <= 0.0) {
            nr_samples_to_create = 0;
            const_dist_between_samples = 0.0;
        } 
        else 
        {
            if (distance_through_occupied_space > min_dist_between_samples)
            {
                // the distance is bigger than the min distance between samples
                // nr_samples_to_create >= 1
                nr_samples_to_create = distance_through_occupied_space / min_dist_between_samples;
                nr_samples_to_create = clamp(nr_samples_to_create, 0, max_nr_samples_per_ray);
                const_dist_between_samples = distance_through_occupied_space / nr_samples_to_create;
            }
            else 
            // else we create 1 sample
            {
                nr_samples_to_create = 1;
                const_dist_between_samples = distance_through_occupied_space;
            }
        }
        
        int nr_samples_created = 0;
        int idx_start = idx * max_nr_samples_per_ray;
        
        if (nr_samples_to_create > 0 && nr_samples_to_create >= min_nr_samples_per_ray)
        {
            float t = t_start;
            
            // jitter just the beginning so the samples are all at the same distance from each other and therefore the dt is all the same
            if (jitter_samples)
            {
                rng.advance(idx); // since all the threads start with the same seed, we need to advance this thread so it gets other numbers different than the otehrs
                t = t + const_dist_between_samples * rng.next_float();
            }

            while (t < t_exit)
            {
                t = clamp(t, t_start, t_exit);
                float3 pos = ray_o + t * ray_d;

                if (nr_samples_created >= nr_samples_to_create)
                    break;

                // create sample                   
                // store positions
                samples_3d[idx_start + nr_samples_created][0] = pos.x;
                samples_3d[idx_start + nr_samples_created][1] = pos.y;
                samples_3d[idx_start + nr_samples_created][2] = pos.z;
                // store dirs
                samples_dirs[idx_start + nr_samples_created][0] = ray_d.x;
                samples_dirs[idx_start + nr_samples_created][1] = ray_d.y;
                samples_dirs[idx_start + nr_samples_created][2] = ray_d.z;
                // store z
                samples_z[idx_start + nr_samples_created][0] = t;
                // samples_dt[idx_start + nr_samples_created][0] = const_dist_between_samples;
                // advance t
                t += const_dist_between_samples;
                nr_samples_created += 1;
            }

            // if (nr_samples_to_create != nr_samples_created)
            // {
            //     printf("%d nr_samples_to_create is %d but nr_samples_created is %d, this should NOT have happened!\n", idx, nr_samples_to_create, nr_samples_created);
            // }
        }
        
        if (nr_samples_created < min_nr_samples_per_ray) {
            // should not have happened
            nr_samples_created = 0;
            // ray_start_end_idx[idx][0] = -1;
            // ray_start_end_idx[idx][1] = -1;
        } else {
            
            // set the max distance between samples
            ray_max_dt[idx][0] = const_dist_between_samples;

            // ray_start_end_idx
            ray_start_end_idx[idx][0] = idx_start;
            ray_start_end_idx[idx][1] = idx_start + nr_samples_created;
        }
    
        // fill remaining samples idx sentinel values
        for (int i = nr_samples_created; i < max_nr_samples_per_ray; i++)
        {
            // samples idx
            samples_idx[idx_start + i][0] = -1;
        }
    }

    __global__ void
    compute_samples_fg_in_grid_occupied_regions_gpu(
        const int nr_rays,
        // input
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_o,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_d,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_t_entry,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_t_exit,
        const float min_dist_between_samples,
        const int min_nr_samples_per_ray,
        const int max_nr_samples_per_ray,
        const int max_nr_samples,
        pcg32 rng,
        const bool jitter_samples,
        // occupancy grid
        const int nr_voxels_per_dim,
        const Eigen::Vector3f grid_extent,
        const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> grid_occupancy_tensor,
        const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> grid_roi_tensor,
        // output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_max_dt,
        torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> samples_idx,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_3d,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_dirs,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_z,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_dt,
        torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> ray_start_end_idx
    ) {
        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_rays)
        {
            return;
        }

        float eps = 1e-6;
        float t_start = ray_t_entry[idx][0];
        float t_exit = ray_t_exit[idx][0];
        // float t_start_occ = 0.0;
        // float t_end_occ = 0.0;
        float t = t_start;
        float3 ray_o = make_float3(rays_o[idx][0], rays_o[idx][1], rays_o[idx][2]);
        float3 ray_d = make_float3(rays_d[idx][0], rays_d[idx][1], rays_d[idx][2]);

        int nr_samples_to_create = 0;
        float const_dist_between_samples = 0.0;
        float3 pos;
        int idx_voxel;
        float dist_to_next_voxel = 0.0;
        float distance_through_occupied_space = 0.0;

        // run once through the occupancy to check how much distance do we traverse through occupied space
        while (t < t_exit)
        {
            // printf("t is %f \n", t);
            pos = ray_o + t * ray_d;
            idx_voxel = pos_to_lin_idx(pos, nr_voxels_per_dim, grid_extent);
            
            // check if within bounds
            if (idx_voxel >= nr_voxels_per_dim * nr_voxels_per_dim * nr_voxels_per_dim || idx_voxel < 0)
            {
                // reached end of the grid
                printf("idx_voxel is %d is out of range, this should NOT have happened!\n", idx_voxel);
                break;
            }

            // if we are in an occupied voxel, accumulate the distance that we traversed through occupied space
            if (grid_roi_tensor[idx_voxel] && grid_occupancy_tensor[idx_voxel])
            {
                distance_through_occupied_space += dist_to_next_voxel;
                // if (distance_through_occupied_space == 0.0)
                // {
                //     t_start_occ = t;
                // }
                // t_end_occ = t;
            }

            dist_to_next_voxel = distance_to_next_voxel(pos, ray_d, nr_voxels_per_dim, grid_extent);
            t += dist_to_next_voxel;
        }
        distance_through_occupied_space = clamp(distance_through_occupied_space, 0.0, t_exit - t_start);

        if (distance_through_occupied_space <= 0.0) {
            nr_samples_to_create = 0;
            const_dist_between_samples = 0.0;
        } 
        else 
        {
            if (distance_through_occupied_space > min_dist_between_samples)
            {
                // the distance is bigger than the min distance between samples
                // nr_samples_to_create >= 1
                nr_samples_to_create = distance_through_occupied_space / min_dist_between_samples;
                // printf("%d nr_samples_to_create is %d \n", idx, nr_samples_to_create);
                nr_samples_to_create = clamp(nr_samples_to_create, 0, max_nr_samples_per_ray);
                // printf("%d nr_samples_to_create is %d \n", idx, nr_samples_to_create);
                const_dist_between_samples = distance_through_occupied_space / (float) nr_samples_to_create;
                // printf("%d const_dist_between_samples is %f \n", idx, const_dist_between_samples);
            }
            else 
            // else we create 1 sample
            {
                nr_samples_to_create = 1;
                const_dist_between_samples = distance_through_occupied_space;
            }
        }

        // if we have samples to create we create them
        int nr_samples_created = 0;
        float remaining_dist_until_border = 0.0;
        float step_size = 0.0;
        float dist_to_next_sample = 0.0;  // distance to traverse in occupied space before sampling
        int idx_start = idx * max_nr_samples_per_ray;
        dist_to_next_voxel = 0.0;

        if (nr_samples_to_create > 0 && nr_samples_to_create >= min_nr_samples_per_ray)
        {
            t = t_start;
            
            // jitter just the beginning so the samples are all at the same distance from each other and therefore the dt is all the same
            if (jitter_samples)
            {
                rng.advance(idx); // since all the threads start with the same seed, we need to advance this thread so it gets other numbers different than the otehrs
                // t = t + const_dist_between_samples * rng.next_float();
                dist_to_next_sample = const_dist_between_samples * rng.next_float();
            }

            float dist_traversed_in_contiguous_occupied_space = 0.0;
            while (t < t_exit)
            {
                t = clamp(t, t_start, t_exit);
                pos = ray_o + t * ray_d;

                if (nr_samples_created >= nr_samples_to_create)
                    break;

                idx_voxel = pos_to_lin_idx(pos, nr_voxels_per_dim, grid_extent);
            
                // check if within bounds
                if (idx_voxel >= nr_voxels_per_dim * nr_voxels_per_dim * nr_voxels_per_dim || idx_voxel < 0)
                {
                    // reached end of the grid
                    printf("idx_voxel is %d is out of range, this should NOT have happened!\n", idx_voxel);
                    break;
                }

                bool in_occupied_space = grid_roi_tensor[idx_voxel] && grid_occupancy_tensor[idx_voxel];
                if (in_occupied_space)
                {
                    if (dist_to_next_sample == 0.0) 
                    {
                        // create sample                   
                        // store positions
                        samples_3d[idx_start + nr_samples_created][0] = pos.x;
                        samples_3d[idx_start + nr_samples_created][1] = pos.y;
                        samples_3d[idx_start + nr_samples_created][2] = pos.z;
                        // store dirs
                        samples_dirs[idx_start + nr_samples_created][0] = ray_d.x;
                        samples_dirs[idx_start + nr_samples_created][1] = ray_d.y;
                        samples_dirs[idx_start + nr_samples_created][2] = ray_d.z;
                        // store z
                        samples_z[idx_start + nr_samples_created][0] = t;
                        nr_samples_created += 1;
                        dist_to_next_sample = const_dist_between_samples;                
                    }
                }

                dist_to_next_voxel = distance_to_next_voxel(pos, ray_d, nr_voxels_per_dim, grid_extent);

                if (in_occupied_space) {
                    step_size = min(dist_to_next_voxel, dist_to_next_sample);
                    // sampling point not reached yet
                    dist_to_next_sample -= step_size;
                    // sampling point reached when dist_to_next_sample is 0
                    if (dist_to_next_sample <= eps)
                        dist_to_next_sample = 0.0;
                } else {
                    // travel to next voxel
                    step_size = dist_to_next_voxel;
                }

                // advance t
                t += step_size;
            }

            // if (nr_samples_to_create != nr_samples_created)
            // {
            //     printf("%d nr_samples_to_create is %d but nr_samples_created is %d, this should NOT have happened!\n", idx, nr_samples_to_create, nr_samples_created);
            //     printf("%d min_dist_between_samples is %f \n", idx, min_dist_between_samples);
            //     printf("%d distance_through_occupied_space is %f \n", idx, distance_through_occupied_space);
            //     printf("%d nr_samples_to_create is %d \n", idx, nr_samples_to_create);
            //     printf("%d const_dist_between_samples is %f \n", idx, const_dist_between_samples);
            // }
        }

        if (nr_samples_created < min_nr_samples_per_ray) {
            // should not have happened
            nr_samples_created = 0;
        } else {
            // set the max distance between samples
            ray_max_dt[idx][0] = const_dist_between_samples;
            // ray_start_end_idx
            ray_start_end_idx[idx][0] = idx_start;
            ray_start_end_idx[idx][1] = idx_start + nr_samples_created;
        }

        // fill remaining samples idx sentinel values
        for (int i = nr_samples_created; i < max_nr_samples_per_ray; i++)
        {
            // samples idx
            samples_idx[idx_start + i][0] = -1;
        }
    }

    __global__ void
    init_with_one_sample_per_ray_gpu(
        const int nr_rays,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_3d,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_dir,
        // output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_samples_3d,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_samples_dirs,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_samples_z,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_samples_dt,
        torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> out_ray_start_end_idx
    ) {

        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_rays)
        {
            return;
        }

        out_ray_start_end_idx[idx][0] = idx;
        out_ray_start_end_idx[idx][1] = idx + 1;
        
        // copy the sample pos
        out_samples_3d[idx][0] = samples_3d[idx][0];
        out_samples_3d[idx][1] = samples_3d[idx][1];
        out_samples_3d[idx][2] = samples_3d[idx][2];
        // copy the sample dir
        out_samples_dirs[idx][0] = samples_dir[idx][0];
        out_samples_dirs[idx][1] = samples_dir[idx][1];
        out_samples_dirs[idx][2] = samples_dir[idx][2];
        // store z
        out_samples_z[idx][0] = 0.0f;
        out_samples_dt[idx][0] = 0.0f;
    }
    
    __global__ void
    contract_samples_gpu(
        const int nr_rays,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_o,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> ray_start_end_idx,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_3d,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_z,
        // output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_samples_3d,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_samples_z
    ) {
        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_rays)
            return;

        int scale = 2;

        // get the indexes of the start and end of the uniform samples
        int idx_start = ray_start_end_idx[idx][0];
        int idx_end = ray_start_end_idx[idx][1];
        int nr_samples = idx_end - idx_start;

        float3 camera_pos = make_float3(
            ray_o[idx][0],
            ray_o[idx][1],
            ray_o[idx][2]
        );

        for (int i = 0; i < nr_samples; i++)
        {
            // read input
            float3 sample_pos = make_float3(
                samples_3d[idx_start + i][0],
                samples_3d[idx_start + i][1],
                samples_3d[idx_start + i][2]
            );
            float sample_z = samples_z[idx_start + i][0];

            // process

            // compute point norm
            float norm = length(sample_pos * scale);

            // check if should contract
            if (norm > 1.0)
            {
                float factor = 2.0f - 1.0f / norm;

                // contract sample pos
                sample_pos = (factor * sample_pos) / norm;

                // compute contracted sample pos distance from camera pos
                sample_z = length(sample_pos - camera_pos);
            }

            // write output (just a copy if not contracted)
            out_samples_3d[idx_start + i][0] = sample_pos.x;
            out_samples_3d[idx_start + i][1] = sample_pos.y;
            out_samples_3d[idx_start + i][2] = sample_pos.z;
            out_samples_z[idx_start + i][0] = sample_z;
        }
    }

    __global__ void
    uncontract_samples_gpu(
        const int nr_rays,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_o,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> ray_start_end_idx,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_3d,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_z,
        // output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_samples_3d,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_samples_z
    ) {
        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_rays)
            return;

        int scale = 2;

        // get the indexes of the start and end of the uniform samples
        int idx_start = ray_start_end_idx[idx][0];
        int idx_end = ray_start_end_idx[idx][1];
        int nr_samples = idx_end - idx_start;

        float3 camera_pos = make_float3(
            ray_o[idx][0],
            ray_o[idx][1],
            ray_o[idx][2]
        );

        for (int i = 0; i < nr_samples; i++)
        {
            // read input
            float3 sample_pos = make_float3(
                samples_3d[idx_start + i][0],
                samples_3d[idx_start + i][1],
                samples_3d[idx_start + i][2]
            );
            float sample_z = samples_z[idx_start + i][0];

            // process

            // compute point norm
            float norm = length(sample_pos * scale);

            // check if should uncontract
            if (norm > 1.0)
            {
                float factor = 1.0f / (2.0f - norm);

                // uncontract sample pos
                sample_pos = (factor * sample_pos) / norm;

                // compute contracted sample pos distance from camera pos
                sample_z = length(sample_pos - camera_pos);
            }

            // write output (just a copy if not contracted)
            out_samples_3d[idx_start + i][0] = sample_pos.x;
            out_samples_3d[idx_start + i][1] = sample_pos.y;
            out_samples_3d[idx_start + i][2] = sample_pos.z;
            out_samples_z[idx_start + i][0] = sample_z;
        }
    }

} // namespace RaySamplerGPU
