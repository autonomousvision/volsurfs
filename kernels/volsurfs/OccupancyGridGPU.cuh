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

namespace OccupancyGridGPU
{

    // from https://github.com/NVlabs/instant-ngp/blob/e1d33a42a4de0b24237685f2ebdc07bcef1ecae9/src/testbed_nerf.cu
    inline constexpr __device__ uint32_t MAX_STEPS() { return 2048 * 2; } // finest number of steps per unit length

    template <typename T>
    __device__ void inline swap(T a, T b)
    {
        T c(a);
        a = b;
        b = c;
    }

    __global__ void
    get_grid_lower_left_voxels_vertices_gpu(
        const int nr_voxels_to_select,
        const int nr_voxels_per_dim,
        const Eigen::Vector3f grid_extent,
        const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> point_indices,
        // output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ll_points
        )
    {
        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_voxels_to_select)
        {
            return;
        }

        int idx_voxel = point_indices[idx];

        // voxels centers
        bool center_grid = true;
        bool get_center_of_voxel = false;
        float3 pos3d = lin_idx_to_3D(idx_voxel, nr_voxels_per_dim, grid_extent, center_grid, get_center_of_voxel);

        ll_points[idx][0] = pos3d.x;
        ll_points[idx][1] = pos3d.y;
        ll_points[idx][2] = pos3d.z;
    }

    __global__ void
    get_grid_samples_gpu(
        const int nr_voxels_to_select,
        const int nr_voxels_per_dim,
        const Eigen::Vector3f grid_extent,
        const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> point_indices,
        pcg32 rng,
        const bool jitter_samples,
        // output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> center_points
        )
    {
        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_voxels_to_select)
        {
            return;
        }

        int idx_voxel = point_indices[idx];

        // voxels centers
        bool center_grid = true;
        bool get_center_of_voxel = true;
        float3 pos3d = lin_idx_to_3D(idx_voxel, nr_voxels_per_dim, grid_extent, center_grid, get_center_of_voxel);

        if (jitter_samples)
        {
            float voxel_size_x = grid_extent.x() / nr_voxels_per_dim;
            float voxel_size_y = grid_extent.y() / nr_voxels_per_dim;
            float voxel_size_z = grid_extent.z() / nr_voxels_per_dim;
            float half_voxel_size_x = voxel_size_x / 2.0;
            float half_voxel_size_y = voxel_size_y / 2.0;
            float half_voxel_size_z = voxel_size_z / 2.0;

            float rand;
            float mov;

            rng.advance(idx * 3); // since all the threads start with the same seed, we need to advance this thread so it gets other numbers different than the otehrs
            // x
            rand = rng.next_float(); // random in range [0,1]
            mov = voxel_size_x * rand - half_voxel_size_x;
            pos3d.x += mov;
            // y
            rand = rng.next_float(); // random in range [0,1]
            mov = voxel_size_y * rand - half_voxel_size_y;
            pos3d.y += mov;
            // z
            rand = rng.next_float(); // random in range [0,1]
            mov = voxel_size_z * rand - half_voxel_size_z;
            pos3d.z += mov;
        }

        center_points[idx][0] = pos3d.x;
        center_points[idx][1] = pos3d.y;
        center_points[idx][2] = pos3d.z;
    }

    __global__ void
    update_grid_values_gpu(
        const int nr_points,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> values_tensor,
        const int nr_voxels_per_dim,
        const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> point_indices,
        const float decay,
        // output
        torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> grid_values_tensor)
    {
        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_points)
        {
            return;
        }

        // get the idx of the voxel corresponding to this point
        // the index here is already in morton order since the points were generated in morton order
        int idx_voxel = point_indices[idx];

        float old_value = grid_values_tensor[idx_voxel] * decay;
        float new_value = values_tensor[idx][0];

        // update
        float updated_value = fmax(new_value, old_value);
        grid_values_tensor[idx_voxel] = updated_value;
    }

    __global__ void update_grid_occupancy_with_density_values_gpu(
        const int nr_points,
        const int nr_voxels_per_dim,
        const Eigen::Vector3f grid_extent,
        const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> point_indices,
        const float occupancy_tresh,
        const bool check_neighbours,
        const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> grid_values_tensor,
        // output
        torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> grid_occupancy_tensor)
    {

        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_points)
        {
            return;
        }

        // get the idx of the voxel corresponding to this point
        int idx_voxel = point_indices[idx];
        // printf("idx_voxel is %d \n", idx_voxel);

        bool is_empty = true;
        if (check_neighbours) {
            
            // lower left corner
            bool center_grid = false;
            bool get_center_of_voxel = false;
            float3 pos3d = lin_idx_to_3D(idx_voxel, nr_voxels_per_dim, grid_extent, center_grid, get_center_of_voxel);
            pos3d = pos3d * nr_voxels_per_dim;

            // printf("pos3d is %d,%d,%d \n", int(pos3d.x), int(pos3d.y), int(pos3d.z));
            for (int i = -1; i <= 1; i++)
            {
                if (pos3d.x + i < 0 || pos3d.x + i > nr_voxels_per_dim - 1)
                    continue;
                
                for (int j = -1; j <= 1; j++)
                {
                    if (pos3d.y + j < 0 || pos3d.y + j > nr_voxels_per_dim - 1)
                        continue;
                    
                    for (int k = -1; k <= 1; k++)
                    {
                        if (pos3d.z + k < 0 || pos3d.z + k > nr_voxels_per_dim - 1)
                            continue;
                        
                        // neighbor indices
                        float3 pos3d_neigh;
                        pos3d_neigh.x = pos3d.x + i;
                        pos3d_neigh.y = pos3d.y + j;
                        pos3d_neigh.z = pos3d.z + k;
                        // printf("pos3d_neigh is %d,%d,%d \n", int(pos3d_neigh.x), int(pos3d_neigh.y), int(pos3d_neigh.z));
                    
                        int idx_voxel_neigh = morton3D(pos3d_neigh.x, pos3d_neigh.y, pos3d_neigh.z);
                        // printf("idx_voxel_neigh is %d \n", idx_voxel_neigh);

                        // // bool out_of_bounds = idx_voxel > (nr_voxels_per_dim * nr_voxels_per_dim * nr_voxels_per_dim) - 1 || idx_voxel < 0;
                        // // if (!out_of_bounds)
                        is_empty = is_empty && grid_values_tensor[idx_voxel_neigh] <= occupancy_tresh;
                    }
                }
            }       
        } else {
            is_empty = grid_values_tensor[idx_voxel] <= occupancy_tresh;
        }
    
        // update
        grid_occupancy_tensor[idx_voxel] = !is_empty;
        // printf("occupancy is %d \n", is_occupied);
    }

    // https://arxiv.org/pdf/2106.10689.pdf
    __device__ float logistic_distribution(float x, float beta)
    {
        float exp_term = clamp(exp(-beta * x), -1e6, 1e6);
        float res = beta * exp_term / (powf((1 + exp_term), 2));
        return res;
    }


    // Function to calculate the distance between two points
    __device__ float euclidean_distance(const Eigen::Vector3f p1, const Eigen::Vector3f p2) {
        return sqrtf(powf(p2.x() - p1.x(), 2) + powf(p2.y() - p1.y(), 2) + powf(p2.z() - p1.z(), 2));
    }
    

    // Function to find the maximum distance between two arbitrary points within a cuboid
    __device__ float max_distance_in_cuboid(const Eigen::Vector3f cuboid_sides) {
    
        float max_distance = 0.0;
        
        // create list of vertices

        for (int k = 0; k <= 1; k++) {
            for (int j = 0; j <= 1; j++) {
                for (int i = 0; i <= 1; i++) {
                    Eigen::Vector3f vertex = Eigen::Vector3f(i * cuboid_sides.x(), j * cuboid_sides.y(), k * cuboid_sides.z());
                    for (int l = 0; l <= 1; l++) {
                        for (int m = 0; m <= 1; m++) {
                            for (int n = 0; n <= 1; n++) {
                                Eigen::Vector3f vertex2 = Eigen::Vector3f(l * cuboid_sides.x(), m * cuboid_sides.y(), n * cuboid_sides.z());
                                float dist = euclidean_distance(vertex, vertex2);
                                if (dist > max_distance) {
                                    max_distance = dist;
                                }
                            }
                        }
                    }
                }
            }
        }

        return max_distance;
    }

    __global__ void
    update_grid_occupancy_with_sdf_values_gpu(
        const int nr_points,
        const Eigen::Vector3f grid_extent,
        const int nr_voxels_per_dim,
        const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> point_indices,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> logistic_beta,
        const float occupancy_thresh,
        const bool check_neighbours,
        const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> grid_values_tensor,
        // output
        torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> grid_occupancy_tensor)
    {
        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_points)
        {
            return;
        }

        // get the idx of the voxel corresponding to this point
        int idx_voxel = point_indices[idx];

        float sdf = grid_values_tensor[idx_voxel]; // signed distance
        float df = fabs(sdf); // distance

        // TODO: check neighbours

        // check if the sdf can posibly be 0 or within the range that it would get a slight density
        // check the sdf that can be reached within this voxel
        Eigen::Vector3f voxel_size = grid_extent / nr_voxels_per_dim;

        // compute the maximum distance between two arbitrary points within a cuboid
        float cube_diagonal = max_distance_in_cuboid(voxel_size);
        float half_cube_diagonal = cube_diagonal / 2.0;
    
        float min_dists_possible_in_voxel = clamp(df - half_cube_diagonal, 0.0, 1e10);
        
        // pass this sdf through the logistic function that neus uses and check what density it gets
        float weight = logistic_distribution(min_dists_possible_in_voxel, logistic_beta[idx][0]);

        grid_occupancy_tensor[idx_voxel] = weight > occupancy_thresh;
    }

    __global__ void
    get_rays_t_near_t_far_gpu(
        const int nr_rays,
        // input
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_o,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_d,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_t_entry,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_t_exit,
        // occupancy grid
        const int nr_voxels_per_dim,
        const Eigen::Vector3f grid_extent,
        const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> grid_occupancy_tensor,
        const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> grid_roi_tensor,
        // output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_grid_t_near,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_grid_t_exit
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
        
        float t = t_start;
        float3 ray_o = make_float3(rays_o[idx][0], rays_o[idx][1], rays_o[idx][2]);
        float3 ray_d = make_float3(rays_d[idx][0], rays_d[idx][1], rays_d[idx][2]);

        float3 pos;
        bool first_time = true;
        int idx_voxel;
        float dist_to_next_voxel = 0.0;

        // set valid values
        ray_grid_t_near[idx][0] = t_start;
        ray_grid_t_exit[idx][0] = t_start;

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
                if (first_time)
                {
                    // store ray_grid_t_near as t
                    ray_grid_t_near[idx][0] = t;
                    first_time = false;
                }
            }

            dist_to_next_voxel = distance_to_next_voxel(pos, ray_d, nr_voxels_per_dim, grid_extent);
            t += dist_to_next_voxel;

            if (grid_roi_tensor[idx_voxel] && grid_occupancy_tensor[idx_voxel]){
                // store ray_grid_t_far as last t inside the grid
                ray_grid_t_exit[idx][0] = clamp(t, t_start, t_exit);
            }
        }
    }

    __global__ void
    check_occupancy_gpu(
        const int nr_points,
        const int nr_voxels_per_dim,
        const Eigen::Vector3f grid_extent,
        const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> grid_values_tensor,
        const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> grid_occupancy_tensor,
        const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> grid_roi_tensor,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> points,
        // output
        torch::PackedTensorAccessor32<bool, 2, torch::RestrictPtrTraits> voxel_occupancy,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> voxel_value
    ) {
        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_points)
        {
            return;
        }

        float3 pos;
        pos.x = points[idx][0];
        pos.y = points[idx][1];
        pos.z = points[idx][2];
        int idx_voxel = pos_to_lin_idx(pos, nr_voxels_per_dim, grid_extent);

        // when you extract a mesh using marching cubes you may have some faces that slightly outside of the occupancy grid
        bool out_of_bounds = idx_voxel >= (nr_voxels_per_dim * nr_voxels_per_dim * nr_voxels_per_dim) || idx_voxel < 0;

        if (!out_of_bounds)
        {
            bool occ = grid_roi_tensor[idx_voxel] && grid_occupancy_tensor[idx_voxel];
            voxel_occupancy[idx][0] = occ;
            float val = grid_values_tensor[idx_voxel];
            voxel_value[idx][0] = val;
        }
        else
        {
            // we are out of bounds so we will set occupancy to 0
            voxel_occupancy[idx][0] = false;
            voxel_value[idx][0] = 0.0;
        }
    }

    __global__ void
    advance_ray_sample_to_next_occupied_voxel_gpu(
        const int nr_points,
        const int nr_voxels_per_dim,
        const Eigen::Vector3f grid_extent,
        const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_dirs,
        const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_3d,
        const torch::PackedTensorAccessor32<bool,1,torch::RestrictPtrTraits> grid_occupancy_tensor,
        const torch::PackedTensorAccessor32<bool,1,torch::RestrictPtrTraits> grid_roi_tensor,
        //output
        torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> new_samples_3d,
        torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> is_within_bounds
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

        //don't go out of bounds
        if(idx >= nr_points) {
            return;
        }

        //load everything for this ray
        float3 ray_o = make_float3(samples_3d[idx][0], samples_3d[idx][1], samples_3d[idx][2]);
        float3 ray_d = make_float3(samples_dirs[idx][0], samples_dirs[idx][1], samples_dirs[idx][2]);

        float eps = 1e-6;

        float prec_t = 0; // previous t
        float t = 0; // current t
        bool within_bounds = true;

        while (within_bounds) {
            float3 pos = ray_o + t * ray_d;
            int idx_voxel = pos_to_lin_idx(pos, nr_voxels_per_dim, grid_extent);
            bool out_of_bounds = false;
            if(idx_voxel >= nr_voxels_per_dim*nr_voxels_per_dim*nr_voxels_per_dim || idx_voxel < 0){
                out_of_bounds = true;
            }
            if (out_of_bounds) {
                within_bounds=false;
                pos = ray_o + prec_t * ray_d;
                new_samples_3d[idx][0]=pos.x;
                new_samples_3d[idx][1]=pos.y;
                new_samples_3d[idx][2]=pos.z;
            } else {
                //advance towards next voxel and check if it's occupied
                float dist_to_next_voxel = distance_to_next_voxel(pos, ray_d, nr_voxels_per_dim, grid_extent);
                prec_t = t;
                t += dist_to_next_voxel;
                t += eps;
                
                //if we are in an occupied voxel, store this new sample point and break
                if (grid_roi_tensor[idx_voxel] && grid_occupancy_tensor[idx_voxel]){
                    new_samples_3d[idx][0]=pos.x;
                    new_samples_3d[idx][1]=pos.y;
                    new_samples_3d[idx][2]=pos.z;
                    break;
                }
            }
        }
        is_within_bounds[idx][0]=within_bounds;
    }

    __global__ void
    get_first_rays_sample_start_of_grid_occupied_regions_gpu(
        const int nr_rays,
        const int nr_voxels_per_dim,
        const Eigen::Vector3f grid_extent,
        const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> rays_o,
        const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> rays_d,
        const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_t_entry,
        const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_t_exit,
        const torch::PackedTensorAccessor32<bool,1,torch::RestrictPtrTraits> grid_occupancy_tensor,
        const torch::PackedTensorAccessor32<bool,1,torch::RestrictPtrTraits> grid_roi_tensor,
        //output
        torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_3d,
        torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_dirs,
        torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_z,
        torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_dt,
        torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> ray_start_end_idx
    ) {
        //each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        //don't go out of bounds
        if(idx >= nr_rays){ 
            return;
        }

        //load everything for this ray
        float3 ray_o=make_float3(rays_o[idx][0], rays_o[idx][1], rays_o[idx][2]);
        float3 ray_d=make_float3(rays_d[idx][0], rays_d[idx][1], rays_d[idx][2]);

        float t_start = ray_t_entry[idx][0];
        float t_exit = ray_t_exit[idx][0];

        float eps = 1e-6;

        float t=t_start; //cur_t
        int nr_steps = 0;
        while (t < t_exit && nr_steps < MAX_STEPS()) {

            float3 pos = ray_o + t * ray_d;
            int idx_voxel = pos_to_lin_idx(pos, nr_voxels_per_dim, grid_extent);
            if(idx_voxel >= nr_voxels_per_dim * nr_voxels_per_dim * nr_voxels_per_dim || idx_voxel<0) {
                // set the ray to be empty
                break;
            }

            float dist_to_next_voxel = distance_to_next_voxel(pos, ray_d, nr_voxels_per_dim, grid_extent);
            t += dist_to_next_voxel;
            
            //tiny epsilon so we make sure that we are now in the next voxel when we sample from it
            t += eps; 
            
            //if we are in an occupied voxel, accumulate the distance that we traversed through occupied space 
            if (grid_roi_tensor[idx_voxel] && grid_occupancy_tensor[idx_voxel]){
                
                //create one sample and return
                ray_start_end_idx[idx][0]=idx;
                ray_start_end_idx[idx][1]=idx+1; 
                //store positions
                samples_3d[idx][0] = pos.x;
                samples_3d[idx][1] = pos.y;
                samples_3d[idx][2] = pos.z;
                //store dirs
                samples_dirs[idx][0] = ray_d.x;
                samples_dirs[idx][1] = ray_d.y;
                samples_dirs[idx][2] = ray_d.z;
                //store z
                samples_z[idx][0] = t;
                samples_dt[idx][0] = 0;

                return;
            }
        }

        // this ray passes only through un-occupied space
        ray_start_end_idx[idx][0]=0;
        ray_start_end_idx[idx][1]=0;
    }
    
} // namespace occupancy grid
