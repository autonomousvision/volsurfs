#pragma once

#include "volsurfs/helper_math.h"

// matrices
#include "volsurfs/mat3.h"
#include "volsurfs/mat4.h"

#include "volsurfs/pcg32.h"

// TODO: these functions are copied from OccupancyGridGPU.cuh, they should be moved to a common file

__host__ __device__ inline uint64_t expand_bits(uint64_t w)
{
    w &= 0x00000000001fffff;
    w = (w | w << 32) & 0x001f00000000ffff;
    w = (w | w << 16) & 0x001f0000ff0000ff;
    w = (w | w << 8) & 0x010f00f00f00f00f;
    w = (w | w << 4) & 0x10c30c30c30c30c3;
    w = (w | w << 2) & 0x1249249249249249;
    return w;
}

// Calculates a 30-bit Morton code for the
// given 3D idx located within the unit cube [0,nr_voxels_per_dim-1].
__host__ __device__ inline uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z)
{
    uint32_t xx = expand_bits(x);
    uint32_t yy = expand_bits(y);
    uint32_t zz = expand_bits(z);
    return xx | (yy << 1) | (zz << 2);
}

__host__ __device__ inline uint32_t morton3D_clamped(int x, int y, int z, const int nr_voxels_per_dim)
    {
        x = clamp(x, 0, nr_voxels_per_dim - 1);
        y = clamp(y, 0, nr_voxels_per_dim - 1);
        z = clamp(z, 0, nr_voxels_per_dim - 1);
        return morton3D(x, y, z);
    }

    __host__ __device__ inline uint32_t morton3D_invert(uint32_t x)
    {
        x = x & 0x49249249;
        x = (x | (x >> 2)) & 0xc30c30c3;
        x = (x | (x >> 4)) & 0x0f00f00f;
        x = (x | (x >> 8)) & 0xff0000ff;
        x = (x | (x >> 16)) & 0x0000ffff;
        return x;
    }

// position in some world coordinates
__host__ __device__ inline int pos_to_lin_idx(float3 pos, const int nr_voxels_per_dim, const Eigen::Vector3f grid_extent)
{
    // we go in reverse order of the lin_idx_to_3D

    // we apply the grid extent because now we have the grid centered around the origin
    pos.x = pos.x / grid_extent.x();
    pos.y = pos.y / grid_extent.y();
    pos.z = pos.z / grid_extent.z();

    // shift so that it doesnt start at 0,0,0 but rather the center of the grid is at 0.0.0
    pos.x = pos.x + 0.5;
    pos.y = pos.y + 0.5;
    pos.z = pos.z + 0.5;

    pos.x = pos.x * nr_voxels_per_dim;
    pos.y = pos.y * nr_voxels_per_dim;
    pos.z = pos.z * nr_voxels_per_dim;

    int morton_idx = morton3D(pos.x, pos.y, pos.z);

    return morton_idx;
}

__host__ __device__ inline float3 lin_idx_to_3D(const uint32_t lin_idx, const int nr_voxels_per_dim, const Eigen::Vector3f grid_extent, const bool center_grid, const bool get_center_of_voxel)
{
    float x = morton3D_invert(lin_idx >> 0);
    float y = morton3D_invert(lin_idx >> 1); // right bit shift divide by 2 and floors
    float z = morton3D_invert(lin_idx >> 2);
    // now the xyz are in range [0,nr_voxels_per_dim-1] so it just does a translation from 1d idx to 3D idx

    x = x / nr_voxels_per_dim;
    y = y / nr_voxels_per_dim;
    z = z / nr_voxels_per_dim;
    // now xyz is in range [0,1-voxel_size]

    // shift so that it doesnt start at 0,0,0 but rather the center of the grid is at 0.0.0
    if (center_grid) {
        x = x - 0.5;
        y = y - 0.5;
        z = z - 0.5;
    }
    
    if (get_center_of_voxel)
    {
        // we want to get the center of the voxel so we shift by half of the voxel_size
        float voxel_size = 1.0 / nr_voxels_per_dim; // we divicde 1 by the voxel per dim because now we have xyz in normalized coordiantes so between 0 and 1-voxelssize
        float half_voxel_size = voxel_size / 2;
        x += half_voxel_size;
        y += half_voxel_size;
        z += half_voxel_size;
    }
    // now we have either the center points of the voxels or the lower left corner and it has extent of 1

    // we apply the grid extent because now we have the grid centered around the origin
    x = x * grid_extent.x();
    y = y * grid_extent.y();
    z = z * grid_extent.z();

    float3 pos = make_float3(x, y, z);

    return pos;
}

// https://forums.developer.nvidia.com/t/sign-function/18375/4
__host__ __device__ inline int sign(float x)
{
    int t = x < 0 ? -1 : 0;
    return x > 0 ? 1 : t;
}

//https://github.com/NVlabs/instant-ngp/blob/master/src/testbed_nerf.cu
__host__ __device__ inline float distance_to_next_voxel(float3 pos, const float3& dir, const int nr_voxels_per_dim, const Eigen::Vector3f grid_extent) { // dda like step
    
    float eps = 1e-6;
    
    if (abs(dir.x) < eps && abs(dir.y) < eps && abs(dir.z) < eps) {
        printf("all dir components are 0, this should not happen\n");
        return 1e10;
    }

    // printf("pos before scaling is %f,%f,%f \n", pos.x, pos.y, pos.z);
    pos.x = pos.x / grid_extent.x();
    pos.y = pos.y / grid_extent.y();
    pos.z = pos.z / grid_extent.z();
    pos.x = pos.x * nr_voxels_per_dim;
    pos.y = pos.y * nr_voxels_per_dim;
    pos.z = pos.z * nr_voxels_per_dim;
    // printf("pos after scaling is %f,%f,%f \n", pos.x, pos.y, pos.z);
    float voxel_size = 1.0;
    // printf("voxel_size is %f \n", voxel_size);
    // printf("dir is %f,%f,%f \n", dir.x, dir.y, dir.z);

    float tx = 1e10;
    if (abs(dir.x) > eps) {
        float pos_x_offset = pos.x + voxel_size * sign(dir.x);
        // printf("pos_x_offset is %f \n", pos_x_offset);
        float pos_x_prime = floorf(pos_x_offset);
        // printf("pos_x_prime is %f \n", pos_x_prime);
        tx = abs(pos_x_prime - pos.x) / nr_voxels_per_dim;
        // printf("tx is %f \n", tx);
        tx = tx * grid_extent.x();
        // printf("tx scaled is %f \n", tx);
    }
    
    float ty = 1e10;
    if (abs(dir.y) > eps) {
        float pos_y_offset = pos.y + voxel_size * sign(dir.y);
        float pos_y_prime = floorf(pos_y_offset);
        // printf("pos_y_prime is %f \n", pos_y_prime);
        ty = abs(pos_y_prime - pos.y) / nr_voxels_per_dim;
        // printf("ty is %f \n", ty);
        ty = ty * grid_extent.y();
        // printf("ty scaled is %f \n", ty);
    }

    float tz = 1e10;
    if (abs(dir.z) > eps) {
        float pos_z_offset = pos.z + voxel_size * sign(dir.z);
        float pos_z_prime = floorf(pos_z_offset);
        // printf("pos_z_prime is %f \n", pos_z_prime);
        tz = abs(pos_z_prime - pos.z) / nr_voxels_per_dim;
        // printf("tz is %f \n", tz);
        tz = tz * grid_extent.z();
        // printf("tz scaled is %f \n", tz);
    }

    float t = min(min(tx, ty), tz) + eps;
    return t;
}