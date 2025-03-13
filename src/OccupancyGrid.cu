#include "volsurfs/OccupancyGrid.cuh"

// c++
//  #include <string>

#include "volsurfs/UtilsPytorch.h"

// my stuff
#include "volsurfs/OccupancyGridGPU.cuh"

using torch::Tensor;

template <typename T>
T div_round_up(T val, T divisor)
{
    return (val + divisor - 1) / divisor;
}

pcg32 OccupancyGrid::m_rng;

// CPU code that calls the kernels
OccupancyGrid::OccupancyGrid(const int nr_voxels_per_dim, const Eigen::Vector3f grid_extent) : m_nr_voxels_per_dim(nr_voxels_per_dim), m_grid_extent(grid_extent)
{
    m_grid_values = make_grid_values(nr_voxels_per_dim);
    m_grid_occupancy = make_grid_occupancy(nr_voxels_per_dim);
    m_grid_roi = make_grid_occupancy(nr_voxels_per_dim);
    std::cout << "occupancy grid of resolution " << nr_voxels_per_dim << " created" << std::endl;
    std::cout << "grid extent: [" << m_grid_extent.x() << ", " << m_grid_extent.y() << ", " << m_grid_extent.z() << "]" << std::endl;
}

OccupancyGrid::~OccupancyGrid()
{
}

Eigen::Vector3f OccupancyGrid::get_grid_extent()
{
    return m_grid_extent;
}

int OccupancyGrid::get_nr_voxels()
{
    return m_nr_voxels_per_dim * m_nr_voxels_per_dim * m_nr_voxels_per_dim;
}

int OccupancyGrid::get_nr_voxels_in_roi()
{
    return m_grid_roi.sum().item<int>();
}

int OccupancyGrid::get_nr_voxels_per_dim()
{
    return m_nr_voxels_per_dim;
}

torch::Tensor OccupancyGrid::get_grid_values()
{
    return m_grid_values;
}

torch::Tensor OccupancyGrid::get_grid_occupancy()
{
    return m_grid_occupancy;
}

torch::Tensor OccupancyGrid::get_grid_roi()
{
    return m_grid_roi;
}

torch::Tensor OccupancyGrid::get_grid_occupancy_in_roi()
{
    return m_grid_occupancy.masked_select(m_grid_roi);
}

int OccupancyGrid::get_nr_occupied_voxels()
{
    int nr_occupied_voxels = get_grid_occupancy().sum().item<int>();
    return nr_occupied_voxels;
}

int OccupancyGrid::get_nr_occupied_voxels_in_roi()
{
    int nr_occupied_voxels = get_grid_occupancy_in_roi().sum().item<int>();
    return nr_occupied_voxels;
}

float OccupancyGrid::get_grid_max_value()
{
    return m_grid_values.max().item<float>();
}

float OccupancyGrid::get_grid_min_value()
{
    return m_grid_values.min().item<float>();
}

float OccupancyGrid::get_grid_max_value_in_roi()
{
    return m_grid_values.masked_select(m_grid_roi).max().item<float>();
}

float OccupancyGrid::get_grid_min_value_in_roi()
{
    return m_grid_values.masked_select(m_grid_roi).min().item<float>();
}

void OccupancyGrid::set_grid_values(const torch::Tensor &grid_values)
{
    m_grid_values = grid_values;
}

void OccupancyGrid::set_grid_occupancy(const torch::Tensor &grid_occupancy)
{
    m_grid_occupancy = grid_occupancy;
}

void OccupancyGrid::init_sphere_roi(const float radius, const float padding)
{
    // assumption: sphere is centered at the origin
    auto samples = get_grid_lower_left_voxels_vertices();
    torch::Tensor ll_vertices = std::get<0>(samples);
    // std::cout << "ll_vertices shape is " << grid_points.sizes() << std::endl;
    // torch::Tensor points_indices = std::get<1>(samples);
    // std::cout << "points_indices shape is " << grid_indices.sizes() << std::endl;
    // print grid_points[0]
    // std::cout << "grid_points[0] is " << grid_points[0] << std::endl;
    // print grid_indices[0]
    // std::cout << "grid_indices[0] is " << grid_indices[0] << std::endl;
    // get all voxels vertices (N,8,3)
    torch::Tensor grid_voxels_vertices = get_grid_all_voxels_vertices(ll_vertices);
    // print shape
    // std::cout << "grid_voxels_vertices shape is " << grid_voxels_vertices.sizes() << std::endl;
    // reshape to (N*8,3)
    grid_voxels_vertices = grid_voxels_vertices.reshape({-1, 3});
    // check if vertices are inside sphere (N, 8)
    auto points = grid_voxels_vertices;
    // check if points are inside sphere
    torch::Tensor points_in_sphere_coords = points;
    torch::Tensor point_dist_from_center_of_sphere = points_in_sphere_coords.norm(2, 1, true); // l2norm, dim ,keepdim
    torch::Tensor is_point_inside_primitive = point_dist_from_center_of_sphere < (radius - padding);
    // reshape to (N,8)
    is_point_inside_primitive = is_point_inside_primitive.reshape({-1, 8});
    // check if all vertices are inside sphere (N)
    torch::Tensor are_all_vertices_inside_sphere = is_point_inside_primitive.all(-1);
    // set grid roi
    m_grid_roi = are_all_vertices_inside_sphere;
    // printing
    int nr_voxels = get_nr_voxels();
    std::cout << "nr voxels " << nr_voxels << std::endl;
    std::cout << "nr voxels in roi " << get_nr_voxels_in_roi() << std::endl;
}

void OccupancyGrid::set_grid_occupancy_full()
{
    m_grid_occupancy.fill_(true);
}

void OccupancyGrid::set_grid_occupancy_empty()
{
    m_grid_occupancy.fill_(false);
}

torch::Tensor OccupancyGrid::make_grid_values(const int nr_voxels_per_dim)
{
    // https://stackoverflow.com/a/108360
    CHECK(nr_voxels_per_dim % 2 == 0) << "We are expecting an even number of voxels because we consider the value of the voxel to live a the center of the cube. We need to have even numbers because we are using morton codes";
    CHECK((nr_voxels_per_dim & (nr_voxels_per_dim - 1)) == 0) << "Nr of voxels should be power of 2 because we are using morton codes";
    torch::Tensor grid = torch::ones({nr_voxels_per_dim * nr_voxels_per_dim * nr_voxels_per_dim}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    // std::cout << "grid values shape is " << grid.sizes() << std::endl;
    return grid;
}

torch::Tensor OccupancyGrid::make_grid_occupancy(const int nr_voxels_per_dim)
{
    // https://stackoverflow.com/a/108360
    CHECK(nr_voxels_per_dim % 2 == 0) << "We are expecting an even number of voxels because we consider the value of the voxel to live a the center of the cube. We need to have even numbers because we are using morton codes";
    CHECK((nr_voxels_per_dim & (nr_voxels_per_dim - 1)) == 0) << "Nr of voxels should be power of 2 because we are using morton codes";

    torch::Tensor grid = torch::ones({nr_voxels_per_dim * nr_voxels_per_dim * nr_voxels_per_dim}, torch::dtype(torch::kBool).device(torch::kCUDA, 0));
    return grid;
}

torch::Tensor OccupancyGrid::get_grid_all_voxels_vertices(const torch::Tensor &ll_vertices){
    // get all voxels vertices (N,8,3)
    Eigen::Vector3f voxel_size = m_grid_extent / m_nr_voxels_per_dim;
    torch::Tensor voxel_size_tensor = eigen2tensor(voxel_size).squeeze(1).cuda(); // make it size 3 only
    // cast to torch::Tensor
    torch::Tensor vertices_offsets = torch::tensor(
                                                {
                                                    {0, 0, 0},
                                                    {0, 0, 1},
                                                    {0, 1, 0},
                                                    {0, 1, 1},
                                                    {1, 0, 0},
                                                    {1, 0, 1},
                                                    {1, 1, 0},
                                                    {1, 1, 1},
                                                },
                                                torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    vertices_offsets = vertices_offsets * voxel_size_tensor;
    // reshape for sum
    torch::Tensor grid_voxels_vertices =  ll_vertices.view({-1, 1, 3}) + vertices_offsets.view({1, -1, 3});
    return grid_voxels_vertices;
}

std::tuple<torch::Tensor, torch::Tensor> OccupancyGrid::get_grid_lower_left_voxels_vertices()
{

    int nr_voxels = get_nr_voxels();
    torch::Tensor ll_vertices = torch::empty({nr_voxels, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    torch::Tensor point_indices = torch::arange(0, nr_voxels, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));

    const dim3 blocks = {(unsigned int)div_round_up(nr_voxels, BLOCK_SIZE), 1, 1};

    OccupancyGridGPU::get_grid_lower_left_voxels_vertices_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_voxels,
        m_nr_voxels_per_dim,
        m_grid_extent,
        point_indices.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        // output
        ll_vertices.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    // wait for the kernel to finish
    cudaDeviceSynchronize();
    // check if any error occurred, if so print it
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    return std::make_tuple(ll_vertices, point_indices);
}

std::tuple<torch::Tensor, torch::Tensor> OccupancyGrid::get_grid_samples(const bool jitter_samples)
{

    int nr_voxels = get_nr_voxels();
    torch::Tensor grid_points = torch::empty({nr_voxels, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    torch::Tensor point_indices = torch::arange(0, nr_voxels, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));

    const dim3 blocks = {(unsigned int)div_round_up(nr_voxels, BLOCK_SIZE), 1, 1};

    OccupancyGridGPU::get_grid_samples_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_voxels,
        m_nr_voxels_per_dim,
        m_grid_extent,
        point_indices.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        m_rng,
        jitter_samples,
        // output
        grid_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    // wait for the kernel to finish
    cudaDeviceSynchronize();
    // check if any error occurred, if so print it
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    if (jitter_samples)
    {
        m_rng.advance();
    }

    return std::make_tuple(grid_points, point_indices);
}

std::tuple<torch::Tensor, torch::Tensor> OccupancyGrid::get_random_grid_samples(const int nr_voxels_to_select, const bool jitter_samples)
{
    int nr_voxels = get_nr_voxels();
    torch::Tensor grid_points = torch::empty({nr_voxels_to_select, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    torch::Tensor point_indices = torch::randint(0, nr_voxels, {nr_voxels_to_select}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
    // std::cout << "point_indices shape is " << point_indices.sizes() << std::endl;

    const dim3 blocks = {(unsigned int)div_round_up(nr_voxels_to_select, BLOCK_SIZE), 1, 1};

    OccupancyGridGPU::get_grid_samples_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_voxels_to_select,
        m_nr_voxels_per_dim,
        m_grid_extent,
        point_indices.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        m_rng,
        jitter_samples,
        // output
        grid_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    // wait for the kernel to finish
    cudaDeviceSynchronize();
    // check if any error occurred, if so print it
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    if (jitter_samples)
    {
        m_rng.advance();
    }

    return std::make_tuple(grid_points, point_indices);
}

std::tuple<torch::Tensor, torch::Tensor> OccupancyGrid::get_random_grid_samples_in_roi(const int nr_voxels_to_select, const bool jitter_samples)
{
    // get all voxels indices in roi
    torch::Tensor roi_voxels_indices = torch::nonzero(m_grid_roi).to(torch::kInt32);
    // random sample of nr_voxels_to_select voxels from roi_voxels_indices
    torch::Tensor random_indices = torch::randint(0, roi_voxels_indices.size(0), {nr_voxels_to_select}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
    torch::Tensor point_indices = roi_voxels_indices.index_select(0, random_indices).squeeze(1);
    torch::Tensor grid_points = torch::empty({nr_voxels_to_select, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    
    const dim3 blocks = {(unsigned int)div_round_up(nr_voxels_to_select, BLOCK_SIZE), 1, 1};

    OccupancyGridGPU::get_grid_samples_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_voxels_to_select,
        m_nr_voxels_per_dim,
        m_grid_extent,
        point_indices.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        m_rng,
        jitter_samples,
        // output
        grid_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    // wait for the kernel to finish
    cudaDeviceSynchronize();
    // check if any error occurred, if so print it
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    if (jitter_samples)
    {
        m_rng.advance();
    }

    return std::make_tuple(grid_points, point_indices);
}

std::tuple<torch::Tensor, torch::Tensor> OccupancyGrid::get_rays_t_near_t_far(
    const torch::Tensor &rays_o,
    const torch::Tensor &rays_d,
    const torch::Tensor &ray_t_entry,
    const torch::Tensor &ray_t_exit
) {
    if (verbose)
        std::cout << "get_rays_t_near_t_far" << std::endl;

    CHECK(rays_o.dim() == 2) << "rays_o should have shape nr_raysx3. However it has sizes" << rays_o.sizes();
    CHECK(rays_d.dim() == 2) << "rays_d should have shape nr_raysx3. However it has sizes" << rays_d.sizes();
    CHECK(ray_t_entry.dim() == 2) << "ray_t_entry should have shape nr_raysx1. However it has sizes" << ray_t_entry.sizes();
    CHECK(ray_t_exit.dim() == 2) << "ray_t_exit should have shape nr_raysx1. However it has sizes" << ray_t_exit.sizes();

    int nr_rays = rays_o.size(0);
    torch::Tensor ray_grid_t_near = torch::empty({nr_rays, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    torch::Tensor ray_grid_t_exit = torch::empty({nr_rays, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

    const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};

    OccupancyGridGPU::get_rays_t_near_t_far_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_rays,
        // input
        rays_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        rays_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_t_entry.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_t_exit.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // occupancy grid
        m_nr_voxels_per_dim,
        m_grid_extent,
        m_grid_occupancy.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
        m_grid_roi.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
        // output
        ray_grid_t_near.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_grid_t_exit.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    // wait for the kernel to finish
    cudaDeviceSynchronize();
    // check if any error occurred, if so print it
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
    else {
        if (verbose)
            std::cout << "No error" << std::endl;
    }

    return std::tuple(ray_grid_t_near, ray_grid_t_exit);
}

std::tuple<torch::Tensor, torch::Tensor> OccupancyGrid::check_occupancy(const torch::Tensor &points)
{

    CHECK(points.scalar_type() == at::kFloat) << "positions should be of type float";
    CHECK(points.dim() == 2) << "positions should have shape (nr_points, 3). However it has sizes" << points.sizes();
    
    // // points must be in range [-m_grid_extent_tensor/2, m_grid_extent_tensor/2)
    // bool are_points_inside_bb = (points >= -m_grid_extent_tensor/2).all(1).all().item<bool>();
    // are_points_inside_bb = are_points_inside_bb && (points < m_grid_extent_tensor/2).all(1).all().item<bool>();
    // std::cout << "are_points_inside_bb " << are_points_inside_bb << std::endl;
    // CHECK(are_points_inside_bb) << "points outside the occupancy grid bounding box";

    int nr_points = points.size(0);

    torch::Tensor voxel_occupancy = torch::ones({nr_points, 1}, torch::dtype(torch::kBool).device(torch::kCUDA, 0));
    torch::Tensor voxel_value = torch::ones({nr_points, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

    const dim3 blocks = {(unsigned int)div_round_up(nr_points, BLOCK_SIZE), 1, 1};

    OccupancyGridGPU::check_occupancy_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_points,
        m_nr_voxels_per_dim,
        m_grid_extent,
        m_grid_values.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        m_grid_occupancy.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
        m_grid_roi.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
        points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // output
        voxel_occupancy.packed_accessor32<bool, 2, torch::RestrictPtrTraits>(),
        voxel_value.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    // wait for the kernel to finish
    cudaDeviceSynchronize();
    // check if any error occurred, if so print it
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    return std::tuple(voxel_occupancy, voxel_value);
}

void OccupancyGrid::update_grid_values(const torch::Tensor &point_indices, const torch::Tensor &values, const float decay)
{

    CHECK(values.dim() == 2) << "values should have shape nr_pointsx1. However it has sizes" << values.sizes();
    CHECK(decay <= 1.0) << "We except the decay to be < 1.0 but it is " << decay;
    CHECK(point_indices.dim() == 1) << "point_indices should have dim 1 correspondin to nr_points. However it has sizes" << point_indices.sizes();

    int nr_points = point_indices.size(0);

    const dim3 blocks = {(unsigned int)div_round_up(nr_points, BLOCK_SIZE), 1, 1};

    OccupancyGridGPU::update_grid_values_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_points,
        values.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        m_nr_voxels_per_dim,
        point_indices.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        decay,
        m_grid_values.packed_accessor32<float, 1, torch::RestrictPtrTraits>()
    );

    // wait for the kernel to finish
    cudaDeviceSynchronize();
    // check if any error occurred, if so print it
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
}

void OccupancyGrid::update_grid_occupancy_with_density_values(const torch::Tensor &point_indices, const float occupancy_tresh, const bool check_neighbours)
{
    CHECK(point_indices.dim() == 1) << "point_indices should have dim 1 correspondin to nr_points. However it has sizes" << point_indices.sizes();

    int nr_points = point_indices.size(0);

    const dim3 blocks = {(unsigned int)div_round_up(nr_points, BLOCK_SIZE), 1, 1};

    OccupancyGridGPU::update_grid_occupancy_with_density_values_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_points,
        m_nr_voxels_per_dim,
        m_grid_extent,
        point_indices.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        occupancy_tresh,
        check_neighbours,
        m_grid_values.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        m_grid_occupancy.packed_accessor32<bool, 1, torch::RestrictPtrTraits>()
    );

    // wait for the kernel to finish
    cudaDeviceSynchronize();
    // check if any error occurred, if so print it
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
}

void OccupancyGrid::update_grid_occupancy_with_sdf_values(const torch::Tensor &point_indices, const torch::Tensor &logistic_beta, const float occupancy_thresh, const bool check_neighbours)
{
    CHECK(point_indices.dim() == 1) << "point_indices should have dim 1 correspondin to nr_points. However it has sizes" << point_indices.sizes();

    int nr_points = point_indices.size(0);

    const dim3 blocks = {(unsigned int)div_round_up(nr_points, BLOCK_SIZE), 1, 1};

    OccupancyGridGPU::update_grid_occupancy_with_sdf_values_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_points,
        m_grid_extent,
        m_nr_voxels_per_dim,
        point_indices.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        logistic_beta.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        occupancy_thresh,
        check_neighbours,
        m_grid_values.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        m_grid_occupancy.packed_accessor32<bool, 1, torch::RestrictPtrTraits>()
    );

    // wait for the kernel to finish
    cudaDeviceSynchronize();
    // check if any error occurred, if so print it
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
}

// TODO: move to RaySampler
RaySamplesPacked OccupancyGrid::get_first_rays_sample_start_of_grid_occupied_regions(const torch::Tensor &rays_o, const torch::Tensor &rays_d, const torch::Tensor &ray_t_entry, const torch::Tensor &ray_t_exit)
{
    int nr_rays = rays_o.size(0);
    RaySamplesPacked ray_samples_packed(nr_rays, nr_rays, 0, 1);

    //fill the samples
    const dim3 blocks = { (unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1 }; 

    OccupancyGridGPU::get_first_rays_sample_start_of_grid_occupied_regions_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_rays,
        m_nr_voxels_per_dim,
        m_grid_extent,
        rays_o.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        rays_d.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        ray_t_entry.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        ray_t_exit.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        m_grid_occupancy.packed_accessor32<bool,1,torch::RestrictPtrTraits>(),
        m_grid_roi.packed_accessor32<bool,1,torch::RestrictPtrTraits>(),
        //output
        ray_samples_packed.samples_3d.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        ray_samples_packed.samples_dirs.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        ray_samples_packed.samples_z.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        ray_samples_packed.samples_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>()
    );
    
    // wait for the kernel to finish
    cudaDeviceSynchronize();
    // check if any error occurred, if so print it
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    return ray_samples_packed;
}

// TODO: move to RaySampler
std::tuple<torch::Tensor, torch::Tensor> OccupancyGrid::advance_ray_sample_to_next_occupied_voxel(const torch::Tensor &samples_dirs, const torch::Tensor &samples_3d)
{
    int nr_points = samples_3d.size(0);

    //fill the samples
    const dim3 blocks = { (unsigned int)div_round_up(nr_points, BLOCK_SIZE), 1, 1 }; 

    torch::Tensor new_samples_3d=samples_3d;
    torch::Tensor is_within_bounds=torch::ones({ nr_points, 1 }, torch::dtype(torch::kBool).device(torch::kCUDA, 0)  );

    OccupancyGridGPU::advance_ray_sample_to_next_occupied_voxel_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_points,
        m_nr_voxels_per_dim,
        m_grid_extent,
        samples_dirs.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        samples_3d.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        m_grid_occupancy.packed_accessor32<bool,1,torch::RestrictPtrTraits>(),
        m_grid_roi.packed_accessor32<bool,1,torch::RestrictPtrTraits>(),
        //output
        new_samples_3d.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        is_within_bounds.packed_accessor32<bool,2,torch::RestrictPtrTraits>()
    );

    // wait for the kernel to finish
    cudaDeviceSynchronize();
    // check if any error occurred, if so print it
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    return std::make_tuple(new_samples_3d, is_within_bounds);
}