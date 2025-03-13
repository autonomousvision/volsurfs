#include "volsurfs/RaySampler.cuh"

// c++
//  #include <string>

#include "volsurfs/UtilsPytorch.h"

// my stuff
#include "volsurfs/RaySamplerGPU.cuh"

using torch::Tensor;

template <typename T>
T div_round_up(T val, T divisor)
{
    return (val + divisor - 1) / divisor;
}

pcg32 RaySampler::m_rng;

// CPU code that calls the kernels
RaySampler::RaySampler()
{
}

RaySampler::~RaySampler()
{
}

RaySamplesPacked RaySampler::init_with_one_sample_per_ray(const torch::Tensor samples_3d, const torch::Tensor samples_dir)
{
    if (verbose)
        std::cout << "init_with_one_sample_per_ray" << std::endl;

    int nr_rays = samples_3d.size(0);
    int max_nr_samples = nr_rays;
    RaySamplesPacked ray_samples_packed(nr_rays, max_nr_samples, 0, 1);

    unsigned int nr_blocks = (unsigned int) div_round_up(nr_rays, BLOCK_SIZE);  // 256
    const dim3 blocks = {nr_blocks, 1, 1};

    RaySamplerGPU::init_with_one_sample_per_ray_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_rays,
        samples_3d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        samples_dir.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // output
        ray_samples_packed.samples_3d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.samples_dirs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.samples_z.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.samples_dt.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>()
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

    return ray_samples_packed;
}

RaySamplesPacked RaySampler::compute_samples_bg(
    const torch::Tensor &rays_o,
    const torch::Tensor &rays_d,
    const torch::Tensor &ray_t_start,
    const float ray_t_far,
    const int nr_samples_per_ray,
    // const float scene_radius,
    // const torch::Tensor &sphere_center,
    const bool jitter_samples
    // const bool contract_3d_samples,
    // const int values_dim
) {
    if (verbose)
        std::cout << "compute_samples_bg" << std::endl;

    CHECK(rays_o.dim() == 2) << "rays_o should have shape nr_raysx3. However it has sizes" << rays_o.sizes();
    CHECK(rays_d.dim() == 2) << "rays_d should have shape nr_raysx3. However it has sizes" << rays_d.sizes();
    CHECK(ray_t_start.dim() == 2) << "ray_t_start should have shape nr_raysx1. However it has sizes" << ray_t_start.sizes();

    int nr_rays = rays_o.size(0);
    int max_nr_samples = nr_rays * nr_samples_per_ray;
    RaySamplesPacked ray_samples_packed(nr_rays, max_nr_samples, 0, 0);
    ray_samples_packed.ray_o = rays_o.clone();
    ray_samples_packed.ray_d = rays_d.clone();
    ray_samples_packed.ray_enter = ray_t_start.clone();
    ray_samples_packed.ray_exit = torch::full({nr_rays, 1}, ray_t_far, torch::dtype(torch::kFloat32));
    ray_samples_packed.is_compacted = true; // always compacted

    // view them a bit different because it's easier to fill them
    ray_samples_packed.samples_z = ray_samples_packed.samples_z.view({nr_rays, nr_samples_per_ray});
    ray_samples_packed.samples_dt = ray_samples_packed.samples_dt.view({nr_rays, nr_samples_per_ray});
    ray_samples_packed.samples_3d = ray_samples_packed.samples_3d.view({nr_rays, nr_samples_per_ray, 3});
    ray_samples_packed.samples_dirs = ray_samples_packed.samples_dirs.view({nr_rays, nr_samples_per_ray, 3});

    const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};

    RaySamplerGPU::compute_samples_bg_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_rays,
        nr_samples_per_ray,
        ray_samples_packed.ray_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.ray_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.ray_enter.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_t_far,
        m_rng,
        jitter_samples,
        // output
        ray_samples_packed.ray_max_dt.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.samples_3d.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        ray_samples_packed.samples_dirs.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        ray_samples_packed.samples_z.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.samples_dt.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>()
        // ray_samples_packed.ray_enter.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // ray_samples_packed.ray_exit.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
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

    if (jitter_samples)
    {
        m_rng.advance();
    }

    ray_samples_packed.samples_z = ray_samples_packed.samples_z.view({max_nr_samples, 1});
    ray_samples_packed.samples_dt = ray_samples_packed.samples_dt.view({max_nr_samples, 1});
    ray_samples_packed.samples_3d = ray_samples_packed.samples_3d.view({max_nr_samples, 3});
    ray_samples_packed.samples_dirs = ray_samples_packed.samples_dirs.view({max_nr_samples, 3});

    if (verbose) {
        std::cout << "nr_rays " << ray_samples_packed.get_nr_rays() << std::endl;
        std::cout << "max_nr_samples " << ray_samples_packed.get_max_nr_samples() << std::endl;
        std::cout << "nr_samples " << ray_samples_packed.get_total_nr_samples() << std::endl;
    }

    return ray_samples_packed;
}

RaySamplesPacked RaySampler::compute_samples_fg(
    const torch::Tensor &rays_o,
    const torch::Tensor &rays_d,
    const torch::Tensor &ray_t_entry,
    const torch::Tensor &ray_t_exit,
    const float min_dist_between_samples,
    const int min_nr_samples_per_ray,
    const int max_nr_samples_per_ray,
    const bool jitter_samples,
    const int values_dim
){
    if (verbose)
        std::cout << "compute_samples_fg" << std::endl;

    CHECK(rays_o.dim() == 2) << "rays_o should have shape nr_raysx3. However it has sizes" << rays_o.sizes();
    CHECK(rays_d.dim() == 2) << "rays_d should have shape nr_raysx3. However it has sizes" << rays_d.sizes();
    CHECK(ray_t_entry.dim() == 2) << "ray_t_entry should have shape nr_raysx1. However it has sizes" << ray_t_entry.sizes();
    CHECK(ray_t_exit.dim() == 2) << "ray_t_exit should have shape nr_raysx1. However it has sizes" << ray_t_exit.sizes();

    int nr_rays = rays_o.size(0);
    int max_nr_samples = nr_rays * max_nr_samples_per_ray;
    RaySamplesPacked ray_samples_packed(nr_rays, max_nr_samples, 0, values_dim);
    ray_samples_packed.ray_o = rays_o.clone();
    ray_samples_packed.ray_d = rays_d.clone();
    ray_samples_packed.ray_enter = ray_t_entry.clone();
    ray_samples_packed.ray_exit = ray_t_exit.clone();
    ray_samples_packed.is_compacted = false;
    
    const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};

    RaySamplerGPU::compute_samples_fg_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_rays,
        ray_samples_packed.ray_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.ray_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.ray_enter.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.ray_exit.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        min_dist_between_samples,
        min_nr_samples_per_ray,
        max_nr_samples_per_ray,
        ray_samples_packed.get_max_nr_samples(),
        m_rng,
        jitter_samples,
        // output
        ray_samples_packed.ray_max_dt.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.samples_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.samples_3d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.samples_dirs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.samples_z.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.samples_dt.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>()
        // ray_samples_packed.ray_enter.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // ray_samples_packed.ray_exit.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
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

    if (jitter_samples)
    {
        m_rng.advance();
    }

    if (verbose) {
        std::cout << "nr_rays " << ray_samples_packed.get_nr_rays() << std::endl;
        std::cout << "max_nr_samples " << ray_samples_packed.get_max_nr_samples() << std::endl;
        std::cout << "nr_samples " << ray_samples_packed.get_total_nr_samples() << std::endl;
    }
    
    ray_samples_packed = ray_samples_packed.compact_to_valid_samples();

    CHECK(ray_samples_packed.is_compacted) << "RaySamplesPacked should be compacted at the end of compute_samples_fg";

    return ray_samples_packed;
}

RaySamplesPacked RaySampler::compute_samples_fg_in_grid_occupied_regions(
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
) {
    if (verbose)
        std::cout << "compute_samples_fg_in_grid_occupied_regions" << std::endl;

    CHECK(rays_o.dim() == 2) << "rays_o should have shape nr_raysx3. However it has sizes" << rays_o.sizes();
    CHECK(rays_d.dim() == 2) << "rays_d should have shape nr_raysx3. However it has sizes" << rays_d.sizes();
    CHECK(ray_t_entry.dim() == 2) << "ray_t_entry should have shape nr_raysx1. However it has sizes" << ray_t_entry.sizes();
    CHECK(ray_t_exit.dim() == 2) << "ray_t_exit should have shape nr_raysx1. However it has sizes" << ray_t_exit.sizes();

    int nr_rays = rays_o.size(0);
    int max_nr_samples = nr_rays * max_nr_samples_per_ray;
    RaySamplesPacked ray_samples_packed(nr_rays, max_nr_samples, 0, values_dim);
    ray_samples_packed.ray_o = rays_o.clone();
    ray_samples_packed.ray_d = rays_d.clone();
    ray_samples_packed.ray_enter = ray_t_entry.clone();
    ray_samples_packed.ray_exit = ray_t_exit.clone();
    ray_samples_packed.is_compacted = false;

    const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};

    RaySamplerGPU::compute_samples_fg_in_grid_occupied_regions_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_rays,
        // input
        ray_samples_packed.ray_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.ray_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.ray_enter.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.ray_exit.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        min_dist_between_samples,
        min_nr_samples_per_ray,
        max_nr_samples_per_ray,
        ray_samples_packed.get_max_nr_samples(),
        m_rng,
        jitter_samples,
        // occupancy grid
        nr_voxels_per_dim, // m_nr_voxels_per_dim
        grid_extent, // m_grid_extent
        grid_occupancy.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(), // m_grid_occupancy
        grid_roi.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(), // m_grid_roi
        // output
        ray_samples_packed.ray_max_dt.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.samples_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.samples_3d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.samples_dirs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.samples_z.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.samples_dt.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>()
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

    if (jitter_samples)
    {
        m_rng.advance();
    }

    if (verbose) {
        std::cout << "nr_rays " << ray_samples_packed.get_nr_rays() << std::endl;
        std::cout << "max_nr_samples " << ray_samples_packed.get_max_nr_samples() << std::endl;
        std::cout << "nr_samples " << ray_samples_packed.get_total_nr_samples() << std::endl;
    }
    
    ray_samples_packed = ray_samples_packed.compact_to_valid_samples();

    CHECK(ray_samples_packed.is_compacted) << "RaySamplesPacked should be compacted at the end of compute_samples_fg_in_grid_occupied_regions";

    return ray_samples_packed;
}

RaySamplesPacked RaySampler::contract_samples(
    const RaySamplesPacked &uncontracted_ray_samples_packed
) {
    if (verbose)
        std::cout << "contract_samples" << std::endl;

    CHECK(!uncontracted_ray_samples_packed.is_empty()) << "RaySamplesPacked must not be empty before calling contract_samples";
    CHECK(uncontracted_ray_samples_packed.is_compacted) << "RaySamplesPacked should be compacted at the beginning of contract_samples";

    // copy current ray_samples_packed to a new one
    int nr_rays = uncontracted_ray_samples_packed.get_nr_rays();
    RaySamplesPacked contracted_ray_samples_packed = uncontracted_ray_samples_packed.copy();

    // call kernel
    const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};

    RaySamplerGPU::contract_samples_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_rays,
        // input
        uncontracted_ray_samples_packed.ray_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        uncontracted_ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        uncontracted_ray_samples_packed.samples_3d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        uncontracted_ray_samples_packed.samples_z.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // output
        contracted_ray_samples_packed.samples_3d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        contracted_ray_samples_packed.samples_z.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
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

    // update dt
    contracted_ray_samples_packed.update_dt(true);

    return contracted_ray_samples_packed;
}

RaySamplesPacked RaySampler::uncontract_samples(
    const RaySamplesPacked &contracted_ray_samples_packed
) {
    if (verbose)
        std::cout << "uncontract_samples" << std::endl;
    
    CHECK(!contracted_ray_samples_packed.is_empty()) << "RaySamplesPacked must not be empty before calling uncontract_samples";
    CHECK(contracted_ray_samples_packed.is_compacted) << "RaySamplesPacked should be compacted at the beginning of contract_samples";

    // copy current ray_samples_packed to a new one
    int nr_rays = contracted_ray_samples_packed.get_nr_rays();
    RaySamplesPacked uncontracted_ray_samples_packed = contracted_ray_samples_packed.copy();

    // call kernel
    const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};

    RaySamplerGPU::uncontract_samples_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_rays,
        // input
        contracted_ray_samples_packed.ray_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        contracted_ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        contracted_ray_samples_packed.samples_3d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        contracted_ray_samples_packed.samples_z.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // output
        uncontracted_ray_samples_packed.samples_3d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        uncontracted_ray_samples_packed.samples_z.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
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

    // update dt
    uncontracted_ray_samples_packed.update_dt(true);

    return uncontracted_ray_samples_packed;
}