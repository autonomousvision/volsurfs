#include "volsurfs/VolumeRendering.cuh"

// c++
//  #include <string>

#include "volsurfs/UtilsPytorch.h"

// my stuff
#include "volsurfs/VolumeRenderingGPU.cuh"

using torch::Tensor;

template <typename T>
T div_round_up(T val, T divisor)
{
    return (val + divisor - 1) / divisor;
}

pcg32 VolumeRendering::m_rng;

// CPU code that calls the kernels
VolumeRendering::VolumeRendering()
{
}

VolumeRendering::~VolumeRendering()
{
}

std::tuple<torch::Tensor, torch::Tensor> VolumeRendering::cumprod_one_minus_alpha_to_transmittance(const RaySamplesPacked &ray_samples_packed, const torch::Tensor &alpha_samples)
{
    if (verbose)
        std::cout << "cumprod_one_minus_alpha_to_transmittance" << std::endl;

    CHECK(ray_samples_packed.is_compacted) << "RaySamplesPacked must be compacted before calling cumprod";

    // int nr_rays = ray_samples_packed.ray_start_end_idx.size(0);
    // int nr_samples_total = ray_samples_packed.samples_z.size(0);
    int nr_rays = ray_samples_packed.get_nr_rays();
    int nr_samples_total = ray_samples_packed.get_total_nr_samples();

    // fill the samples
    const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};

    torch::Tensor transmittance_samples = torch::zeros({nr_samples_total, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    // if we have no samples for this ray, then the bg_transmittance stays as 1 so the background can be fully visible
    torch::Tensor bg_transmittance = torch::ones({nr_rays, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

    if (ray_samples_packed.is_empty())
    {
        return std::make_tuple(transmittance_samples, bg_transmittance);
    }

    VolumeRenderingGPU::cumprod_one_minus_alpha_to_transmittance_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_rays,
        ray_samples_packed.get_max_nr_samples(), // useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
        ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        alpha_samples.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // output
        transmittance_samples.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        bg_transmittance.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
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

    return std::make_tuple(transmittance_samples, bg_transmittance);
}

torch::Tensor VolumeRendering::integrate_with_weights_1d(
    const RaySamplesPacked &ray_samples_packed,
    const torch::Tensor &values,
    const torch::Tensor &weights)
{
    if (verbose)
        std::cout << "integrate_with_weights_1d" << std::endl;

    CHECK(ray_samples_packed.is_compacted) << "RaySamplesPacked must be compacted before calling integrate_with_weights_1d";

    // int nr_rays = ray_samples_packed.ray_start_end_idx.size(0);
    int nr_rays = ray_samples_packed.get_nr_rays();

    // fill the samples
    const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};

    torch::Tensor outputs = torch::zeros({nr_rays, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    
    if (ray_samples_packed.is_empty())
    {
        return outputs;
    }

    VolumeRenderingGPU::integrate_with_weights_1d_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_rays,
        ray_samples_packed.get_max_nr_samples(), // useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
        ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        values.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        weights.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // output
        outputs.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
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

    return outputs;
}

torch::Tensor VolumeRendering::integrate_with_weights_3d(
    const RaySamplesPacked &ray_samples_packed,
    const torch::Tensor &values,
    const torch::Tensor &weights)
{
    if (verbose)
        std::cout << "integrate_with_weights_3d" << std::endl;

    CHECK(ray_samples_packed.is_compacted) << "RaySamplesPacked must be compacted before calling integrate_with_weights_3d";

    // int nr_rays = ray_samples_packed.ray_start_end_idx.size(0);
    int nr_rays = ray_samples_packed.get_nr_rays();

    // fill the samples
    const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};

    torch::Tensor outputs = torch::zeros({nr_rays, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    
    if (ray_samples_packed.is_empty())
    {
        return outputs;
    }

    VolumeRenderingGPU::integrate_with_weights_3d_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_rays,
        ray_samples_packed.get_max_nr_samples(), // useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
        ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        values.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        weights.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // output
        outputs.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
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

    return outputs;
}

torch::Tensor VolumeRendering::sdf2alpha(
    const RaySamplesPacked &ray_samples_packed,
    const torch::Tensor &samples_sdf,
    const torch::Tensor &logistic_beta
)
{
    if (verbose)
        std::cout << "sdf2alpha" << std::endl;

    CHECK(ray_samples_packed.is_compacted) << "RaySamplesPacked must be compacted before calling sdf2alpha";
    CHECK(ray_samples_packed.has_dt) << "ray_samples_packed should have dt";

    int nr_rays = ray_samples_packed.get_nr_rays();
    int nr_samples_total = ray_samples_packed.get_total_nr_samples();

    // fill the samples
    const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};

    torch::Tensor alpha_samples = torch::zeros({nr_samples_total, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

    if (ray_samples_packed.is_empty())
    {
        std::cout << "Empty ray_samples_packed" << std::endl;
        return alpha_samples;
    }

    VolumeRenderingGPU::sdf2alpha_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_rays,
        ray_samples_packed.get_max_nr_samples(), // useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
        ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.samples_dt.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        samples_sdf.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        logistic_beta.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // output
        alpha_samples.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
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

    return alpha_samples;
}

std::tuple<torch::Tensor, torch::Tensor> VolumeRendering::sum_over_rays(const RaySamplesPacked &ray_samples_packed, const torch::Tensor &samples_values)
{
    if (verbose)
        std::cout << "sum_over_rays" << std::endl;

    CHECK(ray_samples_packed.is_compacted) << "RaySamplesPacked must be compacted before calling sum_over_rays";

    // int nr_rays = ray_samples_packed.ray_start_end_idx.size(0);
    // int nr_samples_total = ray_samples_packed.samples_z.size(0);
    int nr_rays = ray_samples_packed.get_nr_rays();
    int nr_samples_total = ray_samples_packed.get_total_nr_samples();
    CHECK(samples_values.size(0) == nr_samples_total) << "samples_values should have size of nr_samples_total x 1. but it has " << samples_values.sizes();
    CHECK(samples_values.size(1) <= 3 || samples_values.size(1) == 32) << "samples_values should have up to 4 values value per sample" << samples_values.sizes();
    int val_dim = samples_values.size(1);

    // fill the samples
    const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};

    torch::Tensor values_sum = torch::zeros({nr_rays, val_dim}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    torch::Tensor values_sum_stored_per_sample = torch::zeros({nr_samples_total, val_dim}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

    if (ray_samples_packed.is_empty())
    {
        return std::make_tuple(values_sum, values_sum_stored_per_sample);
    }

    if (val_dim == 1)
    {
        VolumeRenderingGPU::sum_over_rays_gpu<1><<<blocks, BLOCK_SIZE>>>(
            nr_rays,
            ray_samples_packed.get_max_nr_samples(), // useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
            ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            samples_values.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            // output
            values_sum.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            values_sum_stored_per_sample.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
        );
    }
    else if (val_dim == 2)
    {
        VolumeRenderingGPU::sum_over_rays_gpu<2><<<blocks, BLOCK_SIZE>>>(
            nr_rays,
            ray_samples_packed.get_max_nr_samples(), // useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
            ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            samples_values.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            // output
            values_sum.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            values_sum_stored_per_sample.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
        );
    }
    else if (val_dim == 3)
    {
        VolumeRenderingGPU::sum_over_rays_gpu<3><<<blocks, BLOCK_SIZE>>>(
            nr_rays,
            ray_samples_packed.get_max_nr_samples(), // useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
            ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            samples_values.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            // output
            values_sum.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            values_sum_stored_per_sample.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
        );
    }
    else if (val_dim == 32)
    {
        VolumeRenderingGPU::sum_over_rays_gpu<32><<<blocks, BLOCK_SIZE>>>(
            nr_rays,
            ray_samples_packed.get_max_nr_samples(), // useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
            ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            samples_values.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            // output
            values_sum.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            values_sum_stored_per_sample.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
        );
    }
    else
    {
        LOG(FATAL) << "Val dim not implemented yet";
    }

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

    return std::make_tuple(values_sum, values_sum_stored_per_sample);
}

torch::Tensor VolumeRendering::cumsum_over_rays(const RaySamplesPacked &ray_samples_packed, const torch::Tensor &samples_values, const bool inverse)
{
    if (verbose)
        std::cout << "cumsum_over_rays" << std::endl;

    CHECK(ray_samples_packed.is_compacted) << "RaySamplesPacked must be compacted before calling cumsum_over_rays";

    int nr_rays = ray_samples_packed.get_nr_rays();
    int nr_samples_total = ray_samples_packed.get_total_nr_samples();

    // fill the samples
    const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};

    torch::Tensor values_cumsum_stored_per_sample = torch::zeros({nr_samples_total, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

    if (ray_samples_packed.is_empty())
    {
        return values_cumsum_stored_per_sample;
    }

    VolumeRenderingGPU::cumsum_over_rays_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_rays,
        ray_samples_packed.get_max_nr_samples(), // useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
        ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        samples_values.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        inverse,
        // output
        values_cumsum_stored_per_sample.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
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

    return values_cumsum_stored_per_sample;
}

torch::Tensor VolumeRendering::median_depth_over_rays(const RaySamplesPacked &ray_samples_packed, const torch::Tensor &samples_weights, const float threshold)
{
    if (verbose)
        std::cout << "median_depth" << std::endl;

    CHECK(ray_samples_packed.is_compacted) << "RaySamplesPacked must be compacted before calling median_depth";

    int nr_rays = ray_samples_packed.get_nr_rays();
    int nr_samples_total = ray_samples_packed.get_total_nr_samples();

    torch::Tensor ray_depth = torch::zeros({nr_rays, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    
    if (ray_samples_packed.is_empty())
    {
        return ray_depth;
    }

    // fill the samples
    const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};
    VolumeRenderingGPU::median_depth_over_rays_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_rays,
        ray_samples_packed.get_max_nr_samples(), // useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
        threshold,
        ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.samples_z.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        samples_weights.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // output
        ray_depth.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
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

    return ray_depth;
}

torch::Tensor VolumeRendering::compute_cdf(const RaySamplesPacked &ray_samples_packed, const torch::Tensor &samples_weights)
{
    if (verbose)
        std::cout << "compute_cdf" << std::endl;

    CHECK(ray_samples_packed.is_compacted) << "RaySamplesPacked must be compacted before calling compute_cdf";

    // int nr_rays = ray_samples_packed.ray_start_end_idx.size(0);
    // int nr_samples_total = ray_samples_packed.samples_z.size(0);
    int nr_rays = ray_samples_packed.get_nr_rays();
    int nr_samples_total = ray_samples_packed.get_total_nr_samples();
    CHECK(samples_weights.size(0) == nr_samples_total) << "weights should have size of nr_samples_total x 1. but it has " << samples_weights.sizes();
    CHECK(samples_weights.size(1) == 1) << "weights should have only 1 value per sample " << samples_weights.sizes();

    // fill the samples
    const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};

    torch::Tensor samples_cdf = torch::zeros({nr_samples_total, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

    if (ray_samples_packed.is_empty())
    {
        return samples_cdf;
    }

    VolumeRenderingGPU::compute_cdf_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_rays,
        ray_samples_packed.get_max_nr_samples(), // useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
        ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        samples_weights.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // output
        samples_cdf.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
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

    return samples_cdf;
}

RaySamplesPacked VolumeRendering::importance_sample(
    const RaySamplesPacked &ray_samples_packed,
    const torch::Tensor &samples_cdf,
    const int nr_importance_samples,
    const bool jitter_samples
) {
    if (verbose)
        std::cout << "importance_sample" << std::endl;

    CHECK(!ray_samples_packed.is_empty()) << "RaySamplesPacked should not be empty";
    CHECK(ray_samples_packed.is_compacted) << "RaySamplesPacked must be compacted before calling importance_sample";

    int nr_samples_total = ray_samples_packed.get_total_nr_samples();

    CHECK(samples_cdf.size(0) == nr_samples_total) << "CDF should have size of nr_samples_total x 1. but it has " << samples_cdf.sizes();
    CHECK(samples_cdf.size(1) == 1) << "CDF should have only 1 value per sample " << samples_cdf.sizes();
    
    int nr_rays = ray_samples_packed.get_nr_rays();
    int first_sample_idx = ray_samples_packed.get_max_nr_samples();
    int values_dim = ray_samples_packed.get_values_dim();
    
    int nr_samples_imp_maximum = nr_rays * nr_importance_samples;
    RaySamplesPacked ray_samples_imp(nr_rays, nr_samples_imp_maximum, first_sample_idx, values_dim);
    ray_samples_imp.has_samples_values = false;
    ray_samples_imp.has_dt = false;
    ray_samples_imp.is_compacted = false;

    // fill the samples
    const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};

    VolumeRenderingGPU::importance_sample_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_rays,
        ray_samples_packed.ray_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.ray_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // input_ray_samples_packed
        ray_samples_packed.get_max_nr_samples(),
        ray_samples_imp.get_max_nr_samples(),
        ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed.samples_z.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // cdf
        samples_cdf.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // importance samples
        nr_importance_samples,
        m_rng,
        jitter_samples,
        // output
        ray_samples_imp.samples_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        ray_samples_imp.samples_3d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_imp.samples_dirs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_imp.samples_z.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_imp.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>()
    );

    if (jitter_samples)
    {
        m_rng.advance();
    }

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

    if (verbose) {
        std::cout << "nr_rays " << ray_samples_imp.get_nr_rays() << std::endl;
        std::cout << "max_nr_samples " << ray_samples_imp.get_max_nr_samples() << std::endl;
        std::cout << "nr_samples " << ray_samples_imp.get_total_nr_samples() << std::endl;
    }

    if (verbose) {
        std::cout << "is_compacted " << ray_samples_imp.is_compacted << std::endl;
    }

    ray_samples_imp = ray_samples_imp.compact_to_valid_samples();

    CHECK(ray_samples_imp.is_compacted) << "RaySamplesPacked must be compacted after calling importance_sample";
    CHECK(ray_samples_imp.get_total_nr_samples() > 0) << "nr_samples_imp should be > 0";

    return ray_samples_imp;
}

RaySamplesPacked VolumeRendering::combine_ray_samples_packets(
    const RaySamplesPacked &ray_samples_packed_1,
    const RaySamplesPacked &ray_samples_packed_2,
    const float min_dist_between_samples
) {
    if (verbose)
        std::cout << "combine_ray_samples_packets" << std::endl;

    CHECK(ray_samples_packed_1.is_compacted) << "RaySamplesPacked (1) must be compacted before calling combine_ray_samples_packets";
    CHECK(ray_samples_packed_2.is_compacted) << "RaySamplesPacked (2) must be compacted before calling combine_ray_samples_packets";
    CHECK(ray_samples_packed_1.has_samples_values == ray_samples_packed_2.has_samples_values) << "They are supposed to both has or not have samples values";
    CHECK(ray_samples_packed_1.get_values_dim() == ray_samples_packed_2.get_values_dim()) << "They are supposed to have the same values dims";
    CHECK(ray_samples_packed_1.get_nr_rays() == ray_samples_packed_2.get_nr_rays()) << "They are supposed to have the same number of rays";
    CHECK(!(ray_samples_packed_1.is_empty() && ray_samples_packed_2.is_empty())) << "Both ray_samples_packed are empty";

    if (ray_samples_packed_1.is_empty())
    {
        return ray_samples_packed_2;
    }

    if (ray_samples_packed_2.is_empty())
    {
        return ray_samples_packed_1;
    }

    int nr_rays = ray_samples_packed_1.get_nr_rays();
    int values_dim = ray_samples_packed_1.get_values_dim();

    int nr_samples_1 = ray_samples_packed_1.get_total_nr_samples();
    int nr_samples_2 = ray_samples_packed_2.get_total_nr_samples();
    int max_nr_samples_combined = nr_samples_1 + nr_samples_2;

    RaySamplesPacked ray_samples_combined(nr_rays, max_nr_samples_combined, 0, values_dim);
    ray_samples_combined.ray_o = ray_samples_packed_1.ray_o.clone();
    ray_samples_combined.ray_d = ray_samples_packed_1.ray_d.clone();
    ray_samples_combined.ray_enter = ray_samples_packed_1.ray_enter.clone();
    ray_samples_combined.ray_exit = ray_samples_packed_1.ray_exit.clone();
    ray_samples_combined.has_samples_values = ray_samples_packed_1.has_samples_values;
    ray_samples_combined.has_dt = false;
    ray_samples_combined.ray_max_dt = ray_samples_packed_1.ray_max_dt.clone();
    ray_samples_combined.is_compacted = false;

    torch::Tensor uniform_ray_start_idx = ray_samples_packed_1.ray_start_end_idx.index({torch::indexing::Slice(), 0});
    torch::Tensor uniform_ray_end_idx = ray_samples_packed_1.ray_start_end_idx.index({torch::indexing::Slice(), 1});
    torch::Tensor uniform_nr_samples_per_ray = uniform_ray_end_idx - uniform_ray_start_idx;
    torch::Tensor nr_samples_per_ray_imp = ray_samples_packed_2.get_nr_samples_per_ray();
    // add nr_samples_per_ray_imp to uniform_nr_samples_per_ray
    torch::Tensor combined_nr_samples_per_ray = uniform_nr_samples_per_ray + nr_samples_per_ray_imp;
    torch::Tensor combined_out_indices_start = combined_nr_samples_per_ray.cumsum(0).to(torch::kInt32);
    // drop the last element and add a 0 at the beginning
    combined_out_indices_start = torch::cat({torch::zeros(1, torch::dtype(torch::kInt32).device(torch::kCUDA, 0)), combined_out_indices_start.slice(0, 0, -1)}, 0);

    // fill the samples
    const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};

    VolumeRenderingGPU::combine_ray_samples_packets_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_rays,
        min_dist_between_samples,
        values_dim,
        // input_ray_samples_packed_1
        nr_samples_1,
        ray_samples_packed_1.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed_1.samples_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed_1.samples_3d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed_1.samples_dirs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed_1.samples_z.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed_1.samples_values.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // imp_samples
        nr_samples_2,
        ray_samples_packed_2.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed_2.samples_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed_2.samples_3d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed_2.samples_dirs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed_2.samples_z.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_packed_2.samples_values.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // combined stuff just for sanity checking
        max_nr_samples_combined,
        // where to start writing the output of every ray
        combined_out_indices_start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        // output
        ray_samples_combined.samples_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        ray_samples_combined.samples_3d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_combined.samples_dirs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_combined.samples_z.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_combined.samples_values.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        ray_samples_combined.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>()
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

    if (verbose) {
        std::cout << "nr_rays " << ray_samples_combined.get_nr_rays() << std::endl;
        std::cout << "max_nr_samples " << ray_samples_combined.get_max_nr_samples() << std::endl;
        std::cout << "nr_samples " << ray_samples_combined.get_total_nr_samples() << std::endl;
    }

    ray_samples_combined = ray_samples_combined.compact_to_valid_samples();

    CHECK(ray_samples_combined.is_compacted) << "RaySamplesPacked must be compacted after calling combine_ray_samples_packets";
    CHECK(ray_samples_combined.get_total_nr_samples() > 0) << "total_nr_samples should be > 0";

    return ray_samples_combined;
}

torch::Tensor VolumeRendering::cumprod_one_minus_alpha_to_transmittance_backward(const torch::Tensor &grad_transmittance, const torch::Tensor &grad_bg_transmittance, const RaySamplesPacked &ray_samples_packed, const torch::Tensor &alpha, const torch::Tensor &transmittance, const torch::Tensor &bg_transmittance, const torch::Tensor &cumsumLV)
{
    if (verbose)
        std::cout << "cumprod_one_minus_alpha_to_transmittance_backward" << std::endl;

    int nr_rays = ray_samples_packed.get_nr_rays();
    int nr_samples_total = ray_samples_packed.get_total_nr_samples();
    CHECK(grad_transmittance.size(0) == nr_samples_total) << "grad_transmittance should have size of nr_samples_total x 1. but it has " << grad_transmittance.sizes();

    // fill the samples
    const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};

    torch::Tensor grad_alpha_samples = torch::zeros({nr_samples_total, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

    if (ray_samples_packed.is_empty())
    {
        return grad_alpha_samples;
    }

    VolumeRenderingGPU::cumprod_one_minus_alpha_to_transmittance_backward_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_rays,
        ray_samples_packed.get_max_nr_samples(), // useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
        ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        grad_transmittance.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        grad_bg_transmittance.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        alpha.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        transmittance.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        bg_transmittance.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        cumsumLV.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // output
        grad_alpha_samples.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
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

    return grad_alpha_samples;
}

std::tuple<torch::Tensor, torch::Tensor> VolumeRendering::integrate_with_weights_1d_backward(const torch::Tensor &grad_result, const RaySamplesPacked &ray_samples_packed, const torch::Tensor &values, const torch::Tensor &weights, const torch::Tensor &result)
{
    if (verbose)
        std::cout << "integrate_with_weights_1d_backward" << std::endl;

    int nr_rays = ray_samples_packed.get_nr_rays();
    int nr_samples_total = ray_samples_packed.get_total_nr_samples();
    CHECK(grad_result.size(0) == nr_rays) << "grad_result should have size of nr_samples_total x 1. but it has " << grad_result.sizes();
    CHECK(grad_result.size(1) == 1) << "grad_result should have size of nr_samples_total x 1. but it has " << grad_result.sizes();

    // fill the samples
    const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};

    torch::Tensor grad_values = torch::zeros({nr_samples_total, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    torch::Tensor grad_weights = torch::zeros({nr_samples_total, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

    if (ray_samples_packed.is_empty())
    {
        return std::make_tuple(grad_values, grad_weights);
    }

    VolumeRenderingGPU::integrate_with_weights_1d_backward_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_rays,
        ray_samples_packed.get_max_nr_samples(), // useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
        ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        grad_result.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        values.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        weights.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        result.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // output
        grad_values.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        grad_weights.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
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

    return std::make_tuple(grad_values, grad_weights);
}

std::tuple<torch::Tensor, torch::Tensor> VolumeRendering::integrate_with_weights_3d_backward(const torch::Tensor &grad_result, const RaySamplesPacked &ray_samples_packed, const torch::Tensor &values, const torch::Tensor &weights, const torch::Tensor &result)
{
    if (verbose)
        std::cout << "integrate_with_weights_3d_backward" << std::endl;

    int nr_rays = ray_samples_packed.get_nr_rays();
    int nr_samples_total = ray_samples_packed.get_total_nr_samples();
    CHECK(grad_result.size(0) == nr_rays) << "grad_result should have size of nr_samples_total x 3. but it has " << grad_result.sizes();
    CHECK(grad_result.size(1) == 3) << "grad_result should have size of nr_samples_total x 3. but it has " << grad_result.sizes();

    // fill the samples
    const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};

    torch::Tensor grad_values = torch::zeros({nr_samples_total, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    torch::Tensor grad_weights = torch::zeros({nr_samples_total, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    
    if (ray_samples_packed.is_empty())
    {
        return std::make_tuple(grad_values, grad_weights);
    }

    VolumeRenderingGPU::integrate_with_weights_3d_backward_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_rays,
        ray_samples_packed.get_max_nr_samples(), // useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
        ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        grad_result.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        values.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        weights.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        result.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // output
        grad_values.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        grad_weights.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
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

    return std::make_tuple(grad_values, grad_weights);
}

torch::Tensor VolumeRendering::sum_over_rays_backward(const torch::Tensor &grad_values_sum_per_ray, const torch::Tensor &grad_values_sum_per_sample, const RaySamplesPacked &ray_samples_packed, const torch::Tensor &samples_values)
{
    if (verbose)
        std::cout << "sum_over_rays_backward" << std::endl;

    int nr_rays = ray_samples_packed.get_nr_rays();
    int nr_samples_total = ray_samples_packed.get_total_nr_samples();
    CHECK(grad_values_sum_per_ray.size(0) == nr_rays) << "grad_values_sum_per_ray should have size of nr_rays x 1. but it has " << grad_values_sum_per_ray.sizes();
    CHECK(grad_values_sum_per_sample.size(0) == nr_samples_total) << "grad_values_sum_per_sample should have size nr_sample x 1 " << grad_values_sum_per_sample.sizes();
    int val_dim = samples_values.size(1);

    // fill the samples
    const dim3 blocks = {(unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1};

    torch::Tensor grad_sample_values = torch::zeros({nr_samples_total, val_dim}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

    if (ray_samples_packed.is_empty())
    {
        return grad_sample_values;
    }

    if (val_dim == 1)
    {
        VolumeRenderingGPU::sum_over_rays_backward_gpu<1><<<blocks, BLOCK_SIZE>>>(
            nr_rays,
            ray_samples_packed.get_max_nr_samples(), // useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
            ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            grad_values_sum_per_ray.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            grad_values_sum_per_sample.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            samples_values.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            // output
            grad_sample_values.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
        );
    }
    else if (val_dim == 2)
    {
        VolumeRenderingGPU::sum_over_rays_backward_gpu<2><<<blocks, BLOCK_SIZE>>>(
            nr_rays,
            ray_samples_packed.get_max_nr_samples(), // useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
            ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            grad_values_sum_per_ray.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            grad_values_sum_per_sample.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            samples_values.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            // output
            grad_sample_values.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
        );
    }
    else if (val_dim == 3)
    {
        VolumeRenderingGPU::sum_over_rays_backward_gpu<3><<<blocks, BLOCK_SIZE>>>(
            nr_rays,
            ray_samples_packed.get_max_nr_samples(), // useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
            ray_samples_packed.ray_start_end_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            grad_values_sum_per_ray.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            grad_values_sum_per_sample.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            samples_values.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            // output
            grad_sample_values.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
        );
    }
    else
    {
        LOG(FATAL) << "Val dim not implemented yet";
    }

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

    return grad_sample_values;
}
