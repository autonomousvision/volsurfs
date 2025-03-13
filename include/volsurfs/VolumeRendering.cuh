#pragma once

#include <stdarg.h>

#include "torch/torch.h"

#include <Eigen/Core>

#include "volsurfs/OccupancyGrid.cuh" //include RaySamplesPacked

#include "volsurfs/pcg32.h"

class VolumeRendering
{
public:
    VolumeRendering();
    ~VolumeRendering();

    static std::tuple<torch::Tensor, torch::Tensor> cumprod_one_minus_alpha_to_transmittance(
        const RaySamplesPacked &ray_samples_packed,
        const torch::Tensor &alpha_samples
    );
    static torch::Tensor integrate_with_weights_1d(
        const RaySamplesPacked &ray_samples_packed,
        const torch::Tensor &values,
        const torch::Tensor &weights
    );
    static torch::Tensor integrate_with_weights_3d(
        const RaySamplesPacked &ray_samples_packed,
        const torch::Tensor &values,
        const torch::Tensor &weights
    );
    static torch::Tensor sdf2alpha(
        const RaySamplesPacked &ray_samples_packed,
        const torch::Tensor &samples_sdf,
        const torch::Tensor &logistic_beta
    );
    static std::tuple<torch::Tensor, torch::Tensor> sum_over_rays(
        const RaySamplesPacked &ray_samples_packed,
        const torch::Tensor &sample_values
    );
    static torch::Tensor cumsum_over_rays(
        const RaySamplesPacked &ray_samples_packed,
        const torch::Tensor &sample_values,
        const bool inverse
    );
    static torch::Tensor compute_cdf(
        const RaySamplesPacked &ray_samples_packed,
        const torch::Tensor &samples_weights
    );
    static RaySamplesPacked importance_sample(
        const RaySamplesPacked &ray_samples_packed,
        const torch::Tensor &samples_cdf,
        const int nr_importance_samples,
        const bool jitter_samples
    );
    static RaySamplesPacked combine_ray_samples_packets(
        const RaySamplesPacked &ray_samples_packed,
        const RaySamplesPacked &ray_samples_imp,
        const float min_dist_between_samples
    );
    static torch::Tensor median_depth_over_rays(
        const RaySamplesPacked &ray_samples_packed,
        const torch::Tensor &samples_weights,
        const float threshold
    );

    // backward passes
    static torch::Tensor cumprod_one_minus_alpha_to_transmittance_backward(
        const torch::Tensor &grad_transmittance,
        const torch::Tensor &grad_bg_transmittance,
        const RaySamplesPacked &ray_samples_packed,
        const torch::Tensor &alpha,
        const torch::Tensor &transmittance,
        const torch::Tensor &bg_transmittance,
        const torch::Tensor &cumsumLV
    );
    static std::tuple<torch::Tensor, torch::Tensor> integrate_with_weights_1d_backward(
        const torch::Tensor &grad_result,
        const RaySamplesPacked &ray_samples_packed,
        const torch::Tensor &values,
        const torch::Tensor &weights,
        const torch::Tensor &result
    );
    static std::tuple<torch::Tensor, torch::Tensor> integrate_with_weights_3d_backward(
        const torch::Tensor &grad_result,
        const RaySamplesPacked &ray_samples_packed,
        const torch::Tensor &values,
        const torch::Tensor &weights,
        const torch::Tensor &result
    );
    static torch::Tensor sum_over_rays_backward(
        const torch::Tensor &grad_values_sum_per_ray,
        const torch::Tensor &grad_values_sum_per_sample,
        const RaySamplesPacked &ray_samples_packed,
        const torch::Tensor &sample_values
    );

    static pcg32 m_rng;
    static const bool verbose = false;

private:
};
