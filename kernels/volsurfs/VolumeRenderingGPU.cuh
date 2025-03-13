#pragma once

#include <torch/torch.h>
#include "volsurfs/helper_math.h"

// matrices
#include "volsurfs/mat3.h"
#include "volsurfs/mat4.h"

#define BLOCK_SIZE 256

namespace VolumeRenderingGPU
{

    __device__ float map_range_val(const float input_val, const float input_start, const float input_end, const float output_start, const float output_end)
    {
        float input_clamped = max(input_start, min(input_end, input_val));
        if (input_start >= input_end)
            return output_end;
        return output_start + ((output_end - output_start) / (input_end - input_start)) * (input_clamped - input_start);
    }

    __device__ float clamp_min(float x, float a)
    {
        return max(a, x);
    }

    __global__ void
    cumprod_one_minus_alpha_to_transmittance_gpu(
        const int nr_rays,
        const int max_nr_samples,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> ray_start_end_idx,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> one_minus_alpha_samples,
        // output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> transmittance_samples,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> bg_transmittance)
    {

        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_rays)
            return;

        // get the indexes of the start and end of the uniform samples
        int idx_start = ray_start_end_idx[idx][0];
        int idx_end = ray_start_end_idx[idx][1];
        int nr_samples = idx_end - idx_start;

        if (nr_samples == 0)
            return;

        // if (idx_end > max_nr_samples)
        // {
        //     printf("idx_end %d is greater than max_nr_samples %d, this should not have happened\n", idx_end, max_nr_samples);
        //     return; 
        // }

        // one_minus_alpha_samples = [0.9, 0.5, 0.1]
        // transmittance_samples = [1.0, 0.9, 0.45]
        // bg_transmittance = [0.45]

        float T = 1.f;
        for (int i = 0; i < nr_samples; i++)
        {

            // cumprod
            transmittance_samples[idx_start + i][0] = T;
            if (i < nr_samples - 1)
            { 
                // don't compute the product with the last one, because we want the BG transmittance to correspond to the last T that we stored in transmittance samples
                float one_minus_alpha = one_minus_alpha_samples[idx_start + i][0];
                T *= one_minus_alpha;
            }
        }
        bg_transmittance[idx][0] = T;
    }

    __global__ void
    integrate_with_weights_1d_gpu(
        const int nr_rays,
        const int max_nr_samples,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> ray_start_end_idx,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> values,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> weights,
        // output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> outputs)
    {
        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_rays)
            return;

        // get the indexes of the start and end of the uniform samples
        int idx_start = ray_start_end_idx[idx][0];
        int idx_end = ray_start_end_idx[idx][1];
        int nr_samples = idx_end - idx_start;

        if (nr_samples == 0)
            return;

        // if (idx_end > max_nr_samples)
        // {
        //     printf("idx_end %d is greater than max_nr_samples %d, this should not have happened\n", idx_end, max_nr_samples);
        //     return; 
        // }

        float output = 0;

        for (int i = 0; i < nr_samples; i++)
        {

            // integrate
            float value = values[idx_start + i][0];
            float weight = weights[idx_start + i][0];

            output += weight * value;
        }

        // finish
        outputs[idx][0] = output;
    }

    __global__ void
    integrate_with_weights_3d_gpu(
        const int nr_rays,
        const int max_nr_samples,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> ray_start_end_idx,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> values,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> weights,
        // output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> outputs)
    {
        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_rays)
            return;

        // get the indexes of the start and end of the uniform samples
        int idx_start = ray_start_end_idx[idx][0];
        int idx_end = ray_start_end_idx[idx][1];
        int nr_samples = idx_end - idx_start;
        
        if (nr_samples == 0)
            return;

        // if (idx_end > max_nr_samples)
        // {
        //     printf("idx_end %d is greater than max_nr_samples %d, this should not have happened\n", idx_end, max_nr_samples);
        //     return; 
        // }

        float3 output = make_float3(0, 0, 0);

        for (int i = 0; i < nr_samples; i++)
        {

            // integrate
            float3 value;
            value.x = values[idx_start + i][0];
            value.y = values[idx_start + i][1];
            value.z = values[idx_start + i][2];
            float weight = weights[idx_start + i][0];

            output += weight * value;
        }

        // finish
        outputs[idx][0] = output.x;
        outputs[idx][1] = output.y;
        outputs[idx][2] = output.z;
    }

    __device__ float sigmoid(float x)
    {
        float res = 1.0 / (1.0 + exp(-x));
        return res;
    }

    __global__ void
    sdf2alpha_gpu(
        const int nr_rays,
        const int max_nr_samples,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> ray_start_end_idx,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_dt,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_sdf,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> logistic_beta,
        // output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> alpha_samples
    ) {

        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_rays)
            return;

         // get the indexes of the start and end of the uniform samples
        int idx_start = ray_start_end_idx[idx][0];
        int idx_end = ray_start_end_idx[idx][1];
        int nr_samples = idx_end - idx_start;

        if (nr_samples == 0)
            return;

        // get s_0 dt
        // get s_0 sdf
        // get s_1 sdf

        for (int i = 0; i < nr_samples - 1; i++)
        {
            float dt = samples_dt[idx_start + i][0];
            // printf("dt %f\n", dt);
            float prev_sdf = samples_sdf[idx_start + i][0];
            // printf("prev_sdf %f\n", prev_sdf);
            float next_sdf = samples_sdf[idx_start + i + 1][0];
            // printf("next_sdf %f\n", next_sdf);
            float mid_sdf = (prev_sdf + next_sdf) * 0.5;
            // printf("mid_sdf %f\n", mid_sdf);
            float cos_val = (next_sdf - prev_sdf) / (dt + 1e-6);  // (next_sdf - prev_sdf) / (dt + 1e-6)
            cos_val = clamp(cos_val, -1e3, 0.0);
            // printf("cos_val %f\n", cos_val);
            float prev_esti_sdf = mid_sdf - cos_val * dt * 0.5;  // mid_sdf - cos_val * dt * 0.5
            // printf("prev_esti_sdf %f\n", prev_esti_sdf);
            float next_esti_sdf = mid_sdf + cos_val * dt * 0.5;  // mid_sdf + cos_val * dt * 0.5
            // printf("next_esti_sdf %f\n", next_esti_sdf);
            float logistic_beta_value = logistic_beta[idx_start + i][0];
            // printf("logistic_beta_value %f\n", logistic_beta_value);
            float prev_cdf = sigmoid(prev_esti_sdf * logistic_beta_value);
            // printf("prev_cdf %f\n", prev_cdf);
            float next_cdf = sigmoid(next_esti_sdf * logistic_beta_value);
            // printf("next_cdf %f\n", next_cdf);
            float alpha = (prev_cdf - next_cdf + 1e-6) / (prev_cdf + 1e-6);
            // printf("alpha %f\n", alpha);
            alpha_samples[idx_start + i][0] = alpha;
        }
    }

    template <int val_dim>
    __global__ void
    sum_over_rays_gpu(
        const int nr_rays,
        const int max_nr_samples,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> ray_start_end_idx,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> sample_values,
        // output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> values_sum,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> values_sum_stored_per_sample)
    {

        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_rays)
            return;

        // get the indexes of the start and end of the uniform samples
        int idx_start = ray_start_end_idx[idx][0];
        int idx_end = ray_start_end_idx[idx][1];
        int nr_samples = idx_end - idx_start;

        if (nr_samples == 0)
            return;

        // if (idx_end > max_nr_samples)
        // {
        //     printf("idx_end %d is greater than max_nr_samples %d, this should not have happened\n", idx_end, max_nr_samples);
        //     return; 
        // }

        float val_sum[val_dim]{0};
        for (int i = 0; i < nr_samples; i++)
        {

            for (int v = 0; v < val_dim; v++)
            {
                float val = sample_values[idx_start + i][v];
                val_sum[v] += val;
            }
        }

        // finish, writ ethe sum for the whole ray
        for (int v = 0; v < val_dim; v++)
        {
            values_sum[idx][v] = val_sum[v];
        }

        // store also the sum for each sample
        for (int i = 0; i < nr_samples; i++)
        {
            for (int v = 0; v < val_dim; v++)
            {
                values_sum_stored_per_sample[idx_start + i][v] = val_sum[v];
            }
        }
    }

    __global__ void
    cumsum_over_rays_gpu(
        const int nr_rays,
        const int max_nr_samples,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> ray_start_end_idx,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> sample_values,
        const bool inverse,
        // output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> values_cumsum_stored_per_sample)
    {

        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_rays)
            return;

        // get the indexes of the start and end of the uniform samples
        int idx_start = ray_start_end_idx[idx][0];
        int idx_end = ray_start_end_idx[idx][1];
        int nr_samples = idx_end - idx_start;

        if (nr_samples == 0)
            return;

        // if (idx_end > max_nr_samples)
        // {
        //     printf("idx_end %d is greater than max_nr_samples %d, this should not have happened\n", idx_end, max_nr_samples);
        //     return; 
        // }

        float val_cumsum = 0;
        for (int i = 0; i < nr_samples; i++)
        {

            float val = 0;
            if (inverse)
            {
                val = sample_values[idx_end - 1 - i][0];
            }
            else
            {
                val = sample_values[idx_start + i][0];
            }
            val_cumsum += val;

            if (inverse)
            {
                values_cumsum_stored_per_sample[idx_end - 1 - i][0] = val_cumsum;
            }
            else
            {
                values_cumsum_stored_per_sample[idx_start + i][0] = val_cumsum;
            }
        }
    }


    __global__ void
    median_depth_over_rays_gpu(
        const int nr_rays,
        const int max_nr_samples,
        const float threshold,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> ray_start_end_idx,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_z,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_weights,
        // output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_depth)
    {

        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_rays)
            return;

        // get the indexes of the start and end of the uniform samples
        int idx_start = ray_start_end_idx[idx][0];
        int idx_end = ray_start_end_idx[idx][1];
        int nr_samples = idx_end - idx_start;

        if (nr_samples == 0)
            return;

        float weights_cumsum = 0.0f;
        bool terminated = false;
        for (int i = 0; i < nr_samples; i++)
        {
            weights_cumsum += samples_weights[idx_start + i][0];
            if (weights_cumsum >= threshold)
            {
                // we found the depth
                ray_depth[idx][0] = samples_z[idx_start + i][0];
                terminated = true;
                break;
            }
        }
        if (!terminated)
        {
            // if we didn't find a depth, we just take the last one
            ray_depth[idx][0] = samples_z[nr_samples - 1][0]; 
        }
    }


    __global__ void
    compute_cdf_gpu(
        const int nr_rays,
        const int max_nr_samples,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> ray_start_end_idx,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_weights,
        // output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_cdf)
    {

        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_rays)
            return;

        // get the indexes of the start and end of the uniform samples
        int idx_start = ray_start_end_idx[idx][0];
        int idx_end = ray_start_end_idx[idx][1];
        int nr_samples = idx_end - idx_start;

        if (nr_samples == 0)
            return;

        // if (idx_end > max_nr_samples)
        // {
        //     printf("idx_end %d is greater than max_nr_samples %d, this should not have happened\n", idx_end, max_nr_samples);
        //     return; 
        // }

        if (nr_samples < 2)
        {
            printf("at least 2 samples are needed to compute the cdf, but ray only have %d samples \n", nr_samples);
            return;
        }

        float cdf = 0;
        float weights_sum = 0;
        for (int i = 0; i < nr_samples; i++)
        {

            samples_cdf[idx_start + i][0] = cdf; // we start by writing 0 and we ignore the last weight and we don't sum it. NeuS doesn't compute the weight for the last sample
            float weight = samples_weights[idx_start + i][0];
            weights_sum += weight;
            cdf += weight;
        }

        // sanity check: weights_sum should be close to 1 (or 0)
        // if close to 1, last cdf value should be close to 1
        if (fabs(weights_sum - 1.0) < 1e-3)
        {
            float last_cdf = samples_cdf[idx_end - 1][0];
            if(fabs(last_cdf - 1.0) > 1e-3)
            {
                // printf("weights_sum is 1 but last cdf value is not 1.0, it is %f\n", last_cdf);
                samples_cdf[idx_end - 1][0] = 1.0;
            }
        }
    }

    // https://stackoverflow.com/a/21662870
    __device__ int midpoint(int a, int b)
    {
        return a + (b - a) / 2;
    }

    // return the index of the first value in the samples_cdf that has a value higher then val
    __device__ int binary_search(const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> &samples_cdf, float val, int imin, int imax)
    {
        int nr_iters = 0;
        while (imax >= imin)
        {
            int imid = midpoint(imin, imax);
            float cdf = samples_cdf[imid][0];
            if (cdf > val)
            {
                imax = imid;
            }
            else
            {
                imin = imid;
            }

            if ((imax - imin) == 1)
            {
                return imax;
            }

            nr_iters += 1;
        }

        return imax;
    }

    __global__ void
    importance_sample_gpu(
        const int nr_rays,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_o,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_d,
        // input_ray_samples_packed
        const int uniform_max_nr_samples,
        const int imp_max_nr_samples,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> uniform_ray_start_end_idx,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> uniform_samples_z,
        // cdf
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> samples_cdf,
        // importance samples
        const int nr_importance_samples,
        pcg32 rng,
        const bool jitter_samples,
        // output
        torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> imp_samples_idx,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> imp_samples_3d,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> imp_samples_dirs,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> imp_samples_z,
        torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> imp_ray_start_end_idx
    ) {

        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_rays)
            return;

        // get the indexes of the start and end sample
        int uniform_idx_start = uniform_ray_start_end_idx[idx][0];
        int uniform_idx_end = uniform_ray_start_end_idx[idx][1];
        int uniform_nr_samples = uniform_idx_end - uniform_idx_start;

        if (uniform_nr_samples == 0)
            return;

        // if (uniform_idx_end > uniform_max_nr_samples)
        // {
        //     printf("uniform_idx_end %d is greater than uniform_max_nr_samples %d, this should not have happened\n", uniform_idx_end, uniform_max_nr_samples);
        //     return; 
        // }

        if (nr_importance_samples == 0)
        {
            printf("nr_importance_samples is 0, this should not have happened\n");
            return;
        }

        // get the indices where we should write the importance samples
        int imp_idx_start = idx * nr_importance_samples;
        int imp_idx_end = imp_idx_start + nr_importance_samples;

        if (imp_idx_end > imp_max_nr_samples)
        {
            printf("imp_idx_end %d is greater than imp_max_nr_samples %d, this should not have happened\n", imp_idx_end, imp_max_nr_samples);
            return; 
        }

        // preload some other stuff
        float3 ray_o = make_float3(rays_o[idx][0], rays_o[idx][1], rays_o[idx][2]);
        float3 ray_d = make_float3(rays_d[idx][0], rays_d[idx][1], rays_d[idx][2]);

        for (int i = 0; i < nr_importance_samples; i++)
        {
            // get the distance between samples, in the [0,1] range
            // if we want for example to create 1 sample, then the uniform rand should be 0.5, so just in the middle of the [0,1] range
            // if we create 2 samples, the uniform rand for each would be 0.33 and 0.66
            // we need to get this distance in between them (in the case of 2 samples the dist would be 0.33)
            // we do it by imagining we are creating nr_importance_samples+2 values in the range[0,1]. The extremes would be exactly at 0 and 1 but the second sample is exactly the distance we want
            float dist_in_uniform_space = 1.0 / (nr_importance_samples + 1);
            float uniform_rand = dist_in_uniform_space + i * dist_in_uniform_space;
            if (jitter_samples)
            {
                rng.advance(idx); // since all the threads start with the same seed, we need to advance this thread so it gets other numbers different than the otehrs
                float rand = rng.next_float();
                // move [-0.5/nr_importance_samples, + 0.5/nr_importance_samples]
                float mov = dist_in_uniform_space / 2.0;
                uniform_rand += map_range_val(rand, 0.0, 1.0, -mov, +mov);
            }

            // don't make the uniform rand too close to 1.0 or 0.0 because due to numerical errors you might not find a cdf that is exactly 1
            uniform_rand = clamp(uniform_rand, 0.0 + 1e-6, 1.0 - 1e-6);

            // do importance sampling given the cdf, take inspiration from SIGGRAPH 2021: Global Illumination Based on Surfels the part about importance sampling of the brdf https://www.youtube.com/watch?v=h1ocYFrtsM4
            // binary search for the indices where the cdf crosses this value
            int imax = binary_search(samples_cdf, uniform_rand, uniform_idx_start, uniform_idx_end - 1);
            int imin = max(imax - 1, 0);

            if (imax >= uniform_max_nr_samples) {
                printf("accessing uniform out of bounds (imax)");
                return; 
            }

            if (imin >= uniform_max_nr_samples) {
                printf("accessing uniform out of bounds (imin)");
                return; 
            }

            float cdf_max = samples_cdf[imax][0];
            float cdf_min = samples_cdf[imin][0];
            if (cdf_min > uniform_rand || cdf_max < uniform_rand)
            {
                // compute the maximum cdf and minimum
                float minimum_cdf = 99999;
                float maximum_cdf = -99999;
                for (int d = 0; d < uniform_nr_samples; d++)
                {   
                    if (uniform_idx_start + d >= uniform_max_nr_samples)
                    {
                        printf("reading cdf out of bounds");
                        return; 
                    }

                    float cdf = samples_cdf[uniform_idx_start + d][0];
                    if (cdf < minimum_cdf)
                        minimum_cdf = cdf;
                    if (cdf > maximum_cdf)
                        maximum_cdf = cdf;
                }
            }

            // get eh z values of the imin and imax
            float z_max = uniform_samples_z[imax][0];
            float z_min = uniform_samples_z[imin][0];
            float z_imp = map_range_val(uniform_rand, cdf_min, cdf_max, z_min, z_max);

            // // since for the uniform samples we assume the same, dt we want the importance samples to be not further than ray_min_dt from the samples
            // // get the closest between z_max and z_min and make z_imp to be at most ray_min_dt away from them
            // float dist_to_zmin = z_imp - z_min; // some positive value
            // float dist_to_zmax = z_max - z_imp;
            // if (dist_to_zmin < dist_to_zmax)
            // { 
            //     // we are closest to the previous sample
            //     dist_to_zmin = min(dist_to_zmin, ray_min_dt);
            //     z_imp = z_min + dist_to_zmin;
            // }
            // else
            // { 
            //     // we are closest to the next sample so we snap to it
            //     dist_to_zmax = min(dist_to_zmax, ray_min_dt);
            //     z_imp = z_max - dist_to_zmax;
            // }

            // create the new importance sample
            float3 imp_sample_pos = ray_o + z_imp * ray_d;

            if (imp_idx_start + i >= imp_max_nr_samples)
            {
                printf("writing importance out of bounds");
                return; 
            }

            // write the new sample
            // pos
            imp_samples_3d[imp_idx_start + i][0] = imp_sample_pos.x;
            imp_samples_3d[imp_idx_start + i][1] = imp_sample_pos.y;
            imp_samples_3d[imp_idx_start + i][2] = imp_sample_pos.z;
            // dir
            imp_samples_dirs[imp_idx_start + i][0] = ray_d.x;
            imp_samples_dirs[imp_idx_start + i][1] = ray_d.y;
            imp_samples_dirs[imp_idx_start + i][2] = ray_d.z;
            // z
            imp_samples_z[imp_idx_start + i][0] = z_imp;
        }

        // ray_start_end_idx
        imp_ray_start_end_idx[idx][0] = imp_idx_start;
        imp_ray_start_end_idx[idx][1] = imp_idx_end;
    }

    __global__ void
    combine_ray_samples_packets_gpu(
        const int nr_rays,
        const float min_dist_between_samples,
        const int values_dim,
        // samples_packed_1
        const int pkt_1_max_nr_samples,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> pkt_1_ray_start_end_idx,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> pkt_1_samples_idx,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pkt_1_samples_3d,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pkt_1_samples_dirs,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pkt_1_samples_z,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pkt_1_samples_values,
        // samples_packed_2
        const int pkt_2_max_nr_samples,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> pkt_2_ray_start_end_idx,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> pkt_2_samples_idx,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pkt_2_samples_3d,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pkt_2_samples_dirs,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pkt_2_samples_z,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pkt_2_samples_values,
        // combined stuff just for sanity checking
        const int combined_max_nr_samples,
        // where to start writing the combined samples for each ray
        const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> combined_indices_start,
        // output
        torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> combined_samples_idx,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> combined_samples_3d,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> combined_samples_dirs,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> combined_samples_z,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> combined_samples_values,
        torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> combined_ray_start_end_idx
    ) {

        // ray id
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_rays)
            return;

        // get the indexes of the start and end of the uniform samples
        int pkt_1_idx_start = pkt_1_ray_start_end_idx[idx][0];
        int pkt_1_idx_end = pkt_1_ray_start_end_idx[idx][1];
        int pkt_1_nr_samples = pkt_1_idx_end - pkt_1_idx_start;

        // get the indexes of the start and end of the importance samples
        int pkt_2_idx_start = pkt_2_ray_start_end_idx[idx][0];
        int pkt_2_idx_end = pkt_2_ray_start_end_idx[idx][1];
        int pkt_2_nr_samples = pkt_2_idx_end - pkt_2_idx_start;

        // no samples, just return
        if (pkt_1_nr_samples == 0 && pkt_2_nr_samples == 0)
            return;

        // at least one package has samples for ray idx

        int combined_idx_start = combined_indices_start[idx];
        int max_nr_samples_to_combine = pkt_1_nr_samples + pkt_2_nr_samples;

        // start writing all the samples
        int sample_idx_pkt_1;
        int sample_idx_pkt_2;
        int sample_idx_combined;
        int pkt_1_curr_sample = 0;
        int pkt_2_curr_sample = 0;
        bool finished_reading_all_pkt_1 = false;
        bool finished_reading_all_pkt_2 = false;
        float prec_z = 0.0;
        int nr_written_samples = 0;
        for (int i = 0; i <= max_nr_samples_to_combine - 1; i++)
        {
            if (finished_reading_all_pkt_1 && finished_reading_all_pkt_2)
            {
                break;
            }

            // pkt_x_curr_sample in [0, pkt_x_nr_samples - 1]
            sample_idx_pkt_1 = pkt_1_idx_start + pkt_1_curr_sample;
            sample_idx_pkt_2 = pkt_2_idx_start + pkt_2_curr_sample;
            sample_idx_combined = combined_idx_start + nr_written_samples;

            // check validity of samples indices
            if (sample_idx_pkt_1 >= pkt_1_max_nr_samples)
            {
                printf("pkt_1_idx_start %d, pkt_1_idx_end %d, pkt_1_nr_samples %d, pkt_1_curr_sample %d, sample_idx_pkt_1 % d\n", pkt_1_idx_start, pkt_1_idx_end,  pkt_1_nr_samples, pkt_1_curr_sample, sample_idx_pkt_1);
                return;
            }

            if (sample_idx_pkt_2 >= pkt_2_max_nr_samples)
            {
                printf("pkt_2_idx_start %d, pkt_2_idx_end %d, pkt_2_nr_samples %d, pkt_2_curr_sample %d, sample_idx_pkt_2 % d\n", pkt_2_idx_start, pkt_2_idx_end, pkt_2_nr_samples, pkt_2_curr_sample, sample_idx_pkt_2);
                return;
            }

            if (sample_idx_combined >= combined_max_nr_samples)
            {
                printf("combined_idx_start %d, sample_idx_combined %d, nr_written_samples %d\n", combined_idx_start, sample_idx_combined, nr_written_samples);
                return;
            }

            // we need to sort them in increasing order to z
            // write also dt which is just the capped distance towards the next one
            float z_pkt_1 = 0;
            if (!finished_reading_all_pkt_1)
            {
                z_pkt_1 = pkt_1_samples_z[sample_idx_pkt_1][0];
            }
            else
            {
                z_pkt_1 = 1e10; // gigantic value just to make sure we never choose thise one again
            }
            float z_pkt_2 = 0;
            if (!finished_reading_all_pkt_2)
            {
                z_pkt_2 = pkt_2_samples_z[sample_idx_pkt_2][0];
            }
            else
            {
                z_pkt_2 = 1e10; // gigantic value just to make sure we never choose the imp again
            }

            bool adding_pkt_1_sample = z_pkt_1 < z_pkt_2;
            if (adding_pkt_1_sample)
            {
                if (z_pkt_1 - prec_z < min_dist_between_samples)
                {
                    // skip sample as it is too close to the previous one
                    // printf("skipping sample as it is too close to the previous one\n");
                }
                else {
                    // store sample idx
                    combined_samples_idx[sample_idx_combined][0] = pkt_1_samples_idx[sample_idx_pkt_1][0];
                    // store positions
                    combined_samples_3d[sample_idx_combined][0] = pkt_1_samples_3d[sample_idx_pkt_1][0];
                    combined_samples_3d[sample_idx_combined][1] = pkt_1_samples_3d[sample_idx_pkt_1][1];
                    combined_samples_3d[sample_idx_combined][2] = pkt_1_samples_3d[sample_idx_pkt_1][2];
                    // store dirs
                    combined_samples_dirs[sample_idx_combined][0] = pkt_1_samples_dirs[sample_idx_pkt_1][0];
                    combined_samples_dirs[sample_idx_combined][1] = pkt_1_samples_dirs[sample_idx_pkt_1][1];
                    combined_samples_dirs[sample_idx_combined][2] = pkt_1_samples_dirs[sample_idx_pkt_1][2];
                    // store z
                    combined_samples_z[sample_idx_combined][0] = z_pkt_1;
                    prec_z = z_pkt_1;
                    // store values
                    for (int v = 0; v < values_dim; v++)
                    {
                        combined_samples_values[sample_idx_combined][v] = pkt_1_samples_values[sample_idx_pkt_1][v];
                    }
                
                    nr_written_samples += 1;
                }
                
                // if (pkt_1_curr_sample + 1 == pkt_1_nr_samples - 1)
                if (pkt_1_curr_sample + 1 >= pkt_1_nr_samples)
                {
                    finished_reading_all_pkt_1 = true;
                }
                else {
                    pkt_1_curr_sample += 1;
                }
            }
            else
            {
                if (z_pkt_2 - prec_z < min_dist_between_samples)
                {
                    // skip sample as it is too close to the previous one
                    // printf("skipping sample as it is too close to the previous one\n");
                }
                else {
                    // store sample idx
                    combined_samples_idx[sample_idx_combined][0] = pkt_2_samples_idx[sample_idx_pkt_2][0];
                    // store positions
                    combined_samples_3d[sample_idx_combined][0] = pkt_2_samples_3d[sample_idx_pkt_2][0];
                    combined_samples_3d[sample_idx_combined][1] = pkt_2_samples_3d[sample_idx_pkt_2][1];
                    combined_samples_3d[sample_idx_combined][2] = pkt_2_samples_3d[sample_idx_pkt_2][2];
                    // store dirs
                    combined_samples_dirs[sample_idx_combined][0] = pkt_2_samples_dirs[sample_idx_pkt_2][0];
                    combined_samples_dirs[sample_idx_combined][1] = pkt_2_samples_dirs[sample_idx_pkt_2][1];
                    combined_samples_dirs[sample_idx_combined][2] = pkt_2_samples_dirs[sample_idx_pkt_2][2];
                    // store z
                    combined_samples_z[sample_idx_combined][0] = z_pkt_2;
                    prec_z = z_pkt_2;
                    // store values
                    for (int v = 0; v < values_dim; v++)
                    {
                        combined_samples_values[sample_idx_combined][v] = pkt_2_samples_values[sample_idx_pkt_2][v];
                    }

                    nr_written_samples += 1;
                }
                
                // if (pkt_2_curr_sample + 1 == pkt_2_nr_samples - 1)
                if (pkt_2_curr_sample + 1 >= pkt_2_nr_samples)
                {
                    finished_reading_all_pkt_2 = true;
                }
                else {
                    pkt_2_curr_sample += 1;
                }
            }
        }

        int combined_idx_end = combined_idx_start + nr_written_samples;

        if (combined_idx_end > combined_max_nr_samples)
        {
            printf("combined_idx_end %d is greater than combined_max_nr_samples %d, this should not have happened\n", combined_idx_end, combined_max_nr_samples);
            return;
        }

        // ray_start_end_idx
        combined_ray_start_end_idx[idx][0] = combined_idx_start;
        combined_ray_start_end_idx[idx][1] = combined_idx_end;
    }

    __global__ void
    cumprod_one_minus_alpha_to_transmittance_backward_gpu(
        const int nr_rays,
        const int max_nr_samples,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> ray_start_end_idx,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_transmittance,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_bg_transmittance,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> one_minus_alpha,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> transmittance,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> bg_transmittance,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> cumsumLV,
        // output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_one_minus_alpha_samples)
    {

        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_rays)
            return;

        // get the indexes of the start and end of the uniform samples
        int idx_start = ray_start_end_idx[idx][0];
        int idx_end = ray_start_end_idx[idx][1];
        int nr_samples = idx_end - idx_start;

        if (nr_samples == 0)
            return;

        float bg_transmittance_cur = bg_transmittance[idx][0];           // is the multiplication of all the alphas until now
        float grad_bg_transmittance_cur = grad_bg_transmittance[idx][0]; // is the multiplication of all the alphas until now

        for (int i = 0; i < nr_samples; i++)
        {
            float grad_one_minus_alpha = 0;
            float one_minus_alpha_curr = one_minus_alpha[idx_start + i][0];
            if (i < nr_samples - 1)
            { 
                // this skips the last sample which doesn't participate in the cumprod and therefore gets no gradient from this
                float cumsum_next = cumsumLV[idx_start + i + 1][0];
                grad_one_minus_alpha = cumsum_next / clamp_min(one_minus_alpha_curr, 1e-6);
                grad_one_minus_alpha += grad_bg_transmittance_cur * bg_transmittance_cur / clamp_min(one_minus_alpha_curr, 1e-6);
            }
            // the last sample gets a grad_one_minus_alpha of zero but this is fine because according to sdf2alpha the last sample has either way an alpha of zero
            grad_one_minus_alpha_samples[idx_start + i][0] = grad_one_minus_alpha;
        }
    }

    __global__ void
    integrate_with_weights_1d_backward_gpu(
        const int nr_rays,
        const int max_nr_samples,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> ray_start_end_idx,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_result,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> values,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> weights,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> result,
        // output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_values,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_weights)
    {

        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_rays)
            return;

        // get the indexes of the start and end of the uniform samples
        int idx_start = ray_start_end_idx[idx][0];
        int idx_end = ray_start_end_idx[idx][1];
        int nr_samples = idx_end - idx_start;

        if (nr_samples == 0)
            return;

        float grad_result_cur_ray = grad_result[idx][0];

        for (int i = 0; i < nr_samples; i++)
        {

            float value_cur_sample = values[idx_start + i][0];
            float weight_cur_sample = weights[idx_start + i][0];

            grad_values[idx_start + i][0] = grad_result_cur_ray * weight_cur_sample;
            grad_weights[idx_start + i][0] = grad_result_cur_ray * value_cur_sample;
        }
    }

    __global__ void
    integrate_with_weights_3d_backward_gpu(
        const int nr_rays,
        const int max_nr_samples,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> ray_start_end_idx,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_result,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> values,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> weights,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> result,
        // output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_values,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_weights)
    {

        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_rays)
            return;

        // get the indexes of the start and end of the uniform samples
        int idx_start = ray_start_end_idx[idx][0];
        int idx_end = ray_start_end_idx[idx][1];
        int nr_samples = idx_end - idx_start;

        if (nr_samples == 0)
            return;

        float3 grad_result_cur_ray = make_float3(grad_result[idx][0], grad_result[idx][1], grad_result[idx][2]);

        for (int i = 0; i < nr_samples; i++)
        {

            float3 value_cur_sample = make_float3(values[idx_start + i][0], values[idx_start + i][1], values[idx_start + i][1]);
            float weight_cur_sample = weights[idx_start + i][0];

            grad_values[idx_start + i][0] = grad_result_cur_ray.x * weight_cur_sample;
            grad_values[idx_start + i][1] = grad_result_cur_ray.y * weight_cur_sample;
            grad_values[idx_start + i][2] = grad_result_cur_ray.z * weight_cur_sample;

            grad_weights[idx_start + i][0] =
                grad_result_cur_ray.x * value_cur_sample.x +
                grad_result_cur_ray.y * value_cur_sample.y +
                grad_result_cur_ray.z * value_cur_sample.z;
        }
    }

    template <int val_dim>
    __global__ void
    sum_over_rays_backward_gpu(
        const int nr_rays,
        const int max_nr_samples,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> ray_start_end_idx,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_values_sum_per_ray,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_values_sum_per_sample,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> sample_values,
        // output
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_sample_values)
    {

        // each thread will deal with a new value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // don't go out of bounds
        if (idx >= nr_rays)
            return;

        // get the indexes of the start and end of the uniform samples
        int idx_start = ray_start_end_idx[idx][0];
        int idx_end = ray_start_end_idx[idx][1];
        int nr_samples = idx_end - idx_start;

        if (nr_samples == 0)
            return;

        // load the grad for the whole ray
        float grad_sum_per_ray[val_dim]{0};
        for (int v = 0; v < val_dim; v++)
        {
            grad_sum_per_ray[v] = grad_values_sum_per_ray[idx][v];
        }

        for (int i = 0; i < nr_samples; i++)
        {

            for (int v = 0; v < val_dim; v++)
            {
                float grad_per_sample = grad_values_sum_per_sample[idx_start + i][v];
                grad_sample_values[idx_start + i][v] = grad_sum_per_ray[v] + grad_per_sample;
            }
        }
    }

} // namespace VolumeRenderingGPU
