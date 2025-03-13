import torch
import numpy as np
from torch.nn import functional as F

from volsurfs_py.volume_rendering.volume_rendering_funcs import (
    # VolumeRenderNerfFunc,
    CumprodOneMinusAlphaToTransmittanceFunc,
    IntegrateWithWeights1DFunc,
    IntegrateWithWeights3DFunc,
    SumOverRayFunc,
)


class CumprodOneMinusAlphaToTransmittanceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ray_samples_packed, alpha):
        transmittance, bg_transmittance = CumprodOneMinusAlphaToTransmittanceFunc.apply(
            ray_samples_packed, alpha
        )

        return transmittance, bg_transmittance


class IntegrateWithWeights1DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ray_samples_packed, value_samples, weights_samples):
        outputs = IntegrateWithWeights1DFunc.apply(
            ray_samples_packed, value_samples, weights_samples
        )

        return outputs


class IntegrateWithWeights3DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ray_samples_packed, value_samples, weights_samples):
        outputs = IntegrateWithWeights3DFunc.apply(
            ray_samples_packed, value_samples, weights_samples
        )

        return outputs


class SumOverRayModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ray_samples_packed, sample_values):
        values_sum_per_ray, values_sum_per_sample = SumOverRayFunc.apply(
            ray_samples_packed, sample_values
        )

        return values_sum_per_ray, values_sum_per_sample


class VolumeRendering(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.cumprod_one_minus_alpha_to_transmittance_module = (
            CumprodOneMinusAlphaToTransmittanceModule()
        )
        self.integrator_1d_module = IntegrateWithWeights1DModule()
        self.integrator_3d_module = IntegrateWithWeights3DModule()
        self.sum_ray_module = SumOverRayModule()

    def integrate_1d(self, ray_samples_packed, samples_vals, weights):
        assert samples_vals.shape[1] == 1, "samples_vals should be 1d"
        integrated = self.integrator_1d_module(
            ray_samples_packed, samples_vals, weights
        )
        return integrated

    def integrate_3d(self, ray_samples_packed, samples_vals, weights):
        assert samples_vals.shape[1] == 3, "samples_vals should be 3d"
        integrated = self.integrator_3d_module(
            ray_samples_packed, samples_vals, weights
        )
        return integrated


class VolumeRenderingNeRF(VolumeRendering):
    """volume rendering functions for NeRF"""

    def __init__(self):
        super().__init__()

    def compute_weights(self, ray_samples_packed, samples_densities):
        """compute NeRF weights for each sample along rays"""

        dt = ray_samples_packed.samples_dt
        alpha = torch.clamp(1.0 - torch.exp(-samples_densities * dt), min=0.0, max=1.0)
        transmittance, bg_transmittance = (
            self.cumprod_one_minus_alpha_to_transmittance_module(
                ray_samples_packed, 1 - alpha + 1e-6
            )
        )
        weights = alpha * transmittance

        return weights, bg_transmittance


class VolumeRenderingNeuS(VolumeRendering):
    """volume rendering functions for NeuS"""

    def __init__(self):
        super().__init__()

    def compute_alphas_from_logistic_beta(
        self,
        ray_samples_packed,
        sdf,
        gradients,
        cos_anneal_ratio,
        logistic_beta,
        debug_ray_idx=None,
    ):
        """compute NeuS alphas for each sample along rays"""
        dists = ray_samples_packed.samples_dt
        if debug_ray_idx is not None:
            debug_ray_samples_dt = ray_samples_packed.get_ray_samples_dt(debug_ray_idx)
            print("debug_ray_samples_dt", debug_ray_samples_dt)

        # dot product of gradients and ray directions
        true_cos = (ray_samples_packed.samples_dirs * gradients).sum(-1, keepdim=True)
        if debug_ray_idx is not None:
            debug_ray_start_end_idx = ray_samples_packed.get_ray_start_end_idx(
                debug_ray_idx
            )
            debug_ray_samples_true_cos = true_cos[
                debug_ray_start_end_idx[0] : debug_ray_start_end_idx[1]
            ]
            print("debug_ray_samples_true_cos", debug_ray_samples_true_cos)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations.
        # The anneal strategy below makes the cos value "not dead" at the beginning training iterations,
        # for better convergence.
        iter_cos = (
            F.relu(-true_cos * 0.5 + 0.5)
            * (
                1.0 - cos_anneal_ratio
            )  # maps [-1, 1] to [0, 1] and applies ReLU, [0, 1] to [0, 1]
            + F.relu(-true_cos)
            * cos_anneal_ratio  # maps [-1, 1] to [1, -1] and applies ReLU, [1, -1] to [0, 1]
        )  # alway positive
        iter_cos = -iter_cos  # always non-positive
        if debug_ray_idx is not None:
            debug_ray_start_end_idx = ray_samples_packed.get_ray_start_end_idx(
                debug_ray_idx
            )
            debug_ray_sample_iter_cos = iter_cos[
                debug_ray_start_end_idx[0] : debug_ray_start_end_idx[1]
            ]
            print("debug_ray_sample_iter_cos", debug_ray_sample_iter_cos)

        # https://github.com/Totoro97/NeuS/issues/35
        # useful for the womask setting
        # iter_cos = -(torch.abs(-true_cos))  # always non-positive

        # signed distances estimation at section points
        estimated_next_sdf = sdf + iter_cos * dists * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists * 0.5
        if debug_ray_idx is not None:
            debug_ray_start_end_idx = ray_samples_packed.get_ray_start_end_idx(
                debug_ray_idx
            )
            debug_ray_samples_estimated_next_sdf = estimated_next_sdf[
                debug_ray_start_end_idx[0] : debug_ray_start_end_idx[1]
            ]
            debug_ray_samples_estimated_prev_sdf = estimated_prev_sdf[
                debug_ray_start_end_idx[0] : debug_ray_start_end_idx[1]
            ]
            print(
                "debug_ray_samples_estimated_prev_sdf",
                debug_ray_samples_estimated_prev_sdf,
            )
            print(
                "debug_ray_samples_estimated_next_sdf",
                debug_ray_samples_estimated_next_sdf,
            )

        # sigmoid to [0, 1], steepness controlled by logistic_beta
        prev_cdf = torch.sigmoid(estimated_prev_sdf * logistic_beta)
        next_cdf = torch.sigmoid(estimated_next_sdf * logistic_beta)
        if debug_ray_idx is not None:
            print("logistic_beta", logistic_beta)
            debug_ray_start_end_idx = ray_samples_packed.get_ray_start_end_idx(
                debug_ray_idx
            )
            debug_ray_samples_prev_cdf = prev_cdf[
                debug_ray_start_end_idx[0] : debug_ray_start_end_idx[1]
            ]
            debug_ray_samples_next_cdf = next_cdf[
                debug_ray_start_end_idx[0] : debug_ray_start_end_idx[1]
            ]
            print("debug_ray_samples_prev_cdf", debug_ray_samples_prev_cdf)
            print("debug_ray_samples_next_cdf", debug_ray_samples_next_cdf)

        # alpha (discrete opacity values) computation
        alpha = ((prev_cdf - next_cdf + 1e-6) / (prev_cdf + 1e-6)).clip(0.0, 1.0)
        if debug_ray_idx is not None:
            debug_ray_start_end_idx = ray_samples_packed.get_ray_start_end_idx(
                debug_ray_idx
            )
            debug_ray_samples_alpha = alpha[
                debug_ray_start_end_idx[0] : debug_ray_start_end_idx[1]
            ]
            print("debug_ray_samples_alpha", debug_ray_samples_alpha)

        return alpha

    def compute_transmittance_from_alphas(self, ray_samples_packed, alpha):
        one_minus_alpha = 1 - alpha
        transmittance, _ = self.cumprod_one_minus_alpha_to_transmittance_module(
            ray_samples_packed, one_minus_alpha + 1e-6
        )

        return transmittance

    def compute_weights_from_transmittance_and_alphas(
        self, ray_samples_packed, transmittance, alpha
    ):
        """compute NeuS weights for each sample along rays"""

        weights = alpha * transmittance
        weights = weights.view(-1, 1)

        return weights
