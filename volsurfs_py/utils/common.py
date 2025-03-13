import torch
import numpy as np
import torch


def nchw2nhwc(x):
    x = x.permute(0, 2, 3, 1)
    return x


def nhwc2nchw(x):
    x = x.permute(0, 3, 1, 2)
    return x


# make from N,C,H,W to N,Nrpixels,C
def nchw2nXc(x):
    nr_feat = x.shape[1]
    nr_batches = x.shape[0]
    x = x.permute(0, 2, 3, 1)  # from N,C,H,W to N,H,W,C
    x = x.reshape(nr_batches, -1, nr_feat)
    return x


# make from N,NrPixels,C to N,C,H,W
def nXc2nchw(x, h, w):
    nr_feat = x.shape[2]
    nr_batches = x.shape[0]
    x = x.reshape(nr_batches, h, w, nr_feat)
    x = x.permute(0, 3, 1, 2)  # from N,H,W,C, N,C,H,W
    return x


# make from N,C,H,W to nr_pixels, C
# ONLY works when N is 1
def nchw2lin(x):
    assert x.shape[0] == 1, "nchw2lin only works when N is 1"
    nr_feat = x.shape[1]  # C
    x = x.permute(0, 2, 3, 1).contiguous()  # from N,C,H,W to N,H,W,C
    x = x.reshape(-1, nr_feat)
    return x


# go from nr_pixels, C to H,W,C
def lin2hwc(x, h, w):
    nr_feat = x.shape[1]
    x = x.reshape(h, w, nr_feat)
    return x


# go from H,W,C to N,C,H,W
def hwc2nchw(x):
    # nr_feat = x.shape[-1]
    x = x.unsqueeze(0).permute(0, 3, 1, 2)  # nhwc to nchw
    return x


# go from nr_pixels,S,C to H,W,S,C
def lin2hwsc(x, h, w, s):
    nr_feat = x.shape[-1]
    x = x.reshape(h, w, s, nr_feat)
    return x


# go from nr_pixels, C to 1,C,H,W
def lin2nchw(x, h, w):
    nr_feat = x.shape[1]
    x = x.reshape(1, h, w, nr_feat)
    x = nhwc2nchw(x)
    return x


def img2tex(x):
    x = x.permute(0, 2, 3, 1).squeeze(0)  # nchw to hwc
    return x


def tex2img(x):
    x = x.unsqueeze(0).permute(0, 3, 1, 2)  # nhwc to  nchw
    return x


# https://github.com/NVlabs/instant-ngp/blob/c3f3534801704fe44585e6fbc2dc5f528e974b6e/scripts/common.py#L139
def srgb_to_linear(img):
    limit = 0.04045
    return torch.where(img > limit, torch.pow((img + 0.055) / 1.055, 2.4), img / 12.92)


def linear_to_srgb(img):
    limit = 0.0031308
    return torch.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)


def map_range_val(input_val, input_start, input_end, output_start, output_end):
    input_clamped = max(input_start, min(input_end, input_val))
    if input_start >= input_end:
        return output_end
    return output_start + ((output_end - output_start) / (input_end - input_start)) * (
        input_clamped - input_start
    )


def smoothstep_tensor(e0, e1, x):
    t = torch.clip(((x - e0) / (e1 - e0)), 0, 1)
    return t * t * (3.0 - 2.0 * t)


def smootherstep_tensor(e0, e1, x):
    t = torch.clip(((x - e0) / (e1 - e0)), 0, 1)
    return (t**3) * (t * (t * 6 - 15) + 10)


def smoothstep_val(e0, e1, x):
    t = np.clip(((x - e0) / (e1 - e0)), 0, 1)
    return t * t * (3.0 - 2.0 * t)


def smootherstep_val(e0, e1, x):
    t = np.clip(((x - e0) / (e1 - e0)), 0, 1)
    return (t**3) * (t * (t * 6 - 15) + 10)


# get a parameter t from 0 to 1 and maps it to range 0, 1 but with a very fast increase at the begining and it stops slowly towards 1. From Fast and Funky 1D Nonlinear Transformations
def smoothstop2(t):
    return 1 - pow((1 - t), 2)


def smoothstop3(t):
    return 1 - pow((1 - t), 3)


def smoothstop4(t):
    return 1 - pow((1 - t), 4)


def smoothstop5(t):
    return 1 - pow((1 - t), 5)


def smoothstop_n(t, n):
    return 1 - pow((1 - t), n)


# make it  a power of 2
def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


# Compute a power of two less than or equal to `n`
def previous_power_of_2(n):
    # set all bits after the last set bit
    n = n | (n >> 1)
    n = n | (n >> 2)
    n = n | (n >> 4)
    n = n | (n >> 8)
    n = n | (n >> 16)

    # drop all but the last set bit from `n`
    return n - (n >> 1)


def leaky_relu_init(m, negative_slope=0.2):
    gain = np.sqrt(2.0 / (1.0 + negative_slope**2))

    if isinstance(m, torch.nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // 2
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // 4
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // 8
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * np.sqrt(2.0 / (n1 + n2))
    else:
        return

    m.weight.data.uniform_(-std * np.sqrt(3.0), std * np.sqrt(3.0))
    if m.bias is not None:
        m.bias.data.zero_()

    if isinstance(m, torch.nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    # m.weights_initialized=True


def apply_weight_init_fn(m, fn, negative_slope=1.0):
    should_initialize_weight = True
    if not hasattr(
        m, "weights_initialized"
    ):  # if we don't have this then we need to intiialzie
        # fn(m, is_linear, scale)
        should_initialize_weight = True
    elif m.weights_initialized == False:  # if we have it but it's set to false
        # fn(m, is_linear, scale)
        should_initialize_weight = True
    else:
        print("skipping weight init on ", m)
        should_initialize_weight = False

    if should_initialize_weight:
        # fn(m, is_linear, scale)
        fn(m, negative_slope)
        # m.weights_initialized=True
        for module in m.children():
            apply_weight_init_fn(module, fn, negative_slope)


def linear2color_corr(img, dim: int = -1):
    """Applies ad-hoc 'color correction' to a linear RGB Mugsy image along
    color channel `dim` and returns the gamma-corrected result."""

    if dim == -1:
        dim = len(img.shape) - 1

    gamma = 2.0
    black = 3.0 / 255.0
    color_scale = [1.4, 1.1, 1.6]

    assert img.shape[dim] == 3
    if dim == -1:
        dim = len(img.shape) - 1
    scale = np.array(color_scale).reshape(
        [3 if i == dim else 1 for i in range(img.ndim)]
    )
    img = img * scale / 1.1
    return np.clip(
        (((1.0 / (1 - black)) * 0.95 * np.clip(img - black, 0, 2)) ** (1.0 / gamma))
        - 15.0 / 255.0,
        0,
        2,
    )
