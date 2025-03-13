from rich import print
import torch
import os
import numpy as np
import tinycudann as tcnn
from volsurfs_py.utils.math import round_ste
from mvdatasets.utils.images import (
    normalize_uv_coord,
    non_normalize_uv_coord,
    non_normalized_uv_coords_to_interp_corners,
    pix_to_texel_center_uv_coord,
    uv_coords_to_pix,
    non_normalized_uv_coords_to_lerp_weights,
)


class NeuralTexture(torch.nn.Module):

    def __init__(
        self,
        res,
        nr_channels,
        val_range=(0.0, 1.0),
        anchor=False,
        lerp=False,
        quantize_output=False,
        squeeze_output=False,
        align_to_webgl=False,
    ):
        super(NeuralTexture, self).__init__()
        if isinstance(res, list):
            self.res = torch.tensor(res).long()
        elif isinstance(res, torch.Tensor):
            self.res = res
        else:
            print(
                "[bold red]ERROR[/bold red]: NeuralTexture res should be a list or a torch.Tensor."
            )
            exit(1)
        self.nr_channels = nr_channels
        self.anchor = anchor
        self.lerp = lerp
        self.quantize_output = quantize_output
        self.squeeze_output = squeeze_output
        self.val_range = val_range
        self.align_to_webgl = align_to_webgl

        if self.anchor and self.lerp:
            print(
                "[bold red]ERROR[/bold red]: NeuralTexture cannot anchor and lerp at the same time."
            )
            exit(1)

        encoding_config = {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 15,
            "base_resolution": 16,
            "per_level_scale": 1.5,
        }

        self.encoding = tcnn.Encoding(n_input_dims=2, encoding_config=encoding_config)

        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2,
        }

        self.network = tcnn.Network(
            n_input_dims=self.encoding.n_output_dims,
            n_output_dims=nr_channels,
            network_config=network_config,
        )

        self.model = torch.nn.Sequential(self.encoding, self.network)

    def forward(self, uv_coords, bake=False, iter_nr=None):

        if bake:

            # keep it as it is (assumption: always in texel centers)
            uv_coords_ = uv_coords

        elif self.anchor:
            # anchor uv_coords [0, 1] to pixel coordinates
            # uv coords are width, height
            # res is height, width
            # flip=True
            # results are normalized coordinates of texels centers
            uv_pix = uv_coords_to_pix(uv_coords, self.res, flip=True)

            if self.align_to_webgl:
                # rotate 90
                # i, j -> (width - 1) - j, i
                width = self.res[1].item()
                temp = uv_pix[:, 0].clone()  # i
                uv_pix[:, 0] = (width - 1) - uv_pix[:, 1]  # width - 1 - j
                uv_pix[:, 1] = temp

            uv_coords_center = pix_to_texel_center_uv_coord(uv_pix, self.res, flip=True)
            uv_coords_ = uv_coords_center

        elif self.lerp:
            # uv coords are width, height
            # res is height, width
            # flip=True
            # results are non normalized uv coordinates
            uv_coords_nn = non_normalize_uv_coord(uv_coords, self.res, flip=True)

            if self.align_to_webgl:
                # rotate 90
                # i, j -> (width - 1) - j, i
                width = self.res[1].item()
                temp = uv_coords_nn[:, 0].clone()  # i
                # uv_coords_nn[:, 0] = (width - 1) - uv_coords_nn[:, 1]  # width - 1 - j
                uv_coords_nn[:, 0] = (width) - uv_coords_nn[:, 1]  # width - 1 - j
                uv_coords_nn[:, 1] = temp

            # results are non normalized uv coordinates of the corners
            uv_corners_coords_nn = non_normalized_uv_coords_to_interp_corners(
                uv_coords_nn
            )  # [N, 4, 2]

            # find lerp weights
            lerp_weights = non_normalized_uv_coords_to_lerp_weights(
                uv_coords_nn, uv_corners_coords_nn
            )
            uv_corners_coords_nn = uv_corners_coords_nn.reshape(-1, 2)  # [N*4, 2]

            # normalize corners uv coords
            uv_corners_coords = normalize_uv_coord(
                uv_corners_coords_nn, self.res, flip=True
            )
            uv_coords_ = uv_corners_coords

        else:

            # error
            print(
                "[bold red]ERROR[/bold red]: NeuralTexture should be either anchor or lerp or bake."
            )
            exit(1)

        # uv_coords_ = uv_coords_.half()

        # clamp to [0, 1]
        # uv_coords_ = torch.clamp(uv_coords_, 0.0, 1.0)

        output = self.model(uv_coords_)  # float16

        # cast to float32
        output = output.float()

        # squeeze
        if self.squeeze_output:

            # squeeze output to [0, 1]
            output = torch.sigmoid(output)

            # quantize to 0-255
            if self.quantize_output:
                # and then quantize to 0-255
                output = output * 255.0  # map to [0, 255]
                output = round_ste(output)
                output = output / 255.0  # map to [0, 1]

        if bake:

            # return output as is
            return output

        # expansion and interpolation need to be performed
        # in float16, same precision used in WebGL

        # cast output to float16
        output = output.half()

        # unsqueeze
        if self.squeeze_output:
            # unsuqeeze output to [min, max]
            output = (
                self.val_range[0] + (self.val_range[1] - self.val_range[0]) * output
            )

        # lerp
        if self.lerp:
            output = output.reshape(-1, 4, self.nr_channels)
            output = (output * lerp_weights).sum(dim=1)

        # cast to float32
        output = output.float()

        return output

    @torch.no_grad()
    def render(
        self, res=None, batch_size=2**16, bake=False, preview=False, iter_nr=None
    ):

        if preview:
            # render in preview (low res) mode
            if res is not None:
                print(
                    "[bold yellow]WARNING[/bold yellow]: Rendering neural texture in preview mode, ignoring custom resolution."
                )
            res = torch.tensor([128, 128]).long()
        else:
            # check if should use custom target resolution
            if res is None:
                res = self.res
            else:
                res = torch.tensor(res).long()

        # create grid of uv coordinates
        u_pix = torch.arange(0, res[1].item(), dtype=torch.long)  # width
        v_pix = torch.arange(0, res[0].item(), dtype=torch.long)  # height
        uu, vv = torch.meshgrid([u_pix, v_pix], indexing="ij")
        uv_pix = torch.stack((uu.flatten(), vv.flatten())).t()  # pixel coordinates

        # uv coords are width, height
        # res is height, width
        # flip=True
        # coords are normalized
        uv_coords = pix_to_texel_center_uv_coord(uv_pix, res, flip=True)  # u, v

        # check if should batch the rendering
        if uv_coords.shape[0] > batch_size:
            output = torch.zeros((uv_coords.shape[0], self.nr_channels))
            for i in range(0, uv_coords.shape[0], batch_size):
                uv_coords_batch = uv_coords[i : i + batch_size]
                output[i : i + batch_size] = self.forward(uv_coords_batch, bake=bake)
        else:
            output = self.forward(uv_coords, bake=bake)

        img_torch = output.reshape(
            res[1].item(), res[0].item(), self.nr_channels
        )  # height, width, channels

        # print type
        # print(f"img_torch.dtype: {img_torch.dtype}")

        img_np = img_torch.detach().cpu().numpy()

        # permute(1, 0, 2) with numpy
        # img_np = np.transpose(img_np, (1, 0, 2))

        return img_np

    def save(self, save_path, override_name=None):
        """save checkpoint"""
        model_name = self.__class__.__name__.lower()
        if override_name is not None:
            model_name = override_name
        torch.save(
            self.state_dict(),
            os.path.join(save_path, f"{model_name}.pt"),
        )
        return save_path
