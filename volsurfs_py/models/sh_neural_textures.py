from rich import print
import torch
import os
from volsurfs_py.models.neural_texture import NeuralTexture
from volsurfs_py.encodings.sphericalharmonics import SHEncoder


class SHNeuralTextures(torch.nn.Module):

    def __init__(
        self,
        sh_deg=0,
        nr_channels=3,
        sh_range=[1.0, 5.0, 10.0, 20.0],
        anchor=False,
        lerp=False,
        deg_res=[2048, 1024, 512, 256],
        quantize_output=False,
        squeeze_output=False,
        align_to_webgl=False,
    ):
        super().__init__()

        if sh_deg >= 4:
            print(
                "[bold red]ERROR[/bold red]: SHNeuralTextures only supports SH degrees up to 3."
            )
            exit(1)

        if quantize_output and not squeeze_output:
            print(
                "[bold red]ERROR[/bold red]: quantize_output requires squeeze_output."
            )
            exit(1)

        self.sh_deg = sh_deg
        self.nr_channels = nr_channels

        self.deg_res = deg_res
        self.sh_range = sh_range
        self.deg_nr_coeffs = [1, 3, 5, 7]
        self.nr_coeffs = 0
        for deg in range(sh_deg + 1):
            self.nr_coeffs += self.deg_nr_coeffs[deg]

        neural_textures = []

        for deg in range(sh_deg + 1):

            nt = NeuralTexture(
                res=[self.deg_res[deg], self.deg_res[deg]],
                nr_channels=(nr_channels * self.deg_nr_coeffs[deg]),
                val_range=(-self.sh_range[deg], self.sh_range[deg]),
                anchor=anchor,
                lerp=lerp,
                quantize_output=quantize_output,
                squeeze_output=squeeze_output,
                align_to_webgl=align_to_webgl,
            )
            neural_textures.append(nt)

        self.neural_textures = torch.nn.ModuleList(neural_textures)

    def forward(self, uv_coords, view_dirs=None, iter_nr=None):

        nr_points = uv_coords.shape[0]

        # init out tensor
        output = torch.zeros(nr_points, self.nr_channels, self.nr_coeffs)
        # print(f"output.shape: {output.shape}")

        nr_written_coeffs = 0
        for deg in range(self.sh_deg + 1):
            res = self.neural_textures[deg](uv_coords).reshape(
                nr_points, self.nr_channels, -1
            )
            # print(f"res.shape: {res.shape}")
            # print(nr_written_coeffs, (nr_written_coeffs+self.deg_nr_coeffs[deg]))
            output[
                :, :, nr_written_coeffs : (nr_written_coeffs + self.deg_nr_coeffs[deg])
            ] = res
            nr_written_coeffs += self.deg_nr_coeffs[deg]

        if view_dirs is None:
            return output

        # cast to float16
        sh_coeffs = output.half()

        # evaluate view dependent channels
        raw_output = SHEncoder.eval(sh_coeffs, view_dirs, degree=self.sh_deg)
        output = torch.sigmoid(raw_output)

        # cast to float32
        output = output.float()

        return output

    @torch.no_grad()
    def render(self, batch_size=2**16, preview=False, bake=False, iter_nr=None):

        renders = []
        for deg in range(self.sh_deg + 1):
            render = self.neural_textures[deg].render(
                batch_size=batch_size,
                preview=preview,
                bake=bake,
            )
            render = render.reshape(
                render.shape[0], render.shape[1], self.nr_channels, -1
            )
            renders.append(render)

        return renders

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
