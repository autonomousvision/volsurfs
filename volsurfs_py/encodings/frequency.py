import torch
from volsurfs_py.encodings.encoder import Encoder

#### Positional encoding

# Another interesting tweak used in NeRF is "positional encoding", which postulates the use of a mapping to higher dimensional
# space (using a basis set of high-frequency functions). This greatly enhances the model's capability to capture high-frequency variations.


class FrequencyEncoder(Encoder):
    def __init__(self, input_dim=3, multires=6, include_input=True, **kwargs):
        periodic_fns = [torch.sin, torch.cos]
        output_dim = input_dim * multires * len(periodic_fns) + (
            input_dim if include_input else 0
        )
        super().__init__(input_dim, output_dim)

        # number of functions used in the positional encoding
        self.multires = multires
        self.include_input = include_input
        self.periodic_fns = periodic_fns

    def __call__(self, x, **kwargs):
        """TODO.

        Parameters
        ----------
        x : Tensor
            TODO
        include_input : bool, optional
            TODO, by default true
        multires : int, optional
            TODO, by default 6
        periodic_fns : list of functions, optional
            TODO, by default [torch.sin, torch.cos]

        Returns
        -------
        Tensor
            TODO
        """
        embed_fns = []

        if self.include_input:
            embed_fns.append(x)

        for l in range(self.multires):
            freq = 2.0**l
            for p_fn in self.periodic_fns:
                embed_fns.append(p_fn(x * freq))

        return torch.cat(embed_fns, -1)
