import torch
from volsurfs_py.encodings.encoder import Encoder


class IdentityEncoder(Encoder):
    def __init__(self, input_dim=3, **kwargs):
        super().__init__(input_dim, input_dim)

    def __call__(self, x, **kwargs):
        return x
