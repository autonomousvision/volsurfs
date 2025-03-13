from volsurfs_py.encodings.identity import IdentityEncoder
from volsurfs_py.encodings.frequency import FrequencyEncoder
from volsurfs_py.encodings.sphericalharmonics import SHEncoder
from volsurfs_py.encodings.permutohash import PermutoHashEncoder
from volsurfs_py.encodings.gridhash import GridHashEncoder


def get_encoder(encoding, **kwargs):
    if encoding == "none":
        encoder = IdentityEncoder(input_dim=kwargs["input_dim"])

    elif encoding == "frequency":
        encoder = FrequencyEncoder(
            input_dim=kwargs["input_dim"], multires=kwargs["multires"]
        )

    elif encoding == "spherical_harmonics":
        encoder = SHEncoder(input_dim=kwargs["input_dim"], degree=kwargs["degree"])

    elif encoding == "permutohash":
        if "bb_sides" not in kwargs:
            kwargs["bb_sides"] = None
        return PermutoHashEncoder(
            input_dim=kwargs["input_dim"],
            nr_levels=kwargs["nr_levels"],
            nr_iters_for_c2f=kwargs["nr_iters_for_c2f"],
            bb_sides=kwargs["bb_sides"],
        )

    elif encoding == "gridhash":
        if "bb_sides" not in kwargs:
            kwargs["bb_sides"] = None
        return GridHashEncoder(
            input_dim=kwargs["input_dim"],
            nr_levels=kwargs["nr_levels"],
            nr_iters_for_c2f=kwargs["nr_iters_for_c2f"],
            bb_sides=kwargs["bb_sides"],
        )

    else:
        raise NotImplementedError(
            "Unknown encoding mode, choose from [None, frequency, spherical_harmonics, permutohash, gridhash]"
        )

    return encoder
