import torch
import inspect, re
from inspect import currentframe, getframeinfo


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r"\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", line)
        if m:
            return m.group(1)


def sanity_check(**tensors):
    for key, tensor in tensors.items():
        assert (
            torch.isnan(tensor).any() == False
        ), f"[SANITY CHECK FAILED] {key} has nan"
        assert (
            torch.isinf(tensor).any() == False
        ), f"[SANITY CHECK FAILED] {key} has inf"

    return True
