import torch


def get_offsets_gt(nr_outer_surfs, nr_inner_surfs, delta_surfs, main_surf_shift=0.0):

    outer_offsets = []
    curr_offset = main_surf_shift
    for i in range(nr_outer_surfs):
        curr_offset -= delta_surfs
        outer_offsets.append(curr_offset)

    inner_offsets = []
    curr_offset = main_surf_shift
    for i in range(nr_inner_surfs):
        curr_offset += delta_surfs
        inner_offsets.append(curr_offset)

    offsets_gt = torch.tensor(inner_offsets[::-1] + outer_offsets) - main_surf_shift

    return offsets_gt
