# import torch
# from torch.autograd import Function
# from torch.cuda.amp import custom_bwd, custom_fwd

# # From: https://github.com/ashawkey/torch-ngp/blob/main/activation.py#L18


# class TruncExp(Function):
#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float32)  # cast to float32
#     def forward(ctx, x):
#         ctx.save_for_backward(x)
#         y = torch.exp(x)
#         if torch.isinf(y).any():
#             print(x, y)
#         return y

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, g):
#         x = ctx.saved_tensors[0]
#         return g * torch.exp(x.clamp(-15, 15))


# # trunc_exp = _trunc_exp.apply

import torch
import torch.nn as nn


class TruncatedExp(nn.Module):
    def __init__(self, threshold=10.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        # Apply torch.exp to the input tensor
        exp_x = torch.exp(x)

        # Use torch.clamp to truncate the output values
        truncated_exp_x = torch.clamp(exp_x, max=self.threshold)

        return truncated_exp_x
