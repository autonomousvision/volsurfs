import torch
from torch.autograd import Function


class RoundSTE(Function):
    @staticmethod
    def forward(ctx, input):
        result = input.round()
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: treat rounding as identity in the backward pass
        return grad_output


# Create an instance of the custom function
round_ste = RoundSTE.apply

# def round_ste(x):
#     return x + x.round().detach() - x.detach()
