import torch

from abc import ABC, abstractmethod


class Encoder(ABC, torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        ABC.__init__(self)
        torch.nn.Module.__init__(self)
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def __call__(self, x, **kwargs):
        pass

    def __str__(self):
        # return class name and list of attributes
        encoder_name = self.__class__.__name__
        attributes = []
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                attributes.append(f"{k}={v}")
        return f"{encoder_name}({', '.join(attributes)})"

    def reset(self):
        pass
