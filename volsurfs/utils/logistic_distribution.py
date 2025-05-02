import numpy as np
import torch


def get_logistic_beta_from_variance(variance):
    logistic_beta = np.exp(variance * 10.0)
    logistic_beta = np.clip(logistic_beta, 1e-6, 1e6)
    return logistic_beta


def logistic_distribution(x, beta=1.0):
    output_tensor = True
    if isinstance(x, float):
        x = torch.tensor(x)
        output_tensor = False
    exp_term = torch.clamp(torch.exp(-beta * x), -1e6, 1e6)
    if torch.isnan(exp_term).any():
        print(f"NaN values in exp_term with beta = {beta}")
        exit()
    res = beta * exp_term / (1 + exp_term) ** 2

    if not output_tensor:
        res = res.item()
    return res


def logistic_distribution_stdev(beta=1.0):
    s = 1.0 / beta
    return (s * np.pi) / np.sqrt(3)
