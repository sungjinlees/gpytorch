import torch
import math
import pdb

def sample_mubs(size, num_probes):
    r1_coeff = torch.linspace(0, size - 1, size).unsqueeze(1)
    r2_coeff = ((r1_coeff + 1) * (r1_coeff + 2) / 2)
    r1 = torch.IntTensor(num_probes).random_(size).type_as(r1_coeff).unsqueeze(1).t()
    r2 = torch.IntTensor(num_probes).random_(size).type_as(r1_coeff).unsqueeze(1).t()

    two_pi_n = (2 * math.pi) / size
    re = torch.cos(two_pi_n * (r1_coeff.matmul(r1) + r2_coeff.matmul(r2))) / math.sqrt(size)
    im = torch.sin(two_pi_n * (r1_coeff.matmul(r1) + r2_coeff.matmul(r2))) / math.sqrt(size)

    return re, im

def mubs_trace(matmul_closure, size, num_probes):
    re, im = sample_mubs(size, num_probes)
    return size * ((re * matmul_closure(re)).sum() + (im * matmul_closure(im)).sum()) / num_probes