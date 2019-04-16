import torch
import math


def get_action(probs):
    # torch.Tensor will make a torch.float32 tensor, and multinomial will return a torch.int64 tensor
    action = torch.multinomial(probs, 1)
    return action

