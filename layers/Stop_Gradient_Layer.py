import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional
import numpy as np

class StopGradient(Function):
    @staticmethod
    def forward(ctx, x, coeff:Optional[float]=1.0):
        ctx.coeff = coeff
        output = x * 1.0
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.coeff, None
    
class StopGradientLayer(nn.Module):
    def __init__(self, warmup_steps: int):
        super(StopGradientLayer, self).__init__()
        self.warmup_steps = warmup_steps
        self.step = 0
        def custom_sigmoid(x, a, b, min_value, max_value):
            sigmoid_value = 1 / (1 + np.exp(-a * (x - b)))
            return min_value + sigmoid_value * (max_value - min_value)
        self.func = lambda x: custom_sigmoid(x, 10, 0.5, 1.0, 0.01)
        
    def forward(self, x):
        coeff = self.func(self.step / self.warmup_steps)
        self.step += 1
        return StopGradient.apply(x, coeff)