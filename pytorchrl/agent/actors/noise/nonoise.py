import torch

class NoNoise:
    def __init__(self, num_outputs):
        pass
   
    def sample(self):
        return torch.zeros(1)
    
    def reset(self):
        pass
    