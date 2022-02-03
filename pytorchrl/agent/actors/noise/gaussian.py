import torch


class Normal:
    """ Gaussian Noise """
    def __init__(self, action_size, mean=0, std=0.1):
        self.mean = torch.FloatTensor([mean])
        self.std = torch.FloatTensor([std])
        # TODO: test if sampling per action dimension 
        # improves exploration of one sampled epsilon to 
        # each dimension 
        
    def sample(self):
        return torch.normal(mean=self.mean, std=self.std)
    
    def reset(self,):
        pass
