import torch
import numpy as np

class NoisyDataset(object):
    def __init__(self, mean, covar):
        self.mean = torch.from_numpy(mean)
        self.covar = torch.from_numpy(covar)
        self.n_imgs = 60000

    def __getitem__(self, idx):
        return torch.distributions.multivariate_normal.MultivariateNormal(self.mean, self.covar).sample()
    
    def __len__(self):
        return self.n_imgs
            
            
            
        