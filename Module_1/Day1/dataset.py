import numpy as np
import torch 

class NoisyMnist(object):
    """ Noisy and original Mnist datasets object
    """
    def __init__(self, xs_train, mean_noisy, covar_noisy,niu = 1):
        self.niu = niu
        self.n_imgs = xs_train.shape[0]
        self.train_ds = xs_train
        self.cnt_ori = 0
        self.mean = torch.from_numpy(mean_noisy)
        self.covar = torch.from_numpy(covar_noisy)
        
    def __getitem__(self, idx):
        xt = self.train_ds[idx]
        noisy_dist =torch.distributions.multivariate_normal.MultivariateNormal(self.mean, self.covar)
        yt = noisy_dist.sample()
        return xt,yt
    
    def __len__(self):
        return self.n_imgs 
    

class cNCENoisyMnist(object):
    """ Noisy and original Mnist datasets object
    """
    def __init__(self, xs_train, k):
        self.train_ds = xs_train
        self.cov = covar = torch.from_numpy(np.cov(xs_train.reshape(28**2,xs_train.shape[0]))) #torch.from_numpy(np.diag(xs_train.var(axis=0) * 2)) 
        self.n_imgs = xs_train.shape[0]
        self.k = k
        
        
    def __getitem__(self, idx):
        xt = torch.from_numpy(self.train_ds[idx])
        sample = [xt]
        noisy_samples = []
        noisy_dist =torch.distributions.multivariate_normal.MultivariateNormal(xt, self.cov)
        for n in range(self.k):
            noisy_samples.append(noisy_dist.sample())
        sample.append(noisy_samples)
        return sample
    
    def __len__(self):
        return self.n_imgs 
    