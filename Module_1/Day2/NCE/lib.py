import os
import numpy as np
import array
import struct
from scipy import sparse as sp
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import argparse
from argparse import ArgumentParser 
from scipy.stats import multivariate_normal


def get_transforms():
    """ Transforms for CiFAR-10
    """
    train_tfms = T.Compose([
        T.RandomCrop((28,28)),
        T.Grayscale(),
        T.ToTensor(),
        T.Lambda(lambda x: x.view(x.shape[1]**2)) # reshaping for our input shape
    ])
    return train_tfms

def get_stats(ds):
    """ Computes mean and std for the dataset
    """
    train_loader = DataLoader(ds,
                          batch_size=50000, # full set of images
                          num_workers=4,
                          shuffle=True,
                          pin_memory=True)
    for  batch in train_loader:
        break
    ds_np = np.array(batch[0], dtype = np.float64)
    mean = np.mean(ds_np, dtype=np.float64)
    var = ds_np.var(axis=0, dtype=np.float64)
    return mean, var

def get_mask(s):
    """
    Contructs a Laplacian matrix w/ 4-connected neighboors
    adaptation from: https://stackoverflow.com/questions/34895970/buildin-a-sparse-2d-laplacian-matrix-using-scipy-modules
    our was not working
    inputs:
        - side size (s)
        - out-of-diagonals scale (scalar)
    output: laplacian precision amtrix
    """
    nx, ny = s,s
    N  = nx*ny
    main_diag = np.ones(N)
    side_diag = np.ones(N-1) 
    side_diag[np.arange(1,N)%nx==0] = 0

    up_down_diag = np.ones(N-3)
    diagonals = [main_diag,side_diag,side_diag,up_down_diag,up_down_diag]
    laplacian = sp.diags(diagonals, [0, -1, 1,nx,-nx], format="csr")
    return torch.tensor(laplacian.toarray())

def cmpt_logpn(inp, mean, cov):
    """ Compute log pn for NCE
    inputs:
        - data (inp)
        - mean (mean)
        - covariance (cov)
    """
    return torch.tensor(multivariate_normal.logpdf(inp,mean = mean, cov = cov), dtype = torch.float64)

def cmpt_logpm(inp, lambda_, c_, mask):
    """ Compute log pn for NCE
    inputs:
        - data (inp)
        - precision matrix (lambda_)
        - normalizer (c_)
    """
    lambda_masked = lambda_ * mask
    inp = inp.unsqueeze(2).double()
    log_pm = -0.5 * inp.permute(0,2,1) @ lambda_masked.double() @ inp - c_.double()
    return log_pm.squeeze(-1).squeeze(-1)


def summary_res_nce(iter, loss, c_opt, prec_mat_opt):
    """ Write a brief summary w/ the results
    inputs:
        - iteration of the opt (iter)
        - current loss (loss)
        - c (c opt)
        - lambda (prec_mat_opt)
    """
    print(f"Batch {iter}: Loss: {loss.detach().numpy()}")    
    print(f"\tc: {c_opt.detach().numpy()[0]}")
    sign, neg_logdet = np.linalg.slogdet(prec_mat_opt.detach().numpy())
    print(f"\tTrue c: {0.5*28**2*np.log(2*np.pi) - 0.5*neg_logdet}")
    print()
    
    
def summary_res_cnce(iter, loss, prec_mat_opt):
    """ Write a brief summary w/ the results
    inputs:
        - iteration of the opt (iter)
        - current loss (loss)
        - lambda (prec_mat_opt)
    """
    print(f"{iter}: Loss: {loss.detach().numpy()}")    
    print(f"\tPrecision Matrix: {prec_mat_opt.grad.detach().numpy()}")
    print()
    
    
def get_arguments():
    """ Determines each command lines input and parses them
    """
    parser = ArgumentParser()
    
    # Precision Matrix Type
    parser.add_argument(
        "--lambda_type",
        "-lt",
        choices = ["diag_gauss", "laplace4"],
        default = "diag_gauss",
        help = ("The type of the precision matrix to optimize.")
    )
    
    # First Guess
    parser.add_argument(
        "--c_opt",
        "-c",
        type = float,
        default = -1000.0,
        help="Initial value of c (normalizer)"
    )
    
    

    # Hyperparameters
    parser.add_argument(
        "--batch_size",
        "-b",
        type = int,
        default = 100,
        help="Batch size value; Default: 100"
    )
    
    parser.add_argument(
        "--iterations",
        "-its",
        type = int,
        default = 50,
        help = ("Epochs number; Default: 50")

    )
    
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=1e-2,
        help="Learning rate value; Default: 1e-2"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of noisy images on each sample"
    )
    return parser.parse_args()