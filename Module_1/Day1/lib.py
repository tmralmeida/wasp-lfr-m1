import os
import numpy as np
import array
import struct
from scipy import sparse as sp
from scipy.stats import multivariate_normal
import torch
import argparse
from argparse import ArgumentParser 

DATA_TYPES = {0x08: 'B',  # unsigned byte
              0x09: 'b',  # signed byte
              0x0b: 'h',  # short (2 bytes)
              0x0c: 'i',  # int (4 bytes)
              0x0d: 'f',  # float (4 bytes)
              0x0e: 'd'}  # double (8 bytes)



def parsing_file(file):
    """
    Parsing initial ds file
    """
    with open(file, 'rb') as fd:
        header = fd.read(4)
        zeros, data_type, num_dimensions = struct.unpack('>HBB', header)
        data_type = DATA_TYPES[data_type]
        dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,fd.read(4 * num_dimensions))
        data = array.array(data_type, fd.read())
        data.byteswap()
        return np.array(data).reshape(dimension_sizes)
    
    
def pre_process(x_train):
    """
    Dataset preprocessing
    input: original dataset (x_train)
    output: processed dataset (xs_train)
    """
    
    xs_train = x_train[:]/np.max(x_train) # norm imgs
    mean_ = np.mean(xs_train)
    xs_train -= mean_
    noise = np.random.normal(0, 1/100, xs_train.shape) # add per-pixel gaussian noise 
    xs_train += noise
    return xs_train.reshape(xs_train.shape[0],-1), mean_

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
    inp = inp.unsqueeze(2)
    log_pm = -0.5 * inp.permute(0,2,1) @ lambda_masked @ inp - c_
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