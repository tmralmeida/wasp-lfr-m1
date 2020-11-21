import numpy as np
import torch
from torch.utils.data import DataLoader
from lib import *
from loss import NceLoss
from dataset import NoisyMnist
from tqdm import tqdm


args = get_arguments()

img_file = "data/train-images-idx3-ubyte"

# Get training data and preprocess data
x_train = parsing_file(img_file)
xs_train,_ = pre_process(x_train)

# Get precision matrices
mean_noisy = np.zeros(28**2)
if args.lambda_type == "diag_gauss":
    guess_1 = np.diag(1 / (xs_train.var(axis=0) * 2))
    prec_mat = torch.tensor(guess_1.copy())
    covar_noisy = np.diag(xs_train.var(axis=0) * 2)

# Get dataset
mnist_train = NoisyMnist(xs_train,
                         mean_noisy,
                         covar_noisy)


train_loader = DataLoader(mnist_train,
                          batch_size=args.batch_size,
                          num_workers=4,
                          shuffle=True,
                          pin_memory=True)

# Initialize optimizer
prec_mat_opt = prec_mat.requires_grad_() # first variable to optimize
c_opt = torch.from_numpy(np.array([args.c_opt])).requires_grad_() # second variable to optimize
optim = torch.optim.SGD([prec_mat_opt, c_opt], lr = args.learning_rate) # optimizer

# Initialize dist. objects
p_theta = {
    "pmat":prec_mat_opt,
    "c":c_opt,
}

p_n = {
    "mean": mean_noisy,
    "cov": covar_noisy,
}

# Initialize loss function
criterion = NceLoss(p_theta,p_n)

# Main loop
for iter_, (mnist_batch, noise_batch) in enumerate(train_loader):
    if iter_ == args.iterations:
        break
    optim.zero_grad()
    
    # computing loss
    loss = criterion(mnist_batch,noise_batch)
    loss.backward()
    summary_res_nce(iter_, loss, c_opt, prec_mat_opt)
    # step optimizer
    optim.step()

print()
print("Output")
print("precision_matrix", prec_mat_opt.detach())
print("c", c_opt.detach())



np.save("precision_matrix2.npy", prec_mat_opt.detach())
np.save("normalizer2.npy", c_opt.detach())