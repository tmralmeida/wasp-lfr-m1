import numpy as np
import torch
from torch.utils.data import DataLoader
from lib import *
from loss import NceLoss
from noisy_dataset import NoisyDataset
from tqdm import tqdm
import torchvision


args = get_arguments()

train_tfms = get_transforms()

# Get original dataset
ds = torchvision.datasets.CIFAR10(root = "../data/", 
                                  train = True, 
                                  transform  = train_tfms, 
                                  target_transform = None, 
                                  download = False)

# Get statistics from dataset
mean, var = get_stats(ds)

# Get precision matrices
mean_noisy = np.zeros(28**2, dtype=np.float64) 
# print(mean_noisy)
if args.lambda_type == "diag_gauss":
    guess_1 = np.diag(1 / (var))
    prec_mat = torch.tensor(guess_1.copy())
    covar_noisy = np.diag(var)

# Get noisy dataset
noisy_ds = NoisyDataset(mean_noisy,
                        covar_noisy)

# Loaders
ori_loader = DataLoader(ds,
                        batch_size=args.batch_size,
                        num_workers=4,
                        shuffle=True,
                        pin_memory=True)

noi_loader = DataLoader(noisy_ds, 
                        batch_size=args.batch_size)



# Initialize optimizer
_, neg_logdet = np.linalg.slogdet(prec_mat.numpy())
c = 0.5*28**2*np.log(2*np.pi) - 0.5 * neg_logdet

prec_mat_opt = prec_mat.requires_grad_() # first variable to optimize
c_opt = torch.from_numpy(np.array([c])).requires_grad_() # second variable to optimize
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
for iter_, (ori_batch, noise_batch) in enumerate(zip(ori_loader, noi_loader)):
    if iter_ > args.iterations:
        break
    optim.zero_grad()
    
    # computing loss
    loss = criterion(ori_batch[0],noise_batch)
    loss.backward()
    summary_res_nce(iter_, loss, c_opt, prec_mat_opt)
    # step optimizer
    optim.step()

print()
print("Output")
print("precision_matrix", prec_mat_opt.detach())
print("c", c_opt.detach())



np.save("NCE_results/precision_matrix_mask.npy", prec_mat_opt.detach())
np.save("NCE_results/normalizer_mask.npy", c_opt.detach())