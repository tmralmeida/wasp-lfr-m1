import numpy as np
import torch
from torch.utils.data import DataLoader
from lib import *
from loss import cNceLoss
from dataset import cNCENoisyMnist
from tqdm import tqdm


args = get_arguments()

img_file = "data/train-images-idx3-ubyte"

# Get training data and preprocess data
x_train = parsing_file(img_file)
xs_train,_ = pre_process(x_train)

ds = cNCENoisyMnist(xs_train, 
                    args.k)

# Get precision matrices
guess_1 = np.diag(1 / (xs_train.var(axis=0) * 2))
prec_mat = torch.tensor(guess_1.copy())


train_loader = DataLoader(ds,
                          batch_size=args.batch_size, # 1000
                          num_workers=4,
                          shuffle=True,
                          pin_memory=True)

prec_mat_opt = prec_mat.requires_grad_() # first variable to optimize
optim = torch.optim.SGD([prec_mat_opt], lr = args.learning_rate) # optimizer (lr = 1e-2)

# Initialize loss function
criterion = cNceLoss(prec_mat_opt)

# Main loop
for iter_, (mnist_batch, noise_batch) in enumerate(train_loader):
    if iter_ > args.iterations: 
        break
    optim.zero_grad()
    
    # computing loss
    loss = criterion(mnist_batch,noise_batch)
    loss.backward()
    summary_res_cnce(iter_, loss, prec_mat_opt)
    # step optimizer
    optim.step()

print()
print("Output")
print("precision_matrix", prec_mat_opt.detach())



np.save("cNCE_results/precision_matrix.npy", prec_mat_opt.detach())
