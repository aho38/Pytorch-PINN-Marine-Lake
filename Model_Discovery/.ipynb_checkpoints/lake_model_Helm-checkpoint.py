import torch
import torch.nn as nn
import numpy as np
import time
import pandas as pd
import os
from copy import deepcopy
from pyDOE import lhs # Latin Hypercube Sampling

# Check if CUDA is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

file_path = os.getcwd()
a, b = 0.0, 1.0

# Load True Solution and paramters
df = pd.read_csv(f'/g/g20/ho32/PINNvsFEM/Pytorch-PINN-Marine-Lake/Model_Discovery/synthetic_data/synthetic_data.csv')
x_array, mtrue_array, utrue_array = np.array(df['x']), np.array(df['mtrue']), np.array(df['utrue'])
forcing_array, dist_array = np.array(df['forcing']), np.array(df['distribution'])
omega = np.array(df['omega'])[0]

from scipy.interpolate import interp1d
true_solution = interp1d(x_array, utrue_array)
true_dist = interp1d(x_array, dist_array)
true_forcing = interp1d(x_array, forcing_array)
omega = torch.tensor(omega, dtype=torch.float32, device=device)



def main(Nu,Nf,layers):
    a, b = 0.0, 1.0

    # define training data and add noise
    X_u_train = torch.linspace(a,b, Nu, dtype=torch.float32)[:, None]
    u_train = torch.tensor(true_solution(X_u_train), dtype=torch.float32)
    noise_level = 0.01
    np.random.seed(0)
    u_train += torch.randn(u_train.shape) * abs(u_train).max() * noise_level


    X_f_train = torch.tensor(lhs(1, Nf), dtype=torch.float32)
    dist_f_train = torch.tensor(true_dist(X_f_train), dtype=torch.float32)
    forcing_f_train = torch.tensor(true_forcing(X_f_train), dtype=torch.float32)

    # Move data to the selected device
    X_u_train = X_u_train.to(device)
    u_train = u_train.to(device)
    X_f_train = X_f_train.to(device)
    X_f_train.requires_grad = True
    dist_f_train = dist_f_train.to(device)
    forcing_f_train = forcing_f_train.to(device)

    class PINN(nn.Module):
        def __init__(self, layers):
            super(PINN, self).__init__()
            self.layers = layers
            self.num_layers = len(layers)
            self.loss_function = nn.MSELoss()

            # Neural Network
            self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
            self.linears_m = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]) # separate NN for m since m varies spatially
            self.activation = nn.Tanh()
            
            self.best_loss = 1e10

            # self.m = nn.Parameter(torch.tensor(0.1), requires_grad=True)
            # self.m = torch.tensor(0.0, dtype=torch.float32)

            # Xavier Initialization
            for i in range(len(self.linears)):
                nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
                nn.init.xavier_normal_(self.linears_m[i].weight.data, gain=1.0)
                if self.linears[i].bias is not None:
                    nn.init.zeros_(self.linears[i].bias.data)
                if self.linears_m[i].bias is not None:
                    nn.init.zeros_(self.linears_m[i].bias.data)
        
        def forward(self, x):
            # print(x.shape)
            a_ = torch.tensor(a, dtype=torch.float32)
            b_ = torch.tensor(b, dtype=torch.float32)
            #preprocessing input 
            x = (x - a_)/(b_ - a_) #feature scaling
            # compute m first
            x_copy = x.clone()
            m = self.m_compute(x_copy)
            
            for i, linear in enumerate(self.linears[:-1]):
                x = self.activation(linear(x))
            x = self.linears[-1](x)  # No activation on the last layer (u)
            return x, m
        
        def m_compute(self, x):
            # print(x.shape)
            a_ = torch.tensor(a, dtype=torch.float32)
            b_ = torch.tensor(b, dtype=torch.float32)
            #preprocessing input 
            
            for i, linear in enumerate(self.linears_m[:-1]):
                x = self.activation(linear(x))
            x = self.linears_m[-1](x)  # No activation on the last layer (u)
            return x
        
        def compute_loss(self, x_u, u_true, x_f):
            x_f.requires_grad = True
            # Calculate u from the network at boundary points and collocation points
            u_f_pred, m_pred = self.forward(x_f)
            
            # Compute f (similar to TensorFlow implementation)
            u_f_pred_grads = torch.autograd.grad(u_f_pred, x_f, grad_outputs=torch.ones_like(u_f_pred), retain_graph=True, create_graph=True)[0]
            u_f_pred_xx = torch.autograd.grad(torch.exp(m_pred) * u_f_pred_grads, x_f, grad_outputs=torch.ones_like(u_f_pred_grads[:,[0]]), create_graph=True)[0]

            rhs = forcing_f_train
            f_pred = - u_f_pred_xx + omega * dist_f_train * u_f_pred - rhs
            
            # Calculate the loss
            loss_f = self.loss_function(f_pred, torch.zeros_like(f_pred)) # residual loss
            u_pred, _ = self.forward(x_u)
            loss_u = self.loss_function(u_pred, u_true) # boundary loss
            loss = loss_f + loss_u

            return loss, loss_u, loss_f

        def train(self, x_u, u_true, x_f, epochs, lr, log_path):
            # Move model to the selected device
            self.to(device) 
            # adam optimizer
            optimizer = torch.optim.Adam(self.parameters(), lr=lr,betas=(0.9, 0.999), eps=1e-07)
            
            start_time = time.time()
            for epoch in range(epochs): # loop over the dataset multiple times
                optimizer.zero_grad()
                loss, loss_u, loss_f = self.compute_loss(x_u, u_true, x_f)
                loss.backward()
                optimizer.step()

                if loss < self.best_loss:
                    self.best_loss = loss
                    self.best_model = deepcopy(self.state_dict())
                    self.best_u, self.best_m = self.forward(x_u)
                
                if (epoch+1) % 1000 == 0: # Print the loss every 100 epochs
                    end_time = time.time()
                    print(f"Epoch {(epoch+1)}, loss_misfit: {loss_u.item():1.2e}, loss_f: {loss_f.item():1.2e}, loss: {loss.item():1.2e}, run time: {end_time - start_time}s")
                    start_time = time.time()
    
    
    ## ======== definition of model ends here. Start defining parameter ========
    # layers = [1, 50, 50, 50, 50, 50, 50, 1]
    model = PINN(layers)
    model.to(device)
    epochs = 300000
    learning_rate = 1e-4

    start_time = time.time()
    model.train(X_u_train, u_train, X_f_train, epochs, learning_rate)
    Adam_time = time.time() - start_time
    print('Training time: {:.4f} seconds'.format((Adam_time)))

    start_time1 = time.time()
    ## L-BFGS
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1, tolerance_grad=1e-10, history_size=100, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss, loss_u, loss_f = model.compute_loss(X_u_train, u_train, X_f_train)
        loss.backward()
        return loss
    for lbfgs_iter in range(1):
        optimizer.step(closure)
        loss, loss_u, loss_f = model.compute_loss(X_u_train, u_train, X_f_train)
        if loss < model.best_loss:
            model.best_loss = loss
            model.best_model = deepcopy(model.state_dict())
            model.best_u, model.best_m = model.forward(X_u_train)
        print(f"LBFGS: Loss_misfit: {loss_u.item():1.2e}, loss_f: {loss_f.item():1.2e}, loss: {loss.item():1.2e}")
        # print(f"LBFGS: Loss_misfit: {loss_u.item():1.2e}, loss_f: {loss_f.item():1.2e}, loss: {loss.item():1.2e} , exp(m): {np.exp(model.m.item()):.5f}")
    LBFGS_time = time.time() - start_time1
    print('LBFGS time: {:.4f} seconds'.format((LBFGS_time)))


    # u, m = model.forward(X_u_train)
    u, m = model.best_u, model.best_m

    ## ===================== save model =====================
    import sys
    import json
    from utils import increment_path
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d-%H%M%S")
    dir_name = f"PINN_results_{dt_string}"
    path_ = f'/g/g20/ho32/PINNvsFEM/Pytorch-PINN-Marine-Lake/Model_Discovery/log/{dir_name}'
    save_path = increment_path(path_, mkdir=True)

    # define results dict
    results = {
        'Nu': Nu,
        'Nf': Nf,
        'adam_runtime': Adam_time,
        'lbfgs_runtime': LBFGS_time,
        'x_u_train': X_u_train.cpu().tolist(),
        'u_train': u_train.cpu().tolist(),
        'u_sol': u.detach().cpu().tolist(),
        'm_sol': m.detach().cpu().tolist(),
        'x_for_m_true': x_array.tolist(),
        'm_true': mtrue_array.tolist(),
    }

    # save results and model
    with open(f'{save_path}/results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    Nu_list = [64, 128, 256, 512, 1024,2048]
    Nf_list = [1000,5000,10000]
    
    layers_list = [[1, 50, 50, 50, 50, 50, 50, 1],
                   [1, 50, 50, 50, 50, 50, 1],
                   [1, 50, 50, 50, 50, 1],
                    ]

    # for Nu in Nu_list:
    #     for Nf in Nf_list:
    #         main(Nu,Nf)

    main(64,1000, layers_list[0])
