import torch
import torch.nn as nn
import numpy as np
import time
import pandas as pd
import os
import csv
from copy import deepcopy
import matplotlib.pyplot as plt
from pyDOE import lhs # Latin Hypercube Sampling

# Check if CUDA is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

file_path = os.getcwd()
a, b = 0.0, 1.0

# Load True Solution and paramters
# df = pd.read_csv(f'/g/g20/ho32/PINNvsFEM/Pytorch-PINN-Marine-Lake/Model_Discovery/synthetic_data/synthetic_data.csv')

# df = pd.read_csv(f'/g/g20/ho32/PINNvsFEM/Pytorch-PINN-Marine-Lake/Model_Discovery/synthetic_data/synthetic_data_double_data.csv')
# df = pd.read_csv(f'/Users/alexho/Dropbox/2024_spring/PINN_testing/Pytorch_PINN/Model_Discovery/synthetic_data/synthetic_data_double_data.csv')
df = pd.read_csv('/home/aho38/PINNvsFEM/Pytorch-PINN-Marine-Lake/Model_Discovery/synthetic_data/synthetic_data_double_data.csv')
x_array, mtrue_array, utrue1_array, utrue2_array = np.array(df['x']), np.array(df['mtrue']), np.array(df['utrue1']), np.array(df['utrue2'])
forcing1_array, forcing2_array, dist_array = np.array(df['g1']), np.array(df['g2']), np.array(df['distribution'])
u1_noise, u2_noise = np.array(df['u1_noise']), np.array(df['u2_noise'])
omega = np.array(df['omega'])[0]

from scipy.interpolate import interp1d
true_m = interp1d(x_array, mtrue_array)
# true_solution = interp1d(x_array, utrue_array)
true_solution1 = interp1d(x_array, utrue1_array)
true_solution2 = interp1d(x_array, utrue2_array)
true_dist = interp1d(x_array, dist_array)
# true_forcing = interp1d(x_array, forcing_array)
true_forcing1 = interp1d(x_array, forcing1_array)
true_forcing2 = interp1d(x_array, forcing2_array)
omega = torch.tensor(omega, dtype=torch.float32, device=device)
u1_noise_interp = interp1d(x_array, u1_noise)
u2_noise_interp = interp1d(x_array, u2_noise)



def main(Nu,Nf,layers,noise_level):
    a, b = 0.0, 1.0

    # define training data and add noise
    X_u_train = torch.linspace(a,b, Nu, dtype=torch.float32)[:, None]
    # X_u_train1 = torch.linspace((b-a)/2 + a,b, Nu, dtype=torch.float32)[:, None]
    # X_u_train2 = torch.linspace(a,(b-a)/2 + a, Nu, dtype=torch.float32)[:, None]
    X_m_boundary = torch.tensor([a,b], dtype=torch.float32)[:, None]
    X_u_train1 = torch.linspace(a,b, Nu, dtype=torch.float32)[:, None]
    X_u_train2 = torch.linspace(a,b, Nu, dtype=torch.float32)[:, None]
    # u_train = torch.tensor(true_solution(X_u_train), dtype=torch.float32)
    m_bc = torch.tensor(true_m(X_m_boundary), dtype=torch.float32)
    u_train1 = torch.tensor(true_solution1(X_u_train1), dtype=torch.float32)
    u_train2 = torch.tensor(true_solution2(X_u_train2), dtype=torch.float32)
    u_train1_noise = torch.tensor(u1_noise_interp(X_u_train1), dtype=torch.float32)
    u_train2_noise = torch.tensor(u2_noise_interp(X_u_train2), dtype=torch.float32)
    #noise_level = 0.0
    np.random.seed(0)
    # u_train += torch.randn(u_train.shape) * abs(u_train).max() * noise_level
    u_train1 += u_train1_noise * noise_level
    u_train2 += u_train2_noise * noise_level


    X_f_train1 = X_f_train2 = torch.tensor(lhs(1, Nf), dtype=torch.float32)
    # X_f_train2 = torch.tensor(lhs(1, Nf), dtype=torch.float32)
    dist_f_train1 = torch.tensor(true_dist(X_f_train1), dtype=torch.float32)
    dist_f_train2 = torch.tensor(true_dist(X_f_train2), dtype=torch.float32)
    # forcing_f_train = torch.tensor(true_forcing(X_f_train), dtype=torch.float32)
    forcing_f_train1 = torch.tensor(true_forcing1(X_f_train1), dtype=torch.float32)
    forcing_f_train2 = torch.tensor(true_forcing2(X_f_train2), dtype=torch.float32)

    # Move data to the selected device
    X_m_boundary = X_m_boundary.to(device)
    X_u_train = X_u_train.to(device)
    X_u_train1 = X_u_train1.to(device)
    X_u_train2 = X_u_train2.to(device)
    # u_train = u_train.to(device)
    m_bc = m_bc.to(device)
    u_train1 = u_train1.to(device)
    u_train2 = u_train2.to(device)
    X_f_train1 = X_f_train1.to(device)
    X_f_train2 = X_f_train2.to(device)
    X_f_train1.requires_grad = True
    X_f_train2.requires_grad = True
    # dist_f_train = dist_f_train.to(device)
    dist_f_train1 = dist_f_train1.to(device)
    dist_f_train2 = dist_f_train2.to(device)
    # forcing_f_train = forcing_f_train.to(device)
    forcing_f_train1 = forcing_f_train1.to(device)
    forcing_f_train2 = forcing_f_train2.to(device)

    class PINN(nn.Module):
        def __init__(self, layers: list, betas: list):
            
            super(PINN, self).__init__()
            self.layers = layers
            self.betas = betas
            self.num_layers = len(layers)
            self.loss_function = nn.MSELoss()

            # Neural Network
            self.linears_1 = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
            self.linears_2 = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
            self.linears_m = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]) # separate NN for m since m varies spatially
            self.activation = nn.Tanh()
            
            self.best_loss = 1e10

            # self.m = nn.Parameter(torch.tensor(0.1), requires_grad=True)
            # self.m = torch.tensor(0.0, dtype=torch.float32)

            # Xavier Initialization
            for i in range(len(self.linears_1)):
                nn.init.xavier_normal_(self.linears_1[i].weight.data, gain=1.0)
                nn.init.xavier_normal_(self.linears_2[i].weight.data, gain=1.0)
                nn.init.xavier_normal_(self.linears_m[i].weight.data, gain=1.0)
                if self.linears_1[i].bias is not None:
                    nn.init.zeros_(self.linears_1[i].bias.data)
                if self.linears_2[i].bias is not None:
                    nn.init.zeros_(self.linears_2[i].bias.data)
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
            # u2_out = self.forward2(x_copy)
            u_out = self.linears_1[0](x)
            
            for linear in self.linears_1[1:-1]:
                u_out = self.activation(linear(u_out))
            u_out = self.linears_1[-1](u_out)  # No activation on the last layer (u)
            return u_out,m
        
        def forward2(self, x):
            '''This is for ud2. forward if for ud1'''
            x_copy = x.clone()
            m = self.m_compute(x_copy)
            u_out = self.linears_2[0](x)

            for linear in self.linears_2[1:-1]:
                u_out = self.activation(linear(u_out))
            u_out = self.linears_2[-1](u_out)
            return u_out,m


        
        def m_compute(self, x):
            # print(x.shape)
            a_ = torch.tensor(a, dtype=torch.float32)
            b_ = torch.tensor(b, dtype=torch.float32)
            #preprocessing input 
            m_out = self.linears_m[0](x)
            
            for i, linear in enumerate(self.linears_m[1:-1]):
                m_out = self.activation(linear(m_out))
            m_out = self.linears_m[-1](m_out)  # No activation on the last layer (u)
            return m_out
        
        def compute_loss(self, x_u1, x_u2, u_true1, u_true2, x_f1, x_f2, x_m_bc, m_bc):
            # x_f.requires_grad = True
            x_f1.requires_grad = True
            x_f2.requires_grad = True
            # Calculate u from the network at boundary points and collocation points
            u_f_pred, m_pred1 = self.forward(x_f1)
            u2_f_pred, m_pred2 = self.forward2(x_f2)
            self.u_f_pred = u_f_pred
            self.u2_f_pred = u2_f_pred
            self.m_pred1 = m_pred1
            
            # Compute f1 (similar to TensorFlow implementation)
            u_f_pred_grads = torch.autograd.grad(u_f_pred, x_f1, grad_outputs=torch.ones_like(u_f_pred), retain_graph=True, create_graph=True)[0]
            u_f_pred_xx = torch.autograd.grad(torch.exp(m_pred1) * u_f_pred_grads, x_f1, grad_outputs=torch.ones_like(u_f_pred_grads[:,[0]]), create_graph=True)[0]

            # compute residual for u1
            rhs = forcing_f_train1
            # f_pred = - u_f_pred_xx + omega * dist_f_train * u_f_pred - rhs
            f1_pred = - u_f_pred_xx + omega * dist_f_train1 * u_f_pred - rhs
            loss_f1 = self.loss_function(f1_pred, torch.zeros_like(f1_pred)) # residual loss

            # Compute f2
            u2_f_pred_grads = torch.autograd.grad(u2_f_pred, x_f2, grad_outputs=torch.ones_like(u2_f_pred), retain_graph=True, create_graph=True)[0]
            u2_f_pred_xx = torch.autograd.grad(torch.exp(m_pred2) * u2_f_pred_grads, x_f2, grad_outputs=torch.ones_like(u2_f_pred_grads[:,[0]]), create_graph=True)[0]

            # compute the residual of u2
            rhs2 = forcing_f_train2
            f2_pred = - u2_f_pred_xx + omega * dist_f_train2 * u2_f_pred - rhs2
            
            # Calculate the loss
            # loss_f = self.loss_function(f_pred, torch.zeros_like(f_pred)) # residual loss
            loss_f2 = self.loss_function(f2_pred, torch.zeros_like(f2_pred))

            u_pred, _ = self.forward(x_u1)
            u2_pred, _ = self.forward2(x_u2)
            # loss_u = self.loss_function(u_pred, u_true) # boundary loss
            loss_u1 = self.loss_function(u_pred, u_true1)
            loss_u2 = self.loss_function(u2_pred, u_true2)

            # Compute loss for m bc
            m_bc_pred = self.m_compute(x_m_bc)
            loss_m_bc = self.loss_function(m_bc_pred, m_bc)

            # loss = loss_f + loss_u
            loss = (loss_f1 + loss_u1)*self.betas[0] + (loss_f2 + loss_u2)*self.betas[1] + loss_m_bc*self.betas[2]

            return loss, loss_f1 , loss_f2 , loss_u1 , loss_u2, loss_m_bc

        def training_step(self, x_u1, x_u2, u_true1, u_true2, x_f1, x_f2, epochs, lr, log_path, model_save_path, x_m_bc, m_bc):
            # Move model to the selected device
            self.to(device) 
            # adam optimizer
            if self.betas[0] == 1 and self.betas[1] == 1:
                optimizer = torch.optim.Adam(self.parameters(), lr=lr,betas=(0.9, 0.999), eps=1e-07)
            elif self.betas[0] == 1 and self.betas[1] == 0:
                params_to_optimize = [param for name, param in self.named_parameters() if ('linears_1' in name) or ('linears_m' in name)]
                optimizer = torch.optim.Adam(params_to_optimize, lr=lr,betas=(0.9, 0.999), eps=1e-07)
            elif self.betas[0] == 0 and self.betas[1] == 1:
                params_to_optimize = [param for name, param in self.named_parameters() if ('linears_2' in name) or ('linears_m' in name)]
                optimizer = torch.optim.Adam(params_to_optimize, lr=lr,betas=(0.9, 0.999), eps=1e-07)


            import csv
            if os.path.exists(log_path):
                pass
            else:
                with open(log_path, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['epoch', 'loss_misfit1', 'loss_misfit2', 'loss_f1', 'loss_f2', 'loss_mbc', 'loss', 'run_time', 'm_sol', 'u_sol1', 'u_sol2'])
            
            start_time = time.time()
            for epoch in range(epochs): # loop over the dataset multiple times
                optimizer.zero_grad()
                self.train()
                # if epoch % 2 == 0:
                #     self.betas = [1.0, 0.0]
                # else: 
                #     self.betas = [0.0, 1.0]
                loss, loss_f1 , loss_f2 , loss_u1 , loss_u2, loss_m_bc = self.compute_loss(x_u1, x_u2, u_true1, u_true2, x_f1, x_f2, x_m_bc, m_bc)
                # save the best models before updating the parameters

                if loss < self.best_loss:
                    self.best_loss = loss
                    self.best_model = deepcopy(self.state_dict())
                    self.best_u1, self.best_m = self.forward(X_u_train)
                    self.best_u2, _ = self.forward2(X_u_train)


                
                if (epoch+1) % 100 == 0: # Print the loss every 100 epochs
                    end_time = time.time()
                    print(f"Epoch {(epoch+1)}, loss_misfit1: {loss_u1.item():1.2e}, loss_misfit2: {loss_u2.item():1.2e}, loss_f1: {loss_f1.item():1.2e}, loss_f2: {loss_f2.item():1.2e}, loss_mbc: {loss_m_bc.item():1.2e}, loss: {loss.item():1.2e}, best_loss: {self.best_loss.item():1.2e}, run time: {end_time - start_time}s")

                    model_save = deepcopy(self.state_dict())
                    torch.save(model_save, f'{model_save_path}/model_epoch{epoch+1}.pt')

                    with open(log_path, 'a') as file:
                        self.eval()
                        with torch.no_grad():
                            u1,m = self.forward(X_u_train)
                            u2,_ = self.forward2(X_u_train)

                        writer = csv.writer(file)
                        writer.writerow([epoch+1, loss_u1.item(), loss_u2.item(),loss_f1.item(),loss_f2.item(),loss_m_bc.item(),loss.item(),end_time - start_time,m.detach().cpu().tolist(),u1.detach().cpu().tolist(),u2.detach().cpu().tolist()])
    
                    start_time = time.time()


                ##### Update my weights according to my loss fucntions
                loss.backward()
                optimizer.step()


    ## ======== definition of model ends here. Start defining parameter ========
    ## ===================== save model =====================
    import sys
    import json
    from utils import increment_path
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d-%H%M%S")
    dir_name = f"PINN_results_noise{int(noise_level*100)}_{dt_string}"
    path_ = f'/home/aho38/data/log/NBC_PINN_results_Helm_with_mbc/Nu_{Nu}_Nf_{Nf}/{dir_name}'
    save_path = increment_path(path_, mkdir=True)

    log_path = f'{save_path}/opt_log.csv'
    model_save_path = f'{save_path}/models'
    os.mkdir(model_save_path)


    # layers = [1, 50, 50, 50, 50, 50, 50, 1]
    #       [u1,  u2,  mbc]
    betas = [0.0, 1.0, 1.0]
    model = PINN(layers,betas)
    # print(PINN)
    model.to(device)
    epochs = 1000000
    learning_rate = 1e-4

    # training first with beta1 = 1 and beta 2 = 0
    start_time = time.time()
    model.training_step(X_u_train1, X_u_train2, u_train1, u_train2, X_f_train1, X_f_train2, epochs, learning_rate, log_path, model_save_path, X_m_boundary, m_bc)
    Adam_time = time.time() - start_time
    print('Training time: {:.4f} seconds'.format((Adam_time)))

    start_time1 = time.time()
    ## L-BFGS
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1, tolerance_grad=1e-10, history_size=100, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        # loss, loss_f1 , loss_f2 , loss_u1 , loss_u2 = model.compute_loss(X_u_train, u_train1, u_train2, X_f_train)
        loss, loss_f1 , loss_f2 , loss_u1 , loss_u2, loss_m_bc = model.compute_loss(X_u_train1, X_u_train2, u_train1, u_train2, X_f_train1, X_f_train2, X_m_boundary, m_bc)
        loss.backward()
        return loss
    for lbfgs_iter in range(1):
        optimizer.step(closure)
        loss, loss_f1 , loss_f2 , loss_u1 , loss_u2, loss_m_bc = model.compute_loss(X_u_train1, X_u_train2, u_train1, u_train2, X_f_train1, X_f_train2, X_m_boundary, m_bc)
        if loss < model.best_loss:
            model.best_loss = loss
            model.best_model = deepcopy(model.state_dict())
            model.best_u1, model.best_m = model.forward(X_u_train)
            model.best_u2, _ = model.forward2(X_u_train)
        print(f"LBFGS: loss_misfit1: {loss_u1.item():1.2e}, loss_misfit2: {loss_u2.item():1.2e}, loss_f1: {loss_f1.item():1.2e}, loss_f2: {loss_f2.item():1.2e}, loss_mbc: {loss_m_bc.item():1.2e}, loss: {loss.item():1.2e}")
        # print(f"LBFGS: Loss_misfit: {loss_u.item():1.2e}, loss_f: {loss_f.item():1.2e}, loss: {loss.item():1.2e} , exp(m): {np.exp(model.m.item()):.5f}")
    LBFGS_time = time.time() - start_time1
    print('LBFGS time: {:.4f} seconds'.format((LBFGS_time)))


    # u, m = model.forward(X_u_train)
    # u, m = model.best_u, model.best_m
    u1, u2, m = model.best_u1, model.best_u2, model.best_m
    # u_train1 = torch.tensor(true_solution1(X_u_train), dtype=torch.float32)
    # u_train2 = torch.tensor(true_solution2(X_u_train), dtype=torch.float32)

    # define results dict
    #fig, ax = plt.subplots(1,3, figsize=(18, 5))
    #ax[0].plot(X_u_train.cpu(), m.detach().cpu(), 'r')
    #ax[0].plot(x_array, mtrue_array, 'b')

    #ax[1].plot(X_u_train.cpu(), u1.detach().cpu(), 'r')
    #ax[1].plot(X_u_train.cpu(), u_train1.cpu(), 'b')

    #ax[2].plot(X_u_train.cpu(), u2.detach().cpu(), 'r')
    #ax[2].plot(X_u_train.cpu(), u_train2.cpu(), 'b')
    #plt.show()
    results = {
        'Nu': Nu,
        'Nf': Nf,
        'adam_runtime': Adam_time,
        'lbfgs_runtime': LBFGS_time,
        'layers': layers,
        'x_u_train': X_u_train.cpu().tolist(),
        # 'u_train': u_train.cpu().tolist(),
        'u_train1': u_train1.cpu().tolist(),
        'u_train2': u_train2.cpu().tolist(),
        # 'u_sol': u.detach().cpu().tolist(),
        'u_sol1': u1.detach().cpu().tolist(),
        'u_sol2': u2.detach().cpu().tolist(),
        'm_sol': m.detach().cpu().tolist(),
        'x_for_m_true': x_array.tolist(),
        'm_true': mtrue_array.tolist(),
    }

    # save results and model
    with open(f'{save_path}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # save best model
    torch.save(model.best_model, f'{save_path}/model.pt')

if __name__ == "__main__":
    Nu_list = [64, 128, 256, 512, 1024,2048]
    Nf_list = [1000,5000,10000]
    noise_level = 0.03

    layers_list = [
                #    [1, 20, 20, 20, 20, 20, 20, 20, 1],
                #    [1, 15, 15, 15, 15, 15, 15, 15, 1],
                #    [1, 50, 50, 50, 50, 50, 1],
                   [1, 50, 50, 50, 50, 1],
                #    [1, 50, 50, 50, 1],
                #   [1, 40, 40, 40, 40, 40,1],
                    ]

    # for Nu in Nu_list:
    #     for Nf in Nf_list:
    #         main(Nu,Nf)

    for layers in layers_list:
        # main(8,1000, layers, noise_level)
        # main(16,1000, layers, noise_level)
        # main(32,1000, layers, noise_level)
        main(64,1000, layers, noise_level)
        # main(128,1000,layers, noise_level)
        # main(256,1000, layers, noise_level)
        # main(512,1000, layers, noise_level)
