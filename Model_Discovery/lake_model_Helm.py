import torch
import torch.nn as nn
import numpy as np
import time
import pandas as pd
import os
from pyDOE import lhs # Latin Hypercube Sampling

# Check if CUDA is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

file_path = os.path.dirname(__file__)

Nu = 256
Nf = 1000
a, b = 0.0, 1.0

# Load True Solution and paramters
df = pd.read_csv(f'{file_path}/synthetic_data/synthetic_data.csv')
x_array, mtrue_array, utrue_array = np.array(df['x']), np.array(df['mtrue']), np.array(df['utrue'])
forcing_array, dist_array = np.array(df['forcing']), np.array(df['distribution'])
omega = np.array(df['omega'])[0]

from scipy.interpolate import interp1d
true_solution = interp1d(x_array, utrue_array)
true_dist = interp1d(x_array, dist_array)
true_forcing = interp1d(x_array, forcing_array)
omega = torch.tensor(omega, dtype=torch.float32, device=device)
# def true_solution(x):
#     return np.sin(2 * torch.pi * x) / ((2* torch.pi)**2 + 10)

X_u_train = torch.linspace(a,b, Nu, dtype=torch.float32)[:, None]
u_train = torch.tensor(true_solution(X_u_train), dtype=torch.float32)
X_f_train = torch.tensor(lhs(1, Nf), dtype=torch.float32)
dist_f_train = torch.tensor(true_dist(X_f_train), dtype=torch.float32)
forcing_f_train = torch.tensor(true_forcing(X_f_train), dtype=torch.float32)

# Move data to the selected device
X_u_train = X_u_train.to(device)
u_train = u_train.to(device)
X_f_train = X_f_train.to(device)
dist_f_train = dist_f_train.to(device)
forcing_f_train = forcing_f_train.to(device)



class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = layers
        self.loss_function = nn.MSELoss()

        # Neural Network
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.linears_m = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]) # separate NN for m since m varies spatially
        self.activation = nn.Tanh()

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
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x)  # No activation on the last layer (u)
        return x
    
    def m_compute(self, x):
        # print(x.shape)
        a_ = torch.tensor(a, dtype=torch.float32)
        b_ = torch.tensor(b, dtype=torch.float32)
        #preprocessing input 
        x = (x - a_)/(b_ - a_) #feature scaling
        for linear in self.linears_m[:-1]:
            x = self.activation(linear(x))
        x = self.linears_m[-1](x)  # No activation on the last layer (u)
        return x
    
    def compute_loss(self, x_u, u_true, x_f):
        x_f.requires_grad = True
        # Calculate u from the network at boundary points and collocation points
        u_f_pred = self.forward(x_f)
        m_pred = self.m_compute(x_f)
        
        # Compute f (similar to TensorFlow implementation)
        u_f_pred_grads = torch.autograd.grad(u_f_pred, x_f, grad_outputs=torch.ones_like(u_f_pred), retain_graph=True, create_graph=True)[0]
        u_f_pred_xx = torch.autograd.grad(torch.exp(m_pred) * u_f_pred_grads, x_f, grad_outputs=torch.ones_like(u_f_pred_grads[:,[0]]), create_graph=True)[0]

        rhs = forcing_f_train
        # rhs = (torch.sin(2 * torch.pi * x_f)) 


        f_pred = - u_f_pred_xx + omega * dist_f_train * u_f_pred - rhs
        # f_pred = - u_f_pred_xx + 10 * u_f_pred - rhs
        
        # Calculate the loss
        loss_f = self.loss_function(f_pred, torch.zeros_like(f_pred)) # residual loss
        u_pred = self.forward(x_u)
        loss_u = self.loss_function(u_pred, u_true) # boundary loss
        loss = loss_f + loss_u

        return loss, loss_u, loss_f

    def train(self, x_u, u_true, x_f, epochs, lr):
        # Move model to the selected device
        self.to(device) 
        # adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr,betas=(0.9, 0.999), eps=1e-07)

        for epoch in range(epochs): # loop over the dataset multiple times
            optimizer.zero_grad()
            loss, loss_u, loss_f = self.compute_loss(x_u, u_true, x_f)
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0: # Print the loss every 100 epochs
                print(f"Epoch {epoch}, loss_misfit: {loss_u.item():1.2e}, loss_f: {loss_f.item():1.2e}, loss: {loss.item():1.2e}")
                # print(f"Epoch {epoch}, loss_misfit: {loss_u.item():1.2e}, loss_f: {loss_f.item():1.2e}, loss: {loss.item():1.2e} , exp(m): {np.exp(self.m.item()):.5f}")





layers = [1, 50, 50, 50, 50, 50, 50, 1]
model = PINN(layers)
model.to(device)
epochs = 1000
learning_rate = 1e-4

start_time = time.time()
model.train(X_u_train, u_train, X_f_train, epochs, learning_rate)
print('Training time: {:.4f} minutes'.format((time.time() - start_time)/60))

## L-BFGS
optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=50000, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn="strong_wolfe")
def closure():
    optimizer.zero_grad()
    loss, loss_u, loss_f = model.compute_loss(X_u_train, u_train, X_f_train)
    loss.backward()
    return loss
for lbfgs_iter in range(1):
    optimizer.step(closure)
    loss, loss_u, loss_f = model.compute_loss(X_u_train, u_train, X_f_train)
    print(f"LBFGS: Loss_misfit: {loss_u.item():1.2e}, loss_f: {loss_f.item():1.2e}, loss: {loss.item():1.2e}")
    # print(f"LBFGS: Loss_misfit: {loss_u.item():1.2e}, loss_f: {loss_f.item():1.2e}, loss: {loss.item():1.2e} , exp(m): {np.exp(model.m.item()):.5f}")

u = model.forward(X_u_train)
m = model.m_compute(X_u_train)

import matplotlib.pyplot as plt

# error, error_label = np.sqrt(nmse(np.exp(m.compute_vertex_values()),  np.exp(mtrue_array))), 'NRMSE'
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 3.6)) 
ax[0].plot(X_u_train.cpu(),u_train.cpu(), '-', label=f'$u_d$: $n_x = {Nu}$', markersize=2, linewidth=2.5)
ax[0].plot(X_u_train.cpu(),u.detach().cpu(), '--', label=r'$u_{sol}$', markersize=2, linewidth=2.5)
# ax[0].plot(x, u_analytic, 'o', label=r'$A\sin(2\pi x)$', markersize=2, linewidth=2.5)
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$u$')
ax[0].set_title(f'Data $u_d$ vs. Solution $u$: MSE = {loss_u.item():1.2e}', fontsize=15)
ax[0].grid()
ax[0].legend(fontsize=12)
ax[1].plot(x_array, mtrue_array, '-', label=r'$m_{true}$', color='tab:blue', linewidth=2.5)
# ax[1].plot(X_u_train,np.ones(Nu), '-', label=r'$m_{true}$', color='tab:blue', linewidth=2.5)
ax[1].plot(X_u_train.cpu(),m.detach().cpu(), '--', label=r'$m_{sol}$', color='tab:red', linewidth=2.5)
# ax[1].set_title(f'gamma = {gamma[kappa_min_ind]:.3e}')
ax[1].set_xlabel(r'$x$')
ax[1].set_ylabel(r'$m$')
ax[1].set_title(f'Recovered vs. True Param', fontsize=15)
ax[1].grid()
ax[1].legend(fontsize=12)

ax[2].plot(X_u_train.cpu(), abs(u_train.cpu() - u.detach().cpu()))
# ax[2].plot(x, run.g_sun.compute_vertex_values())
ax[2].set_title(f'Data Misfit: \n $|u - u_d|$', fontsize=15)
ax[2].set_xlabel(r'$x$')
ax[2].set_ylabel(r'$|u_{sol} - u_d|$')
ax[2].grid()

print(f'MSE = {torch.mean((u_train.cpu() - u.detach().cpu())**2)}')
# plt.savefig('Helm.png', dpi=300)
plt.show()
