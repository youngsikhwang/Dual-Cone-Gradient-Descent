import torch
import numpy as np
from tqdm import tqdm
from Network import DNN
from dcgd import DCGD

class PINN_Helmholtz():
    def __init__(self, samplers, test_data, layers, batch_size, lr, optim, dcgd, device):

        self.ics_sampler = samplers[0]
        self.bcs_sampler = samplers[1]
        self.pde_sampler = samplers[2]

        X, _ = self.pde_sampler.sample(np.int32(1e5))

        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_x1, self.sigma_x1 = self.mu_X[0], self.sigma_X[0]
        self.mu_x2, self.sigma_x2 = self.mu_X[1], self.sigma_X[1]

        self.X_test = test_data[0]
        self.u_test = test_data[1]

        self.optim = optim
        self.dcgd = dcgd
        self.lr = lr
 
        self.batch_size = batch_size

        self.k = 1
        self.dnn = DNN(layers).to(device)

        self.device = device

        self.best_test_error = None
        self.best_loss = torch.inf

        self.base_optimizer = torch.optim.Adam(self.dnn.parameters(), self.lr)

        if dcgd == 'center':
           self.optimizer = DCGD(self.base_optimizer, num_pde=1, type='center')
        elif dcgd == 'avg':
           self.optimizer = DCGD(self.base_optimizer, num_pde=1, type='avg')
        elif dcgd == 'proj':
           self.optimizer = DCGD(self.base_optimizer, num_pde=1, type='proj')
        else:
           self.optimizer = self.base_optimizer

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
    
    def net_u(self, x1, x2):
        u = self.dnn(torch.cat([x1, x2], dim=1))
        return u
    
    def net_f(self, x1, x2):
        u = self.net_u(x1, x2)
        u_x1 = torch.autograd.grad(
              u, x1,
              grad_outputs=torch.ones_like(u),
              retain_graph=True,
              create_graph=True
              )[0]/self.sigma_x1
        u_x2 = torch.autograd.grad(
              u, x2,
              grad_outputs=torch.ones_like(u),
              retain_graph=True,
              create_graph=True
              )[0]/self.sigma_x2
        u_xx1 = torch.autograd.grad(
              u_x1, x1,
              grad_outputs=torch.ones_like(u_x1),
              retain_graph=True,
              create_graph=True
              )[0]/self.sigma_x1
        u_xx2 = torch.autograd.grad(
              u_x2, x2,
              grad_outputs=torch.ones_like(u_x2),
              retain_graph=True,
              create_graph=True
              )[0]/self.sigma_x2
        
        f = u_xx1 + u_xx2 + (self.k**2) * u
        return f
    
    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        X = torch.tensor(X, requires_grad=True).float().to(self.device)
        Y = torch.tensor(Y, requires_grad=True).float().to(self.device)
        return X, Y

    def loss_func(self):
        X_bc1_batch, u_bc1_batch = self.fetch_minibatch(self.bcs_sampler[0], self.batch_size)
        X_bc2_batch, u_bc2_batch = self.fetch_minibatch(self.bcs_sampler[1], self.batch_size)
        X_bc3_batch, u_bc3_batch = self.fetch_minibatch(self.bcs_sampler[2], self.batch_size)
        X_bc4_batch, u_bc4_batch = self.fetch_minibatch(self.bcs_sampler[3], self.batch_size)
        X_pde_batch, f_pde_batch = self.fetch_minibatch(self.pde_sampler, self.batch_size*10)

        u1_pred = self.net_u(X_bc1_batch[:,0:1], X_bc1_batch[:,1:2])
        u2_pred = self.net_u(X_bc2_batch[:,0:1], X_bc2_batch[:,1:2])
        u3_pred = self.net_u(X_bc3_batch[:,0:1], X_bc3_batch[:,1:2])
        u4_pred = self.net_u(X_bc4_batch[:,0:1], X_bc4_batch[:,1:2])
        
        f_pred = self.net_f(X_pde_batch[:,0:1], X_pde_batch[:,1:2])
        

        loss_u1 = torch.mean(u1_pred ** 2)
        loss_u2 = torch.mean(u2_pred ** 2)
        loss_u3 = torch.mean(u3_pred ** 2)     
        loss_u4 = torch.mean(u4_pred ** 2)                 

        loss_u = loss_u1 + loss_u2 + loss_u3 +loss_u4 

        loss_f = torch.mean((f_pred-f_pde_batch)**2)

        return [loss_f, loss_u]

    def train(self, nepoch):
        if self.optim != 'adam':
            for epoch in tqdm(range(nepoch)):
                self.dnn.train()
                losses = self.loss_func()
                isstop = self.optimizer.step(losses)
                if isstop:
                    print(f'Gradient conflict occur: {epoch+1}')
                    break
                self.scheduler.step()
                if sum(losses) < self.best_loss:
                    self.best_loss = sum(losses)
                    self.evaluation()

        else:
            for epoch in tqdm(range(nepoch)):
                self.dnn.train()
                loss = sum(self.loss_func())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.evaluation()         

    def evaluation(self):
       self.dnn.eval()
       u_pred, f_pred = self.predict(self.X_test)
       error_u = np.linalg.norm(self.u_test-u_pred,2)/np.linalg.norm(self.u_test,2)
       self.best_test_error = error_u

    def predict(self, X):
        X = (X-self.mu_X)/self.sigma_X
        x1 = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        x2 = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        
        self.dnn.eval()
        u = self.net_u(x1, x2)
        f = self.net_f(x1, x2)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f    