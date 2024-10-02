import torch
import numpy as np
from tqdm import tqdm
from Network import DNN
from dcgd import DCGD

class PINN_Klein_Gordon():
    def __init__(self, samplers, test_data, layers, batch_size, lr, optim, dcgd, device):

        self.ics_sampler = samplers[0]
        self.bcs_sampler = samplers[1]
        self.pde_sampler = samplers[2]

        X, _ = self.pde_sampler.sample(np.int32(1e5))

        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_t, self.sigma_t = self.mu_X[0], self.sigma_X[0]
        self.mu_x, self.sigma_x = self.mu_X[1], self.sigma_X[1]

        self.X_test = test_data[0]
        self.u_test = test_data[1]

        self.optim = optim
        self.dcgd = dcgd
        self.lr = lr
 
        self.batch_size = batch_size

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
    
    def net_u(self, t, x):
        u = self.dnn(torch.cat([t, x], dim=1))
        return u
    
    def net_u_t(self,t, x):
        u = self.net_u(t,x)
        u_t = torch.autograd.grad(
              u, t,
              grad_outputs=torch.ones_like(u),
              retain_graph=True,
              create_graph=True
              )[0]/self.sigma_t
        return u_t
    
    def net_f(self, t, x):
        u = self.net_u(t, x)
        u_x = torch.autograd.grad(
              u, x,
              grad_outputs=torch.ones_like(u),
              retain_graph=True,
              create_graph=True
              )[0]/self.sigma_x
        u_t = torch.autograd.grad(
              u, t,
              grad_outputs=torch.ones_like(u),
              retain_graph=True,
              create_graph=True
              )[0]/self.sigma_t
        u_xx = torch.autograd.grad(
              u_x, x,
              grad_outputs=torch.ones_like(u_x),
              retain_graph=True,
              create_graph=True
              )[0]/self.sigma_x
        u_tt = torch.autograd.grad(
              u_t, t,
              grad_outputs=torch.ones_like(u_t),
              retain_graph=True,
              create_graph=True
              )[0]/self.sigma_t
        
        f = u_tt - u_xx + (u**3)
        return f

    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        X = torch.tensor(X, requires_grad=True).float().to(self.device)
        Y = torch.tensor(Y, requires_grad=True).float().to(self.device)
        return X, Y

    def loss_func(self):
        
        self.x_ic_batch, self.u_ic_batch = self.fetch_minibatch(self.ics_sampler, self.batch_size)
        self.X_bc1_batch, self.u_bc1_batch = self.fetch_minibatch(self.bcs_sampler[0], self.batch_size)
        self.X_bc2_batch, self.u_bc2_batch = self.fetch_minibatch(self.bcs_sampler[1], self.batch_size)
        self.X_pde_batch, self.f_pde_batch = self.fetch_minibatch(self.pde_sampler, self.batch_size)

        u_ic1_pred = self.net_u(self.x_ic_batch[:,0:1], self.x_ic_batch[:,1:2])
        u_ic2_pred = self.net_u_t(self.x_ic_batch[:,0:1], self.x_ic_batch[:,1:2])

        u_bc1_pred = self.net_u(self.X_bc1_batch[:,0:1], self.X_bc1_batch[:,1:2])
        u_bc2_pred = self.net_u(self.X_bc2_batch[:,0:1], self.X_bc2_batch[:,1:2])
        
        f_pred = self.net_f(self.X_pde_batch[:,0:1], self.X_pde_batch[:,1:2])
        

        loss_u1 = torch.mean((u_ic1_pred-self.u_ic_batch) ** 2)
        loss_u2 = torch.mean(u_ic2_pred ** 2)
        loss_u3 = torch.mean((u_bc1_pred - self.u_bc1_batch)**2)
        loss_u4 = torch.mean((u_bc2_pred - self.u_bc2_batch)**2)           

        loss_ui = loss_u1 + loss_u2
        loss_ub = loss_u3 + loss_u4

        # loss_u = loss_u1 + loss_u2 + loss_u3 + loss_u4 

        loss_f = torch.mean((f_pred-self.f_pde_batch)**2)
        
        
        return [loss_f, loss_ui, loss_ub] 

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
        X_star = (X-self.mu_X)/self.sigma_X
        t = torch.tensor(X_star[:, 0:1], requires_grad=True).float().to(self.device)
        x = torch.tensor(X_star[:, 1:2], requires_grad=True).float().to(self.device)
        
        self.dnn.eval()
        u = self.net_u(t, x)
        f = self.net_f(t, x)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f    