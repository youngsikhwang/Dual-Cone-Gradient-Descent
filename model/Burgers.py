import torch
import numpy as np
from tqdm import tqdm
from Network import DNN
from dcgd import DCGD

class PINN_Burgers():
    def __init__(self, samplers, test_data, layers, batch_size, lr, optim, dcgd, device):

        self.nu = 0.01/np.pi
        self.sampler = samplers

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
    

    def net_u(self, x, t):  
        u = self.dnn(torch.cat([x, t], dim=1))
        return u
    
    def net_f(self, x, t):
        """ The pytorch autograd version of calculating residual """
        u = self.net_u(x, t)
        
        u_t = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        f = u_t + u * u_x - self.nu * u_xx
        return f
    
    def fetch_minibatch(self, sampler, batch_size):
        X_ui_train, ui_train, X_ub_train, ub_train, X_f_train = sampler.sample(batch_size)

        X_ui_train = torch.tensor(X_ui_train, requires_grad=True).float().to(self.device)
        X_ub_train = torch.tensor(X_ub_train, requires_grad=True).float().to(self.device)

        X_f_train = torch.tensor(X_f_train, requires_grad=True).float().to(self.device)
        #u_train = torch.tensor(u_train, requires_grad=True).float().to(device)
        ui_train = torch.tensor(ui_train, requires_grad=True).float().to(self.device)
        ub_train = torch.tensor(ub_train, requires_grad=True).float().to(self.device)        
        return X_ui_train, ui_train, X_ub_train, ub_train, X_f_train
    
    def loss_func(self):
        X_ui, ui, X_ub, ub, X_f = self.fetch_minibatch(self.sampler, self.batch_size)
        x_ui = X_ui[:, 0:1]
        t_ui = X_ui[:, 1:2]

        x_ub = X_ub[:, 0:1]
        t_ub = X_ub[:, 1:2]

        x_f = X_f[:, 0:1]
        t_f = X_f[:, 1:2]
        ui_pred = self.net_u(x_ui, t_ui)
        ub_pred = self.net_u(x_ub, t_ub)

        f_pred = self.net_f(x_f, t_f)
        loss_ui = torch.mean((ui - ui_pred) ** 2)
        loss_ub = torch.mean((ub - ub_pred) ** 2)

        loss_f = torch.mean(f_pred ** 2)
        
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
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)

        self.dnn.eval()
        u = self.net_u(x, t)
        f = self.net_f(x, t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f