import torch
import numpy as np

class DCGD(torch.optim.Optimizer):

    def __init__(self, optimizer, num_pde, type='center'):
        '''
        num_pde - the number of the PDEs (boundary conditions excluded)
        type - proj, avg, 
        '''
        defaults = dict()
        super().__init__(optimizer.param_groups, defaults)
        self.optimizer = optimizer
        self.num_pde = num_pde
        self.epsilon = 1e-8
        self.iter = 0
        self.type = type
        self.conflict_TH = 1e-8
        self.Isconflict = False

    def _flatten_grad(self, grads):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad   

    def zero_grad(self):
        return self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step_center(self, losses):
        self.iter += 1
        with torch.enable_grad():
            pde_loss = sum(losses[:self.num_pde])
        self.zero_grad()
        pde_loss.backward(retain_graph=True)
        pde_grad, pde_shape, pde_has_grad = [], [], []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    pde_shape.append(p.shape)
                    pde_grad.append(torch.zeros_like(p).to(p.device))
                    pde_has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                pde_shape.append(p.grad.shape)
                pde_grad.append(p.grad.clone())
                pde_has_grad.append(torch.ones_like(p).to(p.device))
        flatten_pde_grad = self._flatten_grad(pde_grad)                

        for i in range(self.num_pde, len(losses)):
            with torch.enable_grad():
                self.zero_grad()
                losses[i].backward(retain_graph=True)
            bc_grad = []
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        bc_grad.append(torch.zeros_like(p).to(p.device))
                        continue
                    bc_grad.append(p.grad.clone())
            flatten_bc_grad = self._flatten_grad(bc_grad)

            if i == self.num_pde:
                flatten_bc_grads =flatten_bc_grad
            else:
                flatten_bc_grads += flatten_bc_grad

        bc_norm = torch.norm(flatten_bc_grads, p=2) + self.epsilon
        pde_norm = torch.norm(flatten_pde_grad, p=2) + self.epsilon

        center_grads = (flatten_bc_grads/bc_norm+flatten_pde_grad/pde_norm)
        bc_pde_dot = flatten_bc_grads.dot(flatten_pde_grad).item()

        cos_val = bc_pde_dot/(bc_norm*pde_norm)

        if cos_val < (-1+self.conflict_TH):
            self.Isconflict = True

        center_norm_sq = (2*(1+bc_pde_dot/(bc_norm*pde_norm)))
        center_total_dot = center_grads.dot(flatten_bc_grads+flatten_pde_grad).item()

        unflatten_pde_grad = self._unflatten_grad(flatten_pde_grad, pde_shape)
        unflatten_bc_grads = self._unflatten_grad(flatten_bc_grads, pde_shape)
        
        with torch.enable_grad():
            self.zero_grad()
            idx = 0
            for group in self.optimizer.param_groups:
                for param in group['params']:
                    param.grad = (center_total_dot/center_norm_sq)*((unflatten_bc_grads[idx]/bc_norm) +unflatten_pde_grad[idx]/pde_norm)
                    idx += 1

        self.optimizer.step()
        return self.Isconflict

    @torch.no_grad()
    def step_avg(self, losses):
        self.iter += 1
        with torch.enable_grad():
            pde_loss = sum(losses[:self.num_pde])
        self.zero_grad()
        pde_loss.backward(retain_graph=True)
        pde_grad, pde_shape, pde_has_grad = [], [], []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    pde_shape.append(p.shape)
                    pde_grad.append(torch.zeros_like(p).to(p.device))
                    pde_has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                pde_shape.append(p.grad.shape)
                pde_grad.append(p.grad.clone())
                pde_has_grad.append(torch.ones_like(p).to(p.device))
        flatten_pde_grad = self._flatten_grad(pde_grad)                

        for i in range(self.num_pde, len(losses)):
            with torch.enable_grad():
                self.zero_grad()
                losses[i].backward(retain_graph=True)
            bc_grad = []
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        bc_grad.append(torch.zeros_like(p).to(p.device))
                        continue
                    bc_grad.append(p.grad.clone())
            flatten_bc_grad = self._flatten_grad(bc_grad)

            if i == self.num_pde:
                flatten_bc_grads =flatten_bc_grad
            else:
                flatten_bc_grads += flatten_bc_grad

        flatten_total_grad = (flatten_bc_grads + flatten_pde_grad)/2

        bc_norm = torch.norm(flatten_bc_grads, p=2) + self.epsilon
        pde_norm = torch.norm(flatten_pde_grad, p=2) + self.epsilon
 
        total_bc_dot = flatten_total_grad.dot(flatten_bc_grads).item()
        total_pde_dot = flatten_total_grad.dot(flatten_pde_grad).item()
        bc_pde_dot = flatten_bc_grads.dot(flatten_pde_grad).item()

        cos_val = bc_pde_dot/(bc_norm*pde_norm)

        if cos_val < (-1+self.conflict_TH):
            self.Isconflict = True

        unflatten_pde_grad = self._unflatten_grad(flatten_pde_grad, pde_shape)
        unflatten_bc_grads = self._unflatten_grad(flatten_bc_grads, pde_shape)

        with torch.enable_grad():
            self.zero_grad()
            DualOut = (total_bc_dot < 0) or (total_pde_dot < 0)
            if DualOut:
                idx = 0
                for group in self.optimizer.param_groups:
                    for param in group['params']:
                        param.grad = ((1-bc_pde_dot/(bc_norm**2))*(unflatten_bc_grads[idx]) +(1-bc_pde_dot/(pde_norm**2))*unflatten_pde_grad[idx])/2
                        idx += 1
            else:
                total_loss = sum(losses)
                total_loss.backward()
   
        self.optimizer.step()
        return self.Isconflict

    @torch.no_grad()
    def step_proj(self, losses):
        self.iter += 1
        with torch.enable_grad():
            pde_loss = sum(losses[:self.num_pde])
        self.zero_grad()
        pde_loss.backward(retain_graph=True)
        pde_grad, pde_shape, pde_has_grad = [], [], []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    pde_shape.append(p.shape)
                    pde_grad.append(torch.zeros_like(p).to(p.device))
                    pde_has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                pde_shape.append(p.grad.shape)
                pde_grad.append(p.grad.clone())
                pde_has_grad.append(torch.ones_like(p).to(p.device))
        flatten_pde_grad = self._flatten_grad(pde_grad)                

        for i in range(self.num_pde, len(losses)):
            with torch.enable_grad():
                self.zero_grad()
                losses[i].backward(retain_graph=True)
            bc_grad = []
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        bc_grad.append(torch.zeros_like(p).to(p.device))
                        continue
                    bc_grad.append(p.grad.clone())
            flatten_bc_grad = self._flatten_grad(bc_grad)

            if i == self.num_pde:
                flatten_bc_grads =flatten_bc_grad
            else:
                flatten_bc_grads += flatten_bc_grad

        flatten_total_grad = (flatten_bc_grads + flatten_pde_grad)/2

        bc_norm = torch.norm(flatten_bc_grads, p=2) + self.epsilon
        pde_norm = torch.norm(flatten_pde_grad, p=2) + self.epsilon
 
        total_bc_dot = flatten_total_grad.dot(flatten_bc_grads).item()
        total_pde_dot = flatten_total_grad.dot(flatten_pde_grad).item()
        bc_pde_dot = flatten_bc_grads.dot(flatten_pde_grad).item()

        cos_val = bc_pde_dot/(bc_norm*pde_norm)

        if cos_val < (-1+self.conflict_TH):
            self.Isconflict = True

        unflatten_pde_grad = self._unflatten_grad(flatten_pde_grad, pde_shape)
        unflatten_bc_grads = self._unflatten_grad(flatten_bc_grads, pde_shape)

        with torch.enable_grad():
            self.zero_grad()
            if total_bc_dot < 0:
                idx = 0
                for group in self.optimizer.param_groups:
                    for param in group['params']:
                        param.grad = (-bc_pde_dot/(bc_norm**2))*(unflatten_bc_grads[idx]) + unflatten_pde_grad[idx]
                        idx += 1
            elif total_pde_dot < 0:
                idx = 0
                for group in self.optimizer.param_groups:
                    for param in group['params']:
                        param.grad = (unflatten_bc_grads[idx]) +(-bc_pde_dot/(pde_norm**2))*unflatten_pde_grad[idx]
                        idx += 1                
            else:
                total_loss = sum(losses)
                total_loss.backward()
   
        self.optimizer.step()
        return self.Isconflict

    @torch.no_grad()
    def step(self, losses):
        if self.type == 'center':
            self.step_center(losses)
        elif self.type == 'avg':
            self.step_avg(losses)
        elif self.type == 'proj':
            self.step_proj(losses)
        else:
            raise NotImplementedError('TBU')
        return 
