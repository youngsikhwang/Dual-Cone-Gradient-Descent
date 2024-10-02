from utils import *

class Klein_Gordon_sampler:
    def __init__(self, a_1=1, a_2=4, k=1):
        self.a_1 = a_1
        self.a_2 = a_2
        self.k = k

        self.ics_coords = np.array([[0.0, 0.0],
                                [0.0, 1.0]])
        self.bc1_coords = np.array([[0.0, 0.0],
                                [1.0, 0.0]])
        self.bc2_coords = np.array([[0.0, 1.0],
                                [1.0, 1.0]])
        self.dom_coords = np.array([[0.0, 0.0],
                                [1.0, 1.0]])

    def u(self,x):
        """
        :param x: x = (t, x)
        """
        return x[:, 1:2] * np.cos(5 * np.pi * x[:, 0:1]) + (x[:, 0:1] * x[:, 1:2])**3

    def u_tt(self,x):
        return - 25 * np.pi**2 * x[:, 1:2] * np.cos(5 * np.pi * x[:, 0:1]) + 6 * x[:,0:1] * x[:,1:2]**3

    def u_xx(self,x):
        return np.zeros((x.shape[0], 1)) +  6 * x[:,1:2] * x[:,0:1]**3

    def f(self,x):
        return self.u_tt(x) -  self.u_xx(x) +  self.u(x)**3

    def samplers(self):
        ics_sampler = Sampler(2, self.ics_coords, lambda x: self.u(x), name='Dirichlet IC1')

        bc1 = Sampler(2, self.bc1_coords, lambda x: self.u(x), name='Dirichlet BC1')
        bc2 = Sampler(2, self.bc2_coords, lambda x: self.u(x), name='Dirichlet BC2')

        bcs_sampler = [bc1, bc2]

        pde_sampler = Sampler(2, self.dom_coords, lambda x: self.f(x), name='Forcing')

        return [ics_sampler, bcs_sampler, pde_sampler]

    def testset(self):
        nn = 100
        t = np.linspace(self.dom_coords[0, 0], self.dom_coords[1, 0], nn)[:, None]
        x = np.linspace(self.dom_coords[0, 1], self.dom_coords[1, 1], nn)[:, None]
        t, x = np.meshgrid(t, x)
        X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))

        # Exact solution
        u_star = self.u(X_star)
        f_star = self.f(X_star)

        return [X_star, u_star]