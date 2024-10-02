from utils import *

class Helmholtz_sampler:
    def __init__(self, a_1=1, a_2=4, k=1):
        self.a_1 = a_1
        self.a_2 = a_2
        self.k = k

        self.bc1_coords = np.array([[-1.0, -1.0],
                           [1.0, -1.0]])
        self.bc2_coords = np.array([[1.0, -1.0],
                                [1.0, 1.0]])
        self.bc3_coords = np.array([[1.0, 1.0],
                                [-1.0, 1.0]])
        self.bc4_coords = np.array([[-1.0, -1.0],
                                [-1.0, 1.0]])

        self.dom_coords = np.array([[-1.0, -1.0],
                                [1.0, 1.0]])

    def u(self, x):
        return np.sin(self.a_1 * np.pi * x[:, 0:1]) * np.sin(self.a_2 * np.pi * x[:, 1:2])

    def u_xx(self, x):
        return - (self.a_1 * np.pi) ** 2 * np.sin(self.a_1 * np.pi * x[:, 0:1]) * np.sin(self.a_2 * np.pi * x[:, 1:2])

    def u_yy(self, x):
        return - (self.a_2 * np.pi) ** 2 * np.sin(self.a_1 * np.pi * x[:, 0:1]) * np.sin(self.a_2 * np.pi * x[:, 1:2])

    def f(self, x):
        return self.u_xx(x) + self.u_yy(x) + (self.k**2) * self.u(x)

    def samplers(self):
        ics_sampler = None

        bc1 = Sampler(2, self.bc1_coords, lambda x: self.u(x), name='Dirichlet BC1')
        bc2 = Sampler(2, self.bc2_coords, lambda x: self.u(x), name='Dirichlet BC2')
        bc3 = Sampler(2, self.bc3_coords, lambda x: self.u(x), name='Dirichlet BC3')
        bc4 = Sampler(2, self.bc4_coords, lambda x: self.u(x), name='Dirichlet BC4')
        bcs_sampler = [bc1, bc2, bc3, bc4]

        pde_sampler = Sampler(2, self.dom_coords, lambda x: self.f(x), name='Forcing')

        return [ics_sampler, bcs_sampler, pde_sampler]

    def testset(self):
        nn = 100
        x1 = np.linspace(self.dom_coords[0, 0], self.dom_coords[1, 0], nn)[:, None]
        x2 = np.linspace(self.dom_coords[0, 1], self.dom_coords[1, 1], nn)[:, None]
        x1, x2 = np.meshgrid(x1, x2)

        X_star = np.hstack((x1.flatten()[:, None], x2.flatten()[:, None]))

        # Exact solution
        u_star = self.u(X_star)
        f_star = self.f(X_star)

        return [X_star, u_star]