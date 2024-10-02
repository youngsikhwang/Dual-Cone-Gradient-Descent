from utils import *
import scipy
from pyDOE import lhs

data = scipy.io.loadmat('/home/user_hys/User/hys/PINN/burgers_shock.mat')

class Burgers_sampler:
    def __init__(self, data= data):
        self.data = data
        self.t = data['t'].flatten()[:,None]
        self.x = data['x'].flatten()[:,None]
        self.Exact = np.real(data['usol']).T
        self.X, self.T = np.meshgrid(self.x,self.t)
        self.X_star = np.hstack((self.X.flatten()[:,None], self.T.flatten()[:,None]))
        self.u_star = self.Exact.flatten()[:,None]       
        self.lb = self.X_star.min(0)
        self.ub = self.X_star.max(0)    

    def sample(self, batch_size):
         xx1 = np.hstack((self.X[0:1,:].T, self.T[0:1,:].T))
         uu1 = self.Exact[0:1,:].T
         xx2 = np.hstack((self.X[:,0:1], self.T[:,0:1]))
         uu2 = self.Exact[:,0:1]
         xx3 = np.hstack((self.X[:,-1:], self.T[:,-1:]))
         uu3 = self.Exact[:,-1:]

         idx1 = np.random.choice(xx1.shape[0], batch_size//3, replace=False)
         idx2 = np.random.choice(xx2.shape[0], batch_size//3, replace=False)
         idx3 = np.random.choice(xx3.shape[0], batch_size//3, replace=False)

         xx1 = xx1[idx1, :]
         xx2 = xx2[idx2, :]
         xx3 = xx3[idx3, :]

         uu1 = uu1[idx1, :]
         uu2 = uu2[idx2, :]
         uu3 = uu3[idx3, :]

         X_ui_train = np.vstack([xx1])
         X_ub_train = np.vstack([xx2, xx3])
         X_f_train = self.lb + (self.ub-self.lb)*lhs(2, batch_size*10)

         ui_train = np.vstack([uu1])
         ub_train = np.vstack([uu2, uu3])
 
         return [X_ui_train, ui_train, X_ub_train, ub_train, X_f_train]       


    def testset(self):        
        return [self.X_star, self.u_star]