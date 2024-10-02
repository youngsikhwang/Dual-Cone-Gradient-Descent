import torch

import numpy as np

import time
import math
import argparse
import os

from utils import *
from DataSampler.helmholtz_sampler import Helmholtz_sampler
from DataSampler.klein_gordon_sampler import Klein_Gordon_sampler
from DataSampler.burgers_sampler import Burgers_sampler
from model.Helmholtz import PINN_Helmholtz
from model.Klein_Gordon import PINN_Klein_Gordon
from model.Burgers import PINN_Burgers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PINN')
    parser.add_argument('--equation', type=str, default="helmholtz", help="Select PDE: helmholtz, burgers, klein-Gordon")
    parser.add_argument('--device_n', type=str, default="0")  # set to "cpu" enables cpu training 
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--width', type=int, default=50)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--iter', type=int, default=50000)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--plot', type=bool, default=True, help="Plot the absolute error")
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--optim', type=str, default="adam", help="Select base optimizer: ")
    parser.add_argument('--dcgd', type=str, default="center", help="Select DCGD type: center, avg, proj")
    parser.add_argument('--betas', type=tuple, default=(0.9,0.999))

    command_args = parser.parse_args()

    Equation = command_args.equation
    seed = command_args.seed
    layers =  gen_layer(2,command_args.depth, command_args.width) 
    batch_size = command_args.batch
    lr = command_args.lr
    optim = command_args.optim
    betas = command_args.betas
    device_n = command_args.device_n
    dcgd = command_args.dcgd
    result = []

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]= device_n 

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    np.random.seed(seed)
    torch.manual_seed(seed)

    for i in range(command_args.repeat):
        if Equation == 'helmholtz':
            H_samplers = Helmholtz_sampler()
            samplers = H_samplers.samplers()
            test_data = H_samplers.testset()
            model = PINN_Helmholtz(samplers, test_data, layers, batch_size, lr, betas, optim, dcgd, device)
        elif Equation == 'klein_gordon':
            K_samplers = Klein_Gordon_sampler()
            samplers = K_samplers.samplers()
            test_data = K_samplers.testset()
            model = PINN_Klein_Gordon(samplers, test_data, layers, batch_size, lr, betas, optim, dcgd, device)
        elif Equation == 'burgers':
            B_samplers = Burgers_sampler()
            test_data = B_samplers.testset()
            model = PINN_Burgers(B_samplers, test_data, layers, batch_size, lr, betas, optim, dcgd, device)

        model.train(command_args.iter)
        result.append(model.best_test_error)
    result = np.array(result)

    output_file = f'./results/{Equation}_results/{Equation}_{layers}_{optim}_{lr}_{batch_size}.txt'

    with open(output_file, 'w') as file:
        file.write(f'Equation: {Equation} \n')
        file.write(f'Mean of L2 error: {np.mean(result)} \n')
        file.write(f'Std of L2 error: {np.std(result)} \n')
        file.write(f'Max of L2 error: {np.max(result)} \n')
        file.write(f'Min of L2 error: {np.min(result)} \n')
  
    model.errorplot()

    print(f'Results saved to {output_file}')