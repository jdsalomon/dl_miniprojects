import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

from dlc_practical_prologue import *

from Sub_LeNet5 import *
from Sub_LeNet5_WS import *
from Sub_LeNet5_WS_auxLoss import *

DEBUG_PROMPT = True

BATCH_SIZE = 100
LEARNING_RATE = 0.01
MOMENTUM = 0.9

n = 20
nb_models = 6 #3 models each with two options: adaptive lr or not
results = torch.zeros(2 * nb_models, n)
loss_function1 = nn.BCELoss()
loss_function2 = F.nll_loss

#Optimal learning rates for each model obtained using grid search
LR_SLN5 = 0.01
LR_SLN5_WS = 0.0001
LR_SLN5_WS_AL = 0.01

#Optimal batch sizes for each model obtained using grid search
BS_SLN5 = 250
BS_SLN5_WS = 100
BS_SLN5_WS_AL = 50

results = torch.zeros(2 * nb_models, n)

VERBOSE = False
for e in range(n):
    print(f"Run number {e}")
    
    torch.manual_seed(e)
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)

    network_compare = Sub_LeNet5()
    optimizer = optim.Adam(network_compare.parameters(),lr=LR_SLN5)
    train_acc, train_losses, test_acc, test_losses = network_compare.fit(train_input,train_target,\
                                                                         test_input,test_target,loss_function1,\
                                                                         optimizer, batch_size=BS_SLN5, adaptive_lr=False, verbose=VERBOSE)

    results[[0, 1], e] = torch.tensor([train_acc[network_compare.best_epoch], test_acc[network_compare.best_epoch]])

    network_compare_adapt_lr = Sub_LeNet5()
    optimizer = optim.Adam(network_compare_adapt_lr.parameters(),lr=LR_SLN5)
    train_acc, train_losses, test_acc, test_losses = network_compare_adapt_lr.fit(train_input,train_target,\
                                                                         test_input,test_target,loss_function1,\
                                                                         optimizer, batch_size=BS_SLN5,\
                                                                                  adaptive_lr=True, verbose=VERBOSE)

    results[[2, 3], e] = torch.tensor([train_acc[network_compare_adapt_lr.best_epoch], test_acc[network_compare_adapt_lr.best_epoch]])

    network_compare_ws = Sub_LeNet5_WS()
    optimizer = optim.Adam(network_compare_ws.parameters(),lr=LR_SLN5_WS)
    train_acc, train_losses, test_acc, test_losses = network_compare_ws.fit(train_input,train_target,\
                                                                         test_input,test_target,loss_function1,\
                                                                         optimizer, batch_size=BS_SLN5_WS,\
                                                                            adaptive_lr=False, verbose=VERBOSE)

    results[[4, 5], e] = torch.tensor([train_acc[network_compare_ws.best_epoch], test_acc[network_compare_ws.best_epoch]])

    network_compare_ws_adapt_lr = Sub_LeNet5_WS()
    optimizer = optim.Adam(network_compare_ws_adapt_lr.parameters(),lr=LR_SLN5_WS)
    train_acc, train_losses, test_acc, test_losses = network_compare_ws_adapt_lr.fit(train_input,train_target,\
                                                                         test_input,test_target,loss_function1,\
                                                                         optimizer, batch_size=BS_SLN5_WS,\
                                                                                     adaptive_lr=True, verbose=VERBOSE)

    results[[6, 7], e] = torch.tensor([train_acc[network_compare_ws_adapt_lr.best_epoch], test_acc[network_compare_ws_adapt_lr.best_epoch]])

    network_compare_ws_al = Sub_LeNet5_WS_auxLoss()
    optimizer = optim.Adam(network_compare_ws_al.parameters(),lr=LR_SLN5_WS_AL)
    train_acc, train_losses, test_acc, test_losses = network_compare_ws_al.fit(train_input, train_target,
                                                                             train_classes,
                                                                             test_input, 
                                                                             test_target, 
                                                                             test_classes, 
                                                                             loss_function2,\
                                                                             optimizer, batch_size=BS_SLN5_WS_AL,\
                                                                             adaptive_lr=False, verbose=VERBOSE)

    results[[8, 9], e] = torch.tensor([train_acc[network_compare_ws_al.best_epoch], test_acc[network_compare_ws_al.best_epoch]])

    network_compare_ws_al_adapt_lr = Sub_LeNet5_WS_auxLoss()
    optimizer = optim.Adam(network_compare_ws_al_adapt_lr.parameters(),lr=LR_SLN5_WS_AL)
    train_acc, train_losses, test_acc, test_losses = network_compare_ws_al_adapt_lr.fit(train_input, train_target,
                                                                             train_classes,
                                                                             test_input, 
                                                                             test_target, 
                                                                             test_classes, 
                                                                             loss_function2,\
                                                                             optimizer, batch_size=BS_SLN5_WS_AL,\
                                                                             adaptive_lr=True, verbose=VERBOSE)

    results[[10, 11], e] = torch.tensor([train_acc[network_compare_ws_al_adapt_lr.best_epoch], test_acc[network_compare_ws_al_adapt_lr.best_epoch]])

mean = results.mean(1)
std = results.std(1)
print("Mean accuracy from 20 runs:")
print("Sub_LeNet5, lr={0} : Train accuracy = {1} (std = {2}), Test accuracy = {3} (std = {4})".format(BS_SLN5, mean[0], std[0], mean[1], std[1]))
print("Sub_LeNet5 with adaptive lr, base_lr={0} : Train accuracy = {1} (std = {2}), Test accuracy = {3} (std = {4})".format(BS_SLN5, mean[2], std[2], mean[3], std[3]))
print("Sub_LeNet5_WS, lr={0} : Train accuracy = {1} (std = {2}), Test accuracy = {3} (std = {4})".format(BS_SLN5_WS, mean[4], std[4], mean[5], std[5]))
print("Sub_LeNet5_WS with adaptive lr, base_lr={0} : Train accuracy = {1} (std = {2}), Test accuracy = {3} (std = {4})".format(BS_SLN5_WS, mean[6], std[6], mean[7], std[7]))
print("Sub_LeNet5_WS_auxLoss, lr={0} : Train accuracy = {1} (std = {2}), Test accuracy = {3} (std = {4})".format(BS_SLN5_WS_AL, mean[8], std[8], mean[9], std[9]))
print("Sub_LeNet5_WS_auxLoss with adaptive lr, base_lr={0} : Train accuracy = {1} (std = {2}), Test accuracy = {3} (std = {4})".format(BS_SLN5_WS_AL, mean[10], std[10], mean[11], std[11]))
