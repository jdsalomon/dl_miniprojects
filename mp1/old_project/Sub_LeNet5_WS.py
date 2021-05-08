import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

N_EPOCHS = 25

#Default values
BATCH_SIZE = 100
LEARNING_RATE = 0.01
MOMENTUM = 0.9

class Sub_LeNet5_WS(nn.Module):
    def __init__(self, lr=LEARNING_RATE):
        super(Sub_LeNet5_WS, self).__init__()
        
        #Image recognition functions - Similar to previous model, but the same functions are used for the images to apply weight sharing on those functions.
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1)
        self.average = nn.AvgPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
                
        self.fc1 = nn.Linear(120, 82)
        self.fc2 = nn.Linear(82, 10)
        
        #Digit Comparison layer
        self.fc3 = nn.Linear(20,100)
        self.fc4 = nn.Linear(100,100)
        self.fc5 = nn.Linear(100,1)
        
        #For adaptive lr
        self.prev_loss = -1
        self.lr_min = lr / 100
        
        #To find the best model (to avoid overfitting)
        self.best_test_acc = -1
        self.best_epoch = -1
        
    def forward(self, im1, im2):   
        im1 = im1.reshape(-1, 1, 14, 14)
        out1 = torch.tanh(self.conv1(im1))
        out1 = self.average(out1)
        out1 = torch.tanh(self.conv2(out1))
        out1 = out1.view(-1, out1.shape[1])
        out1 = torch.relu(self.fc1(out1))
        out1 = torch.log_softmax(self.fc2(out1), dim=1)
        
        im2 = im2.reshape(-1, 1, 14, 14)
        out2 = torch.tanh(self.conv1(im2))
        out2 = self.average(out2)
        out2 = torch.tanh(self.conv2(out2))
        out2 = out2.view(-1, out2.shape[1])
        out2 = torch.relu(self.fc1(out2))
        out2 = torch.log_softmax(self.fc2(out2), dim=1)
     
        pair = torch.cat((out1,out2),1)
        pair = torch.relu(self.fc3(pair))
        pair = torch.relu(self.fc4(pair))
        comparison = torch.sigmoid(self.fc5(pair))
        return comparison
    
    def update_lr(self, new_loss, old_loss, optimizer, threshold=0.03, verbose=False):
        for param_group in optimizer.param_groups:
            diff = old_loss - new_loss
            if diff/old_loss <= threshold and param_group['lr'] > self.lr_min:
                param_group['lr'] = param_group['lr'] / 2

                if verbose:
                    print("Lower the rate. New rate :", param_group['lr'])
        
    def fit(self, train, train_target, test, test_target, loss_function, optimizer, batch_size=BATCH_SIZE, adaptive_lr=True, suffix="", verbose=False):
        train_target = train_target.float()
        train_acc, train_losses, test_acc, test_losses = [], [], [], []
        
        for epoch in range(N_EPOCHS):
            sum_loss, correct, total = 0, 0, 0
            permutation = torch.randperm(train.size(0))
            
            for batch_start in range(0, train.size(0), batch_size):
                # Get the batch
                indices = permutation[batch_start:batch_start+batch_size]
                batch_train, batch_target = train[indices], train_target[indices]
                batch_train1, batch_train2 = batch_train[:, 0], batch_train[:, 1]

                # Forward pass                
                output = self(batch_train1, batch_train2).view(-1)
                
                # Metrics
                loss = loss_function(output, batch_target)
                sum_loss += loss.item()
                
                pred = output.round()
                total += batch_target.size(0)
                correct += (batch_target == pred).sum().item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if adaptive_lr:
                self.update_lr(sum_loss, self.prev_loss, optimizer)
                self.prev_loss = sum_loss
                
            # Test the model
            test_pred, acc_test, test_loss = self.predict(test, test_target, loss_function)
            if acc_test > self.best_test_acc:
                self.best_test_acc = acc_test
                self.best_epoch = epoch
                #torch.save(best_model.state_dict(), f'Project 1/data/Sub_LeNet5_WS/best_sub_LN5_ws{suffix}.pt')
                
            # Save the accuracy and loss
            train_acc.append(correct/total)
            train_losses.append(sum_loss)
            test_acc.append(acc_test)
            test_losses.append(test_loss)
            
            if verbose:
                for param_group in optimizer.param_groups:
                    print(f'Epoch: {epoch}', 
                          'Train aux loss: {:.5f}'.format(train_losses[-1]), 
                          f'Train Accuracy: {train_acc[-1]}', 
                          f'Test Accuracy: {test_acc[-1]}',
                          f'Learning Rate: {param_group["lr"]}',
                          sep='\t')
        if verbose:
            print(f'Best Model was at epoch {self.best_epoch} and has test accuracy of {self.best_test_acc}')
        
    
        return train_acc, train_losses, test_acc, test_losses
                        
    def predict(self, test, target, loss_function):
        target = target.float()
        with torch.no_grad():
            test1, test2 = test[:, 0], test[:, 1]
            
            output = self(test1, test2).view(-1)
            loss = loss_function(output, target)
            pred = output.round()
            correct = (target == pred).sum().item()
            acc = correct / test.size(0)
            return pred, acc, loss