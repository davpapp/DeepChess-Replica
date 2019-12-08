import chess.pgn
import chess
import pickle
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader


class Evaluator(nn.Module):
    def __init__(self):
        super(Evaluator, self).__init__()
        self.lin1 = nn.Linear(400, 200)
        self.lin2 = nn.Linear(200, 100)
        self.lin3 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        return x

    # The loss function (which we chose to include as a method of the class, but doesn't need to be)
    # returns the loss and optimizer used by the model
    def get_loss(self, learning_rate):
        # Loss function
        loss = nn.CrossEntropyLoss()
        # Optimizer, self.parameters() returns all the Pytorch operations that are attributes of the class
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        return loss, optimizer

net = Evaluator()


learning_rate = 1e-2
n_epochs = 2

def train_model(net):
    """ Train a the specified network.

        Outputs a tuple with the following four elements
        train_hist_x: the x-values (batch number) that the training set was 
            evaluated on.
        train_loss_hist: the loss values for the training set corresponding to
            the batch numbers returned in train_hist_x
        test_hist_x: the x-values (batch number) that the test set was 
            evaluated on.
        test_loss_hist: the loss values for the test set corresponding to
            the batch numbers returned in test_hist_x
    """ 
    loss, optimizer = net.get_loss(learning_rate)
    # Define some parameters to keep track of metrics
    print_every = 50
    idx = 0
    print("started training...")
    train_hist_x = []
    train_loss_hist = []
    test_hist_x = []
    test_loss_hist = []

    training_start_time = time.time()
    # Loop for n_epochs
    for epoch in range(n_epochs):
        print("epoch %(e)d" % {"e":epoch})
        running_loss = 0.0
        start_time = time.time()

        for i, data in enumerate(train_loader, 0):
            #print(i, data)

            # Get inputs in right form
            # print(data)
            #print(data['image']) #rint(data['inputs'])
            inputs, labels = data['board'], data['outcome']
            #print(inputs)
            #print(labels)
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)
          
            # In Pytorch, We need to always remember to set the optimizer gradients to 0 before we recompute the new gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)
          
            print("forward passed")
            # Compute the loss and find the loss with respect to each parameter of the model
            loss_size = loss(outputs, labels)
            loss_size.backward()
            print("calculate loss")
            # Change each parameter with respect to the recently computed loss.
            optimizer.step()

            # Update statistics
            running_loss += loss_size.data.item()
            print("updated loss")
            # Print every 20th batch of an epoch
            if (i % print_every) == print_every-1:
                print("Epoch {}, Iteration {}\t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch + 1, i+1,running_loss / print_every, time.time() - start_time))
                # Reset running loss and time
                train_loss_hist.append(running_loss / print_every)
                train_hist_x.append(idx)
                running_loss = 0.0
                start_time = time.time()
                idx += 1

            # At the end of the epoch, do a pass on the test set
            total_test_loss = 0
            print("test set pass")
            for i, data in enumerate(test_loader, 0):
                inputs, labels = data['board'], data['outcome']

                # Wrap tensors in Variables
                inputs, labels = Variable(inputs).to(device, dtype=torch.float), Variable(labels).to(device, dtype=torch.long)

                # Forward pass
                test_outputs = net(inputs)
                test_loss_size = loss(test_outputs, labels)
                total_test_loss += test_loss_size.data.item()
            
            test_loss_hist.append(total_test_loss / len(test_loader))
            test_hist_x.append(idx)
            print("Validation loss = {:.2f}".format(
                total_test_loss / len(test_loader)))

        print("Training finished, took {:.2f}s".format(
        time.time() - training_start_time))
        return train_hist_x, train_loss_hist, test_hist_x, test_loss_hist