#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' git clone https://github.com/UCMerced-ML/LC-model-compression')
get_ipython().system(' pip3 install -e ./LC-model-compression')


# Restart the runtime after running the above cell.

# In[1]:


import lc
from lc.torch import ParameterTorch as Param, AsVector, AsIs
from lc.compression_types import ConstraintL0Pruning, LowRank, RankSelection, AdaptiveQuantization
from lc.models.torch import lenet300_classic, lenet300_modern_drop, lenet300_modern

import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import datasets
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
torch.set_num_threads(4)
device = torch.device('cuda') 


# # Reference Network

# ### Network Definition and Training Function

# In[2]:


# 4 layer NN: 1st and 2nd layers are convolutional with 
# 20 5x5 and 50 5x5 filters respectively and 
# having 2x2 max pooling after each, 
# 3rd and 4th layers are fully connected: 500 and 5 neurons respectively.

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 20, kernel_size = 5)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 20, out_channels = 50, kernel_size = 5)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 5)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def train_net(net, parameters, final_model=False):
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []

    max_val_acc = 0
    epochs_per_early_stop_check = parameters["epochs_per_early_stop_check"]
    early_stop_thresh = 1e-5
    intitial_early_stop_patience = 3
    early_stop_patience = intitial_early_stop_patience

    train_loader, _, _ = data_loader(parameters["batch_size"])
    params = list(filter(lambda p: p.requires_grad, net.parameters()))
    optimizer = optim.SGD(params, 
                          lr=parameters["lr"], 
                          momentum=parameters["momentum"], 
                          weight_decay=parameters["weight_decay"],
                          nesterov = parameters["nesterov"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=parameters["step_size"], gamma=parameters["gamma"])
    max_epochs=100
    for epoch in range(max_epochs):
        avg_loss = []
        for i, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.cuda()
            target = target.cuda().to(dtype=torch.long)
            out = net(x)
            loss = net.loss(out, target)
            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            # -------------------------------------------------------------------------------
            if (final_model):
                acc_train, loss_train, acc_val, loss_val = train_val_acc_eval_f(net.eval(), tuning=(not final_model))
                train_accs.append(acc_train)
                val_accs.append(acc_val)
                train_losses.append(loss_train)
                val_losses.append(loss_val)
            # ------------------------------------------------------------------------------- 
        scheduler.step()

        print(f"\tepoch #{epoch} is finished.")
        print(f"\t  avg. train loss: {np.mean(avg_loss):.6f}")
        ## Note: when preparing final report, this chunk should be put in the inner loop
        ## to record errors for each SGD step, rather than just for each epoch (visualize
        ## errs function should also be modified accordingly)
        ## During hyperparameter tuning, it will be sufficient to find error rates only
        ## on each epoch
        # ------------------------------------------------------------------------------- 
        if (not final_model):
            acc_train, loss_train, acc_val, loss_val = train_val_acc_eval_f(net.eval(), tuning=(not final_model))
            train_accs.append(acc_train)
            val_accs.append(acc_val)
        print(f"\t#Train err: {100-acc_train*100:.2f}%, train loss: {loss_train}")
        print(f"\t#Validation err: {100-acc_val*100:.5f}%, validation loss: {loss_val}\n")
        # ------------------------------------------------------------------------------- 
        if (epoch % epochs_per_early_stop_check == epochs_per_early_stop_check - 1):
            if (max_val_acc + early_stop_thresh < acc_val):
                max_val_acc = acc_val
                early_stop_patience = intitial_early_stop_patience
            else:
                early_stop_patience -= 1
            if (early_stop_patience == 0):
                break;
    
    total_steps = len(val_accs)
    accs = np.zeros((2, total_steps), dtype=float)
    losses = None
    if final_model:
        losses = np.zeros((2, total_steps), dtype=float)
        for i in range(total_steps):
            losses[0, i] = train_losses[i]
            losses[1, i] = val_losses[i]
    for i in range(total_steps):
        accs[0, i] = train_accs[i]
        accs[1, i] = val_accs[i]
    visualize_accs(accs, losses, final_model)
    print(accs)
    print(losses)
    print("#" + str(parameters))
    print(f"\t#Train err: {100-acc_train*100:.2f}%, train loss: {loss_train}")
    print(f"\t#Validation err: {100-acc_val*100:.5f}%, validation loss: {loss_val}\n")


# ### Dataloader

# In[3]:


def data_loader(batch_size=2048, n_workers=2, tuning=False):
    train_data_th = datasets.MNIST(root='./datasets', download=True, train=True)
    test_data_th = datasets.MNIST(root='./datasets', download=True, train=False)

    # Get subset of digits that we will be using
    indices = (train_data_th.targets == 0) | (train_data_th.targets == 2) | (train_data_th.targets == 5) | (train_data_th.targets == 6) | (train_data_th.targets == 7)
    train_data, train_targets = train_data_th.data[indices], train_data_th.targets[indices]

    indices = (test_data_th.targets == 0) | (test_data_th.targets == 2) | (test_data_th.targets == 5) | (test_data_th.targets == 6) | (test_data_th.targets == 7)
    test_data, test_targets = test_data_th.data[indices], test_data_th.targets[indices]

    # Change labels to be in range 0 - C-1 so cross entropy function works
    for i, digit in enumerate([0,2,5,6,7]):
        train_targets = torch.where(train_targets == digit, i, train_targets)
        test_targets = torch.where(test_targets == digit, i, test_targets)

    data_train = np.array(train_data[:]).reshape([-1, 1, 28, 28]).astype(np.float32)
    data_test = np.array(test_data[:]).reshape([-1, 1, 28, 28]).astype(np.float32)

    data_train = (data_train / 255)
    dtrain_mean = data_train.mean(axis=0)
    data_train -= dtrain_mean
    data_test = (data_test / 255).astype(np.float32)
    data_test -= dtrain_mean

    train_data = TensorDataset(torch.from_numpy(data_train), train_targets)

    # Create validation set
    val_split = int(0.3 * len(train_data))
    train_data, val_data = random_split(train_data, [len(train_data) - val_split, val_split], generator=torch.Generator().manual_seed(1778))

    # Take subset of full train and validation sets for hyperparameter tuning
    if (tuning):
        subset_proportion = 0.4
        train_subset_size = int(len(train_data) * subset_proportion)
        val_subset_size = int(len(val_data) * subset_proportion)
        train_data, _ = random_split(train_data, [train_subset_size, len(train_data) - train_subset_size], generator=torch.Generator().manual_seed(1778))
        val_data, _ = random_split(val_data, [val_subset_size, len(val_data) - val_subset_size], generator=torch.Generator().manual_seed(1778))

    test_data = TensorDataset(torch.from_numpy(data_test), test_targets)

    train_loader = DataLoader(train_data, num_workers=n_workers, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, num_workers=n_workers, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, num_workers=n_workers, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ### Utility Functions

# In[71]:


def compute_acc_loss(forward_func, data_loader):
    correct_cnt, ave_loss = 0, 0
    for batch_idx, (x, target) in enumerate(data_loader):
        with torch.no_grad():
            target = target.cuda()
            score, loss = forward_func(x.cuda(), target)
            _, pred_label = torch.max(score.data, 1)
            correct_cnt += (pred_label == target.data).sum().item()
            ave_loss += loss.data.item() * len(x)
    accuracy = correct_cnt * 1.0 / len(data_loader.dataset)
    ave_loss /= len(data_loader.dataset)
    return accuracy, ave_loss

def compute_compression_ratio(lc_alg):
    compressed_model_bits = lc_alg.count_param_bits() + (20 + 50 + 500 + 5)*32
    # For finding # of parameters of convolutional layers
    #  https://towardsdatascience.com/understanding-and-calculating-the-number-of-parameters-in-convolution-neural-networks-cnns-fc88790d530d
    #self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 20, kernel_size = 5)
    #self.conv2 = nn.Conv2d(in_channels = 20, out_channels = 50, kernel_size = 5)
    #self.fc1 = nn.Linear(64, 500)
    #self.fc2 = nn.Linear(500, 5)
    #                           Convolutional layers                Linear layers
    uncompressed_model_bits = (((1*5*5 + 1)*20 + (20*5*5 + 1)*50) + (4*4*50*500 + 500*5 + 500+5))*32
    compression_ratio = uncompressed_model_bits/compressed_model_bits
    return compression_ratio

def train_test_acc_eval_f(net):
    train_loader, _, test_loader = data_loader()
    def forward_func(x, target):
        y = net(x)
        return y, net.loss(y, target)
    with torch.no_grad():
        acc_train, loss_train = compute_acc_loss(forward_func, train_loader)
        acc_test, loss_test = compute_acc_loss(forward_func, test_loader)

    print(f"Train err: {100-acc_train*100:.2f}%, train loss: {loss_train}")
    print(f"TEST ERR: {100-acc_test*100:.2f}%, test loss: {loss_test}")

def test_acc_eval_f(net):
    _, _, test_loader = data_loader()
    def forward_func(x, target):
        y = net(x)
        return y, net.loss(y, target)
    with torch.no_grad():
        acc_test, _ = compute_acc_loss(forward_func, test_loader)

    return acc_test
  
def train_val_acc_eval_f(net, tuning):
    train_loader, val_loader, _ = data_loader()
    def forward_func(x, target):
        y = net(x)
        return y, net.loss(y, target)
    with torch.no_grad():
        acc_train, loss_train = compute_acc_loss(forward_func, train_loader)
        acc_val, loss_val = compute_acc_loss(forward_func, val_loader)

    return acc_train, loss_train, acc_val, loss_val

# Visualizes training and validation accuracy on one plot or test error, depending on if it's the final model
def visualize_accs(accs, losses, final_model):
    epochs = np.arange(len(accs[0]))
    fig = plt.figure()
    ax = plt.gca()
    if (not final_model):
        ax.plot(epochs, accs[0], "b-", label="Train")
        ax.plot(epochs, accs[1], "g-", label="Validation")
        ax.set_xlabel('Epoch')
        ax.set_title('Accuracy per Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
    else:
        ax.plot(epochs, accs[0], "b-", label="Train")
        ax.plot(epochs, accs[1], "g-", label="Validation")
        ax.set_xlabel('SGD Step')
        ax.set_title('Accuracy per SGD Step')
        ax.set_ylabel('Error (%)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        fig2 = plt.figure()
        ax2 = plt.gca()
        ax2.plot(epochs, losses[0], "b-", label="Train")
        ax2.plot(epochs, losses[1], "g-", label="Validation")
        ax2.set_xlabel('SGD Step')
        ax2.set_title('Loss per SGD Step')
        ax2.set_ylabel('Loss')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend()

def visualize_params(test_error, num_params):
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(num_params, test_error)
    ax.set_xlabel('Number of Compressed Parameters')
    ax.set_title('Number of Compressed Parameters vs. Test Error')
    ax.set_ylabel('Test Error (%)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([0, 10])

def visualize_ratios(test_error, ratios):
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(ratios, test_error)
    ax.set_xlabel('Compression Ratio')
    ax.set_title('Compression Ratio  vs. Test Error')
    ax.set_ylabel('Test Error (%)')
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_ylim([0, 10])

# This should only be used when reporting the final model - hard-coded to evaluate on test set
def report_confumat(net):
    net.cuda()

    _, _, test_loader = data_loader(batch_size=10000, n_workers=0)
    test_set, test_labels = next(iter(test_loader))

    out = net(test_set.to(torch.device('cuda')))
    _, preds = out.max(1)

    labels = [0, 2, 5, 6, 7]

    conf_matrix = confusion_matrix(y_true=test_labels.numpy(), y_pred=preds.cpu().detach().numpy())

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    ax.set_xticklabels(['']+labels)
    ax.set_yticklabels(['']+labels)
    plt.show()


# ## Tuning Hyperparameters

# In[ ]:


parameters = {
    # Number of sample images used in each SGD step
    "batch_size": 2**4, 

    # Magnitude of SGD step (should decrease as batch size decreases, and vice versa)
    "lr": .005, 

    # Learning rate decrease schedule - learning rate will change as a factor of gamma after each epoch
    #     Note: for very small batch sizes, this happens relatively infrequently
    "gamma":.95, 

    # Learning rate decrease schdule - lr changes by a factor of gamma every step_size epochs
    "step_size": 1, 

    # Magnitude of previous SGD step's update vector to add to the current SGD step's update vector
    "momentum": 0.9, 

    # Magnitude of L2 weight penalty to add to model's loss function - 0 for no l2 regularization
    "weight_decay": 0, 

    # Not too sure, does something similar to momentum, used to hopefully make the training converge faster
    "nesterov": True, 

    # Early stopping paramter - every epochs_per_early_stop_check, an early stopping condition will be checked
    # If three such checks are made in a row indication that the model is no longer improving, training stops
    #     Note: for models with larger batch sizes, this should probably be relatively high (5-10 maybe)
    #     It can be arbitrarily high and training will always stop at 100 epochs
    #     No impact on actual model training, only impacts duration of training
    "epochs_per_early_stop_check": 5
}
#{'batch_size': 16, 'lr': 0.005, 'gamma': 0.95, 'step_size': 1, 'momentum': 0.9, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
parameters = {'batch_size': 16, 'lr': 0.005, 'gamma': 0.95, 'step_size': 1, 'momentum': 0.9, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}


# In[ ]:


net = Net().to(device)
train_net(net, parameters, final_model=False)


# In[ ]:


#{"batch_size": 2048, "lr": .1, "gamma":.9, "step_size": 1, "momentum": 0.5, "weight_decay": 0, "nesterov": True, "epochs_per_early_stop_check": 10}
  #Train err: 3.17%, train loss: 0.10264849333569061
  #Validation err: 2.99604%, validation loss: 0.10082331210913936

#{'batch_size': 2048, 'lr': 0.1, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 10}
  #Train err: 0.55%, train loss: 0.018500898148893385
  #Validation err: 1.30017%, validation loss: 0.04058479166017406

#{'batch_size': 2048, 'lr': 0.1, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0.1, 'nesterov': True, 'epochs_per_early_stop_check': 10}
  #Train err: 3.69%, train loss: 0.19727211416691773
  #Validation err: 3.27869%, validation loss: 0.19464021327301498

#{'batch_size': 2048, 'lr': 0.1, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 10}
	#Train err: 1.04%, train loss: 0.04033823409514834
	#Validation err: 1.30017%, validation loss: 0.05110103836827456

#{'batch_size': 256, 'lr': 0.05, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 10}
  #Train err: 0.61%, train loss: 0.030782110267138296
  #Validation err: 0.93273%, validation loss: 0.03826093443534077

#{'batch_size': 256, 'lr': 0.01, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 10}
	#Train err: 1.17%, train loss: 0.043716297954205396
	#Validation err: 1.15885%, validation loss: 0.047841235978445674

#{'batch_size': 256, 'lr': 0.01, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 10}
	#Train err: 1.17%, train loss: 0.043716297954205396
	#Validation err: 1.15885%, validation loss: 0.047841235978445674

#{'batch_size': 256, 'lr': 0.04, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 10}
  #Train err: 0.67%, train loss: 0.03125638201428476
  #Validation err: 0.87620%, validation loss: 0.03796432829167696

#{'batch_size': 256, 'lr': 0.0375, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 8}
	#Train err: 0.67%, train loss: 0.031911701137243315
	#Validation err: 0.87620%, validation loss: 0.039300043344362906

#{'batch_size': 256, 'lr': 0.04, 'gamma': 0.5, 'step_size': 5, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 10}
  #Train err: 0.65%, train loss: 0.030985873048157656
  #Validation err: 0.84794%, validation loss: 0.0375866007634716

#{'batch_size': 256, 'lr': 0.04, 'gamma': 0.5, 'step_size': 4, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 10}
	#Train err: 0.69%, train loss: 0.0324429041424463
	#Validation err: 0.96099%, validation loss: 0.03873025083471113

#{'batch_size': 256, 'lr': 0.04, 'gamma': 0.65, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 10}
	#Train err: 1.17%, train loss: 0.0424277632562227
	#Validation err: 1.35670%, validation loss: 0.04859118297729807

#{'batch_size': 256, 'lr': 0.04, 'gamma': 0.75, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 10}
	#Train err: 1.01%, train loss: 0.0379838088686152
	#Validation err: 1.24364%, validation loss: 0.04493645774890232

#{'batch_size': 256, 'lr': 0.035, 'gamma': 0.75, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 10}
	#Train err: 1.01%, train loss: 0.038521571092076545
	#Validation err: 0.96099%, validation loss: 0.043056812042601034

#{'batch_size': 256, 'lr': 0.035, 'gamma': 0.65, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 8}
	#Train err: 1.21%, train loss: 0.045564366692258405
	#Validation err: 1.41323%, validation loss: 0.05283862946859317

#{'batch_size': 256, 'lr': 0.04, 'gamma': 0.85, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 8}
	#Train err: 0.79%, train loss: 0.033932102367628454
	#Validation err: 0.98926%, validation loss: 0.040318960152301096

#{'batch_size': 256, 'lr': 0.04, 'gamma': 0.4, 'step_size': 5, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 8}
	#Train err: 0.68%, train loss: 0.03218289652420569
	#Validation err: 0.87620%, validation loss: 0.0384966552788218

#{'batch_size': 256, 'lr': 0.04, 'gamma': 0.4, 'step_size': 4, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 8}
	#Train err: 0.73%, train loss: 0.033358698349821476
	#Validation err: 0.98926%, validation loss: 0.040087291908725765

#{'batch_size': 256, 'lr': 0.04, 'gamma': 0.4, 'step_size': 4, 'momentum': 0.1, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 6}
	#Train err: 3.06%, train loss: 0.1074703614725623
	#Validation err: 2.93951%, validation loss: 0.10627454068193738

#{'batch_size': 256, 'lr': 0.04, 'gamma': 0.4, 'step_size': 4, 'momentum': 0.5, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 6}
	#Train err: 2.20%, train loss: 0.08294672526188138
	#Validation err: 2.62860%, validation loss: 0.08354651216162741

#{'batch_size': 256, 'lr': 0.04, 'gamma': 0.5, 'step_size': 3, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 6}
  #Train err: 0.69%, train loss: 0.03364661749625622
  #Validation err: 0.84794%, validation loss: 0.04012064782743605

#{'batch_size': 256, 'lr': 0.04, 'gamma': 0.5, 'step_size': 2, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 6}
  #Train err: 0.90%, train loss: 0.036477812544204465
  #Validation err: 0.79141%, validation loss: 0.03676502219174392

#{'batch_size': 256, 'lr': 0.04, 'gamma': 0.5, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 6}
	#Train err: 0.97%, train loss: 0.04007465636372104
	#Validation err: 0.87620%, validation loss: 0.03986180851045404

#{'batch_size': 256, 'lr': 0.04, 'gamma': 0.75, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 6}
	#Train err: 0.85%, train loss: 0.036348523716478384
	#Validation err: 0.98926%, validation loss: 0.04148892608906604

#{'batch_size': 256, 'lr': 0.03, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 6}
	#Train err: 0.71%, train loss: 0.03232140436248724
	#Validation err: 0.93273%, validation loss: 0.03983777600149437

#{'batch_size': 256, 'lr': 0.025, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 6}
	#Train err: 0.73%, train loss: 0.03311249506190535
	#Validation err: 0.93273%, validation loss: 0.03922237090265232

#{'batch_size': 256, 'lr': 0.015, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 8}
	#Train err: 0.88%, train loss: 0.03615223240944766
	#Validation err: 1.04579%, validation loss: 0.042104390417707374

#{'batch_size': 256, 'lr': 0.015, 'gamma': 0.95, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 8}
	#Train err: 0.78%, train loss: 0.03228224773508634
	#Validation err: 1.04579%, validation loss: 0.03987738015651366

#{'batch_size': 256, 'lr': 0.05, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 8}
  #Train err: 0.62%, train loss: 0.03029245081855807
  #Validation err: 0.76314%, validation loss: 0.03761576529678617

#{'batch_size': 256, 'lr': 0.045, 'gamma': 0.75, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 8}
	#Train err: 0.97%, train loss: 0.038201305351054024
	#Validation err: 1.27191%, validation loss: 0.04606578458014216

#{'batch_size': 256, 'lr': 0.05, 'gamma': 0.5, 'step_size': 4, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 8}
	#Train err: 0.71%, train loss: 0.0317742958260599
	#Validation err: 0.79141%, validation loss: 0.03849701658354566

#{'batch_size': 256, 'lr': 0.05, 'gamma': 0.25, 'step_size': 4, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 8}
	#Train err: 0.75%, train loss: 0.03306950848231944
	#Validation err: 0.84794%, validation loss: 0.03870768648813915

#{'batch_size': 256, 'lr': 0.05, 'gamma': 0.25, 'step_size': 3, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 8}
	#Train err: 0.91%, train loss: 0.0358474415171054
	#Validation err: 0.93273%, validation loss: 0.04020787683489836

#{'batch_size': 256, 'lr': 0.075, 'gamma': 0.5, 'step_size': 2, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 8}
	#Train err: 0.78%, train loss: 0.033065514682337295
	#Validation err: 1.01752%, validation loss: 0.04116438041557492

#{'batch_size': 256, 'lr': 0.065, 'gamma': 0.5, 'step_size': 2, 'momentum': 0.95, 'weight_decay': 0.01, 'nesterov': True, 'epochs_per_early_stop_check': 8}
	#Train err: 0.80%, train loss: 0.034910520163270856
	#Validation err: 1.04579%, validation loss: 0.04202789992489944

        #{'batch_size': 256, 'lr': 0.05, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 8}
          #Train err: 0.00%, train loss: 0.0009248352643642069
          #Validation err: 0.70661%, validation loss: 0.035158593715113255

#{'batch_size': 256, 'lr': 0.05, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 8}
	#Train err: 0.00%, train loss: 0.0009747934003149398
	#Validation err: 0.81967%, validation loss: 0.029555434110501246

#{'batch_size': 256, 'lr': 0.05, 'gamma': 0.95, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 8}
  #Train err: 0.00%, train loss: 0.00026671748604740976
  #Validation err: 0.76314%, validation loss: 0.040049300981299094

#{'batch_size': 256, 'lr': 0.05, 'gamma': 0.85, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 8}
	#Train err: 0.00%, train loss: 0.0020431274432660073
	#Validation err: 0.70661%, validation loss: 0.027773354945882286

#{'batch_size': 256, 'lr': 0.05, 'gamma': 0.85, 'step_size': 1, 'momentum': 0.8, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 8}
	#Train err: 0.74%, train loss: 0.02525420079022184
	#Validation err: 1.38496%, validation loss: 0.03923805013463246

#{'batch_size': 256, 'lr': 0.05, 'gamma': 0.8, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 8}
	#Train err: 0.10%, train loss: 0.006036224760802809
	#Validation err: 1.13058%, validation loss: 0.037855800616077406

#{'batch_size': 256, 'lr': 0.05, 'gamma': 0.8, 'step_size': 2, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 8}
	#Train err: 0.00%, train loss: 0.0009068694270699638
	#Validation err: 0.76314%, validation loss: 0.0338849965419774

#{'batch_size': 256, 'lr': 0.05, 'gamma': 0.8, 'step_size': 3, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 8}
	#Train err: 0.00%, train loss: 0.00037800573813875633
	#Validation err: 0.70661%, validation loss: 0.03590552966437345

#{'batch_size': 256, 'lr': 0.05, 'gamma': 0.8, 'step_size': 4, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 8}
	#Train err: 0.00%, train loss: 0.00023748720689250748
	#Validation err: 0.84794%, validation loss: 0.036585278078332464

#{'batch_size': 32, 'lr': 0.005, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 8}
	#Train err: 0.00%, train loss: 0.0016233104194492796
	#Validation err: 0.87620%, validation loss: 0.031606494796021706

#{'batch_size': 64, 'lr': 0.01, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 4}
	#Train err: 0.00%, train loss: 0.0019204076975150857
	#Validation err: 0.73488%, validation loss: 0.02664106812827192

#{'batch_size': 64, 'lr': 0.0075, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 4}
	#Train err: 0.05%, train loss: 0.004088930661118654
	#Validation err: 0.87620%, validation loss: 0.03062591945711437

#{'batch_size': 64, 'lr': 0.0085, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
	#Train err: 0.00%, train loss: 0.0025044748462130163
	#Validation err: 0.76314%, validation loss: 0.02832371295825144

        #{'batch_size': 64, 'lr': 0.011, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
          #Train err: 0.00%, train loss: 0.001500206267936769
          #Validation err: 0.62182%, validation loss: 0.027539270135551405

#{'batch_size': 64, 'lr': 0.015, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
	#Train err: 0.00%, train loss: 0.0008092841702917691
	#Validation err: 0.76314%, validation loss: 0.030543041205451676

#{'batch_size': 64, 'lr': 0.015, 'gamma': 0.85, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
	#Train err: 0.00%, train loss: 0.0019004748747936747
	#Validation err: 0.79141%, validation loss: 0.034142297815264046

#{'batch_size': 64, 'lr': 0.011, 'gamma': 0.85, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
	#Train err: 0.04%, train loss: 0.003554045889606021
	#Validation err: 0.81967%, validation loss: 0.029110537759787017

#{'batch_size': 64, 'lr': 0.011, 'gamma': 0.95, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
	#Train err: 0.00%, train loss: 0.0004909151004060611
	#Validation err: 0.73488%, validation loss: 0.029969342978467976

#{'batch_size': 64, 'lr': 0.011, 'gamma': 0.9, 'step_size': 2, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
	#Train err: 0.00%, train loss: 0.00044735293693153744
	#Validation err: 0.81967%, validation loss: 0.030688964353780615

#{'batch_size': 16, 'lr': 0.005, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
	#Train err: 0.00%, train loss: 0.0003427335543038194
	#Validation err: 0.87620%, validation loss: 0.034066283946921086

#{'batch_size': 16, 'lr': 0.001, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
	#Train err: 0.29%, train loss: 0.012300654336117035
	#Validation err: 1.04579%, validation loss: 0.03360803353712342

#{'batch_size': 16, 'lr': 0.005, 'gamma': 0.85, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
	#Train err: 0.00%, train loss: 0.0008374504391248374
	#Validation err: 0.81967%, validation loss: 0.030493847563559554

#{'batch_size': 16, 'lr': 0.005, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.9, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
  #Train err: 0.00%, train loss: 0.000799425926407828
  #Validation err: 0.52007%, validation loss: 0.019508428942599562

#{'batch_size': 16, 'lr': 0.005, 'gamma': 0.95, 'step_size': 1, 'momentum': 0.9, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
  #Train err: 0.00%, train loss: 0.000206410945005094
  #Validation err: 0.42962%, validation loss: 0.02269454082045567

#{'batch_size': 16, 'lr': 0.005, 'gamma': 0.95, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
	#Train err: 0.00%, train loss: 7.455142983325663e-05
	#Validation err: 0.44093%, validation loss: 0.02074236580991455

#{'batch_size': 32, 'lr': 0.005, 'gamma': 0.95, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
	#Train err: 0.00%, train loss: 0.0002603895595676695
	#Validation err: 0.50876%, validation loss: 0.021126534869613252

#{'batch_size': 32, 'lr': 0.0075, 'gamma': 0.95, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
	#Train err: 0.00%, train loss: 9.259251893909518e-05
	#Validation err: 0.49746%, validation loss: 0.026563515674602925

#{'batch_size': 32, 'lr': 0.0025, 'gamma': 0.95, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
	#Train err: 0.00%, train loss: 0.001139837196421658
	#Validation err: 0.55399%, validation loss: 0.02074500241708021

        #{'batch_size': 16, 'lr': 0.005, 'gamma': 0.95, 'step_size': 1, 'momentum': 0.9, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
          #Train err: 0.00%, train loss: 0.00018645416114801026
          #Validation err: 0.40701%, validation loss: 0.023587418449310813

#{'batch_size': 16, 'lr': 0.005, 'gamma': 0.95, 'step_size': 1, 'momentum': 0.75, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
	#Train err: 0.02%, train loss: 0.002809928969097461
	#Validation err: 0.58790%, validation loss: 0.020294339582745204

#{'batch_size': 16, 'lr': 0.005, 'gamma': 0.95, 'step_size': 1, 'momentum': 0.8, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
	#Train err: 0.00%, train loss: 0.0011180235550250649
	#Validation err: 0.53137%, validation loss: 0.021807030449611377

#{'batch_size': 256, 'lr': 0.05, 'gamma': 0.9, 'step_size': 1, 'momentum': 0.95, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 8}
	#Train err: 0.00%, train loss: 0.000454865331289335
	#Validation err: 0.52007%, validation loss: 0.022628355003501684


# ### Save model for later

# In[ ]:


#PATH = "/content/state_dicts/" + str(parameters).replace(' ', '').replace(':', '').replace('\'', '').replace(',', '__').strip('{').strip('}') + ".pt"
#torch.save(net.state_dict(), PATH)


# ### Load and evaluate saved model
# Manually fill in the paramters of the model that you want to load.

# In[26]:


parameters = {'batch_size': 16, 'lr': 0.005, 'gamma': 0.95, 'step_size': 1, 'momentum': 0.9, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
file_name = str(parameters).replace(' ', '').replace(':', '').replace('\'', '').replace(',', '__').strip('{').strip('}') + ".pt"
net = Net().cuda()
net.load_state_dict(torch.load("content/state_dicts/" + file_name))
acc_train, loss_train, acc_val, loss_val = train_val_acc_eval_f(net.cuda().eval(), tuning=False)
print(f"\tTrain err: {100-acc_train*100:.5f}%, train loss: {loss_train}")
print(f"\tValidation err: {100-acc_val*100:.5f}%, validation loss: {loss_val}\n")


# In[27]:


report_confumat(net)
train_test_acc_eval_f(net)
# 16:
# Train err: 0.00%, train loss: 0.0002674004728562928
# TEST ERR: 0.59%, test loss: 0.02399608348410554
# 256:
#Train err: 0.00%, train loss: 0.00045486536840799935
#TEST ERR: 0.61%, test loss: 0.02706974568359691


# In[ ]:


net = Net()

layers = [lambda x=x: getattr(x, 'weight') for x in net.modules() if isinstance(x, nn.Linear)]

print(len(layers))

layers = [lambda x=x: getattr(x, 'weight') for x in net.modules() if isinstance(x, nn.Linear) or isinstance(x, nn.Conv2d)]

print(len(layers))


# # Compression

# In[6]:


def my_l_step(model, lc_penalty, step):
    train_loader, val_loader, test_loader = data_loader()
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    # ------------------- Learning rate parameter
    lr = (0.2)*(0.98**step)
    # -------------------------------------------
    optimizer = optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    print(f'L-step #{step} with lr: {lr:.5f}')
    epochs_per_step_ = 10
    if step == 0:
        epochs_per_step_ = epochs_per_step_ * 2
    for epoch in range(epochs_per_step_):
        avg_loss = []
        for x, target in train_loader:
            optimizer.zero_grad()
            x = x.to(device)
            target = target.to(dtype=torch.long, device=device)
            out = model(x)
            loss = model.loss(out, target) + lc_penalty()
            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        print(f"\tepoch #{epoch} is finished.")
        print(f"\t  avg. train loss: {np.mean(avg_loss):.6f}")


# In[7]:


mu_s = [5e-5 * ((1.1) ** n) for n in range(30)]


# ### Pruning

# In[10]:


parameters = {'batch_size': 16, 'lr': 0.005, 'gamma': 0.95, 'step_size': 1, 'momentum': 0.9, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
file_name = str(parameters).replace(' ', '').replace(':', '').replace('\'', '').replace(',', '__').strip('{').strip('}') + ".pt"
net = Net().cuda()
net.load_state_dict(torch.load("content/state_dicts/" + file_name))


layers = [lambda x=x: getattr(x, 'weight') for x in net.modules() if isinstance(x, nn.Linear) or isinstance(x, nn.Conv2d)]
compression_tasks = {
    Param(layers, device): (AsVector, ConstraintL0Pruning(
            kappa=25000
          ), 'pruning')
}

lc_alg = lc.Algorithm(
    model=net,                            # model to compress
    compression_tasks=compression_tasks,  # specifications of compression
    l_step_optimization=my_l_step,        # implementation of L-step
    mu_schedule=mu_s,                     # schedule of mu values
    evaluation_func=train_test_acc_eval_f # evaluation function
)
lc_alg.run()                              # entry point to the LC algorithm
print('Compressed_params:', lc_alg.count_params())
print('Compression_ratio:', compute_compression_ratio(lc_alg))


# ### Mix of Pruning and Quantization

# In[5]:


parameters = {'batch_size': 16, 'lr': 0.005, 'gamma': 0.95, 'step_size': 1, 'momentum': 0.9, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
file_name = str(parameters).replace(' ', '').replace(':', '').replace('\'', '').replace(',', '__').strip('{').strip('}') + ".pt"
net = Net().cuda()
net.load_state_dict(torch.load("content/state_dicts/" + file_name))

layers = [lambda x=x: getattr(x, 'weight') for x in net.modules() if isinstance(x, nn.Linear) or isinstance(x, nn.Conv2d)]

compression_tasks = {
    Param(layers, device): [
        (AsVector, ConstraintL0Pruning(kappa=20000), 'pruning'),
        (AsVector, AdaptiveQuantization(k=2), 'quant')
    ]
}

lc_alg = lc.Algorithm(
    model=net,                            # model to compress
    compression_tasks=compression_tasks,  # specifications of compression
    l_step_optimization=my_l_step,        # implementation of L-step
    mu_schedule=mu_s,                     # schedule of mu values
    evaluation_func=train_test_acc_eval_f # evaluation function
)
lc_alg.run()
print('Compressed_params:', lc_alg.count_params())
print('Compression_ratio:', compute_compression_ratio(lc_alg))


# ### Mix of Pruning and Low-Rank
# Try using pruning on linear layers and Low-Rank on convolutional layers, and vice versa

# In[ ]:


parameters = {'batch_size': 16, 'lr': 0.005, 'gamma': 0.95, 'step_size': 1, 'momentum': 0.9, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
file_name = str(parameters).replace(' ', '').replace(':', '').replace('\'', '').replace(',', '__').strip('{').strip('}') + ".pt"
net = Net().cuda()
net.load_state_dict(torch.load("content/state_dicts/" + file_name))

layers = [lambda x=x: getattr(x, 'weight') for x in net.modules() if isinstance(x, nn.Linear) or isinstance(x, nn.Conv2d)]

compression_tasks = {
    Param(layers, device): [
        (AsVector, ConstraintL0Pruning(kappa=2662), 'pruning'),
        (AsVector, AdaptiveQuantization(k=2), 'quant')
    ]
}

lc_alg = lc.Algorithm(
    model=net,                            # model to compress
    compression_tasks=compression_tasks,  # specifications of compression
    l_step_optimization=my_l_step,        # implementation of L-step
    mu_schedule=mu_s,                     # schedule of mu values
    evaluation_func=train_test_acc_eval_f # evaluation function
)
lc_alg.run()
print('Compressed_params:', lc_alg.count_params())
print('Compression_ratio:', compute_compression_ratio(lc_alg))


# ### Quantization

# In[21]:


parameters = {'batch_size': 16, 'lr': 0.005, 'gamma': 0.95, 'step_size': 1, 'momentum': 0.9, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
file_name = str(parameters).replace(' ', '').replace(':', '').replace('\'', '').replace(',', '__').strip('{').strip('}') + ".pt"
net = Net().cuda()
net.load_state_dict(torch.load("content/state_dicts/" + file_name))

layers = [lambda x=x: getattr(x, 'weight') for x in net.modules() if isinstance(x, nn.Linear) or isinstance(x, nn.Conv2d)]
# k = 2 for each layer gives 30x compression
# k = 4 for each layer gives 15x compression
# k = 8 for each layer gives 10x compression
# k = 16 for each layer gives about 8x compression
# k = 32 for each layer gives about 6.2x compression
# k = 64 for each layer gives about 5.28x compresssion
# k = 128 for each layer gives about 4.5x compression

compression_tasks = {
    Param(layers[0], device): (AsVector, AdaptiveQuantization(k=64), 'layer0_quant'),
    Param(layers[1], device): (AsVector, AdaptiveQuantization(k=64), 'layer1_quant'),
    Param(layers[2], device): (AsVector, AdaptiveQuantization(k=64), 'layer2_quant'),
    Param(layers[3], device): (AsVector, AdaptiveQuantization(k=64), 'layer3_quant')
}

lc_alg = lc.Algorithm(
    model=net,                            # model to compress
    compression_tasks=compression_tasks,  # specifications of compression
    l_step_optimization=my_l_step,        # implementation of L-step
    mu_schedule=mu_s,                     # schedule of mu values
    evaluation_func=train_test_acc_eval_f # evaluation function
)
lc_alg.run()  
print('Compressed_params:', lc_alg.count_params())
print('Compression_ratio:', compute_compression_ratio(lc_alg))


# ### Low-rank compression with automatic rank selection

# In[7]:


parameters = {'batch_size': 16, 'lr': 0.005, 'gamma': 0.95, 'step_size': 1, 'momentum': 0.9, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
file_name = str(parameters).replace(' ', '').replace(':', '').replace('\'', '').replace(',', '__').strip('{').strip('}') + ".pt"
net = Net().cuda()
net.load_state_dict(torch.load("content/state_dicts/" + file_name))

layers = [lambda x=x: getattr(x, 'weight') for x in net.modules() if isinstance(x, nn.Linear) or isinstance(x, nn.Conv2d)]
# ----------------- alpha - compresssion parameter
alpha=1e-9
# alpha=1e-3
# ------------------------------------------------
compression_tasks = {
    Param(layers[0], device): (AsIs, RankSelection(conv_scheme='scheme_2', alpha=alpha, criterion='storage', module=layers[0], normalize=True), "layer1_lr"),
    Param(layers[1], device): (AsIs, RankSelection(conv_scheme='scheme_2', alpha=alpha, criterion='storage', module=layers[1], normalize=True), "layer2_lr"),
    Param(layers[2], device): (AsIs, RankSelection(conv_scheme='scheme_1', alpha=alpha, criterion='storage', module=layers[2], normalize=True), "layer3_lr"),
    Param(layers[3], device): (AsIs, RankSelection(conv_scheme='scheme_1', alpha=alpha, criterion='storage', module=layers[2], normalize=True), "layer3_lr")
}

lc_alg = lc.Algorithm(
    model=net,                            # model to compress
    compression_tasks=compression_tasks,  # specifications of compression
    l_step_optimization=my_l_step,        # implementation of L-step
    mu_schedule=mu_s,                     # schedule of mu values
    evaluation_func=train_test_acc_eval_f # evaluation function
)
lc_alg.run()
print('Compressed_params:', lc_alg.count_params())
print('Compression_ratio:', compute_compression_ratio(lc_alg))


# ### Mix of Pruning, Low Rank, and Quantization

# In[28]:


# layer 0 kappa = 104
# layer 1 kappa = 2505
# layer 2 k = 2
# layer 3 alpha = 1e-9
# Train err: 0.06%, train loss: 0.0024381824390053057
# TEST ERR: 0.74%, test loss: 0.025427111382986627
# Compressed_params: 405136
# Compression_ratio: 22.971771039394586

# layer 0 kappa = 104
# layer 1 kappa = 2505
# layer 2 k = 2
# layer 3 alpha = 2.5e-9
# Train err: 0.09%, train loss: 0.0027759067967801187
# TEST ERR: 0.67%, test loss: 0.025797346877296033
# Compressed_params: 405136
# Compression_ratio: 22.973618130436893

# layer 0 kappa = 104
# layer 1 kappa = 2505
# layer 2 alpha = 2e-9
# layer 3 k = 2
# Train err: 0.19%, train loss: 0.005692896033211272
# TEST ERR: 1.17%, test loss: 0.03771758408024755
# Compressed_params: 10311
# Compression_ratio: 48.08948577239336

# layer 0 kappa = 104
# layer 1 kappa = 2505
# layer 2 alpha = 2.5e-9
# layer 3 k = 2
# Train err: 0.04%, train loss: 0.0023015917632584425
# TEST ERR: 0.80%, test loss: 0.026643005014989517
# Compressed_params: 10311
# Compression_ratio: 48.08661930842

# layer 0 kappa = 104
# layer 1 kappa = 2505
# layer 2 alpha = 3e-9
# layer 3 k = 2
# Train err: 0.09%, train loss: 0.0029669747408118475
# TEST ERR: 0.80%, test loss: 0.029288866328254556
# Compressed_params: 10311
# Compression_ratio: 48.08594489596993


# In[33]:


parameters = {'batch_size': 16, 'lr': 0.005, 'gamma': 0.95, 'step_size': 1, 'momentum': 0.9, 'weight_decay': 0, 'nesterov': True, 'epochs_per_early_stop_check': 5}
file_name = str(parameters).replace(' ', '').replace(':', '').replace('\'', '').replace(',', '__').strip('{').strip('}') + ".pt"
net = Net().cuda()
net.load_state_dict(torch.load("content/state_dicts/" + file_name))

layers = [lambda x=x: getattr(x, 'weight') for x in net.modules() if isinstance(x, nn.Linear) or isinstance(x, nn.Conv2d)]

alpha = 7.5e-8

compression_tasks = {
    Param(layers[0], device): (AsVector, ConstraintL0Pruning(kappa=104), 'pruning'),
    Param(layers[1], device): (AsVector, ConstraintL0Pruning(kappa=2505), 'pruning'),
    Param(layers[2], device): (AsIs, RankSelection(conv_scheme='scheme_1', alpha=alpha, criterion='storage', module=layers[2], normalize=True), "layer2_lr"),
    Param(layers[3], device): (AsVector, AdaptiveQuantization(k=2), 'layer3_quant'),
}

lc_alg = lc.Algorithm(
    model=net,                            # model to compress
    compression_tasks=compression_tasks,  # specifications of compression
    l_step_optimization=my_l_step,        # implementation of L-step
    mu_schedule=mu_s,                     # schedule of mu values
    evaluation_func=train_test_acc_eval_f # evaluation function
)
lc_alg.run()
print('Compressed_params:', lc_alg.count_params())
print('Compression_ratio:', compute_compression_ratio(lc_alg))


# ### Quantization alone

# In[39]:


# k=64 for each layer:
# TEST ERR: 0.61%, test loss: 0.03328548995416101
# Compressed_params: 428256
# Compression_ratio: 5.285763619096953

# k=32 for each layer:
# TEST ERR: 0.59%, test loss: 0.031655197397087734
# Compressed_params: 428128
# Compression_ratio: 6.341930805883572

# k=16 for each layer:
# TEST ERR: 0.57%, test loss: 0.03163435025814852
# Compressed_params: 428064
# Compression_ratio: 7.916197196106319

# k=8 for each layer:
# TEST ERR: 0.61%, test loss: 0.03194960453728469
# Compressed_params: 428032
# Compression_ratio: 10.521825591672394

# k=4 for each layer:
# TEST ERR: 0.74%, test loss: 0.027538442431905275
# Compressed_params: 428016
# Compression_ratio: 15.67517647489119

# k=2 for each layer:
# TEST ERR: 0.76%, test loss: 0.022537321148626888
# Compressed_params: 428008
# Compression_ratio: 30.704613841524573

q_test_errors = [0.61, 0.59, 0.57, 0.61, 0.74, 0.76]
q_n_compressed_parameters = [428256, 428128, 428064, 428032, 428016, 428008]
q_compression_ratios = [5.285763619096953, 6.341930805883572, 7.916197196106319, 10.521825591672394, 15.67517647489119, 30.704613841524573]


# ### Low-Rank Alone

# In[40]:


# alpha = 1e-9
# TEST ERR: 0.59%, test loss: 0.028039038540883056
# Compressed_params: 100300
# Compression_ratio: 4.2721658476562805

# alpha = 1.5e-9
# TEST ERR: 0.59%, test loss: 0.02737808789913152
# Compressed_params: 57450
# Compression_ratio: 7.457632648622194

# alpha = 2e-9
# TEST ERR: 1.86%, test loss: 0.07579609481713036
# Compressed_params: 25150
# Compression_ratio: 17.028589166537326

# 	alpha = 2.625e-9
# TEST ERR: 1.53%, test loss: 0.0711624405881988
# Compressed_params: 19450
# Compression_ratio: 22.01436654761427

# alpha = 2.5e-9
# TEST ERR: 1.53%, test loss: 0.07167478040287344
# Compressed_params: 18500
# Compression_ratio: 23.143737079694553

l_test_errors = [0.59, 0.59, 1.86, 1.53, 1.53]
l_n_compressed_parameters = [100300, 57450, 25150, 19450, 18500]
l_compression_ratios = [4.2721658476562805, 7.457632648622194, 17.028589166537326, 22.01436654761427, 23.143737079694553]


# ### Pruning Alone

# In[41]:


# kappa = 80000:
# TEST ERR: 0.53%, test loss: 0.024508386036735372
# Compressed_params: 80000
# Compression_ratio: 4.707208436345564

# kappa = 40000:
# TEST ERR: 0.59%, test loss: 0.02454856142310277
# Compressed_params: 40000
# Compression_ratio: 9.01502218850799

# kappa = 30000
# TEST ERR: 0.57%, test loss: 0.02393761289540244
# Compressed_params: 30000
# Compression_ratio: 11.817379675458067

# kappa = 25000
# TEST ERR: 78.98%, test loss: 1.6420240499246828
# Compressed_params: 25000
# Compression_ratio: 13.760718578736025

# kappa = 20000:
# TEST ERR: 79.96%, test loss: nan
# Compressed_params: 20000
# Compression_ratio: 19.637840330142076

# kappa = 12500:
# TEST ERR: 79.96%, test loss: nan
# Compressed_params: 12500
# Compression_ratio: 30.932457608386752

p_test_errors = [0.53, 0.59, 0.57, 78.98, 79.96, 79.96]
p_n_compressed_parameters = [80000, 40000, 30000, 25000, 20000, 12500]
p_compression_ratios = [4.707208436345564, 9.01502218850799, 11.817379675458067, 13.760718578736025, 19.637840330142076, 30.932457608386752]


# In[69]:


test_errs = [p_test_errors, q_test_errors, l_test_errors]

params = [p_n_compressed_parameters, q_n_compressed_parameters, l_n_compressed_parameters]
ratios = [p_compression_ratios, q_compression_ratios, l_compression_ratios]

compression_type = ["Pruning", "Quantization", "Low-Rank"]

x_axis = [params, ratios]
x_axis_labels = ["Number of Compressed Parameters", "Compression Ratio"]
y_axis = test_errs
y_axis_labels = ["Train", "Test"]
legend = {"Pruning": "bo-", "Quantization": "ro-", "Low-Rank": "go-"}

from matplotlib import pyplot as plt
# hline - reference network error
x_max = [430000, 32]
ref_test_err = .59
for curr_x in range(2): # Compressed Params and Compression Ratio
    fig = plt.figure()
    ax = plt.gca()
    ax.hlines(y=ref_test_err, xmin=0, xmax=x_max[curr_x], color='k', linestyles='dashed', label="Uncompressed Reference Network")
    for curr_type in range(3):
        ax.plot(x_axis[curr_x][curr_type], test_errs[curr_type], legend[compression_type[curr_type]], label=compression_type[curr_type])
        ax.set_xlabel(x_axis_labels[curr_x])
        ax.set_title('Test Error (%) vs. ' + x_axis_labels[curr_x])
        ax.set_ylabel('Test Error (%)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim([0, 2])
    ax.legend()

fig = plt.figure()
ax = plt.gca()
ax.hlines(y=ref_test_err, xmin=0, xmax=65, color='k', linestyles='dashed', label="Uncompressed Reference Network")
ax.plot([2**x for x in range(1,7,1)], q_test_errors, 'ro-', label="Quantization")
ax.set_xlabel("Codebook Size")
ax.set_title('Test Error (%) vs. Codebook Size (Quantization)')
ax.set_ylabel('Test Error (%)')
ax.set_xscale('linear')
ax.set_yscale('linear')
ax.legend()


# ### Part 1 Results

# In[90]:


# Quantization
test_errs_q = [1.31, 1.31, 1.31, 1.33, 1.37, 1.45]
ratios_q = [5.278211669964247, 6.333326391636644, 7.904382825147406, 10.50205468977391, 15.631856431644955, 
            30.537202530374536]

# Low-Rank
test_errs_l = [1.25, 1.37, 1.37, 1.35, 78.98]
ratios_l = [5.114453200076879, 8.7413770448722164, 10.962552525335756, 15.730034876160076, 32.274711946634326]

# Pruning
test_errs_p = [1.25, 1.25, 1.21, 1.21, 1.23, 1.19, 1.17, 1.33]
ratios_p = [5.001867911627809, 6.207001113787051, 8.354945054945055, 12.976122810038508, 16.31688875092455, 
            20.78356715383339, 24.467737857159275, 29.934964951381907]


# ### Part 2 Results

# In[91]:


q_test_errors = [0.61, 0.59, 0.57, 0.61, 0.74, 0.76]
q_compression_ratios = [5.285763619096953, 6.341930805883572, 7.916197196106319, 10.521825591672394, 15.67517647489119, 30.704613841524573]

l_test_errors = [0.59, 0.59, 1.86, 1.53, 1.53]
l_compression_ratios = [4.2721658476562805, 7.457632648622194, 17.028589166537326, 22.01436654761427, 23.143737079694553]

p_test_errors = [0.53, 0.59, 0.57, 78.98, 79.96, 79.96]
p_compression_ratios = [4.707208436345564, 9.01502218850799, 11.817379675458067, 13.760718578736025, 19.637840330142076, 30.932457608386752]


# In[111]:


fig = plt.figure()
ax = plt.gca()
ax.hlines(y=1.33, xmin=0, xmax=32, color='k', linestyles='solid', label="P1 Uncompressed Reference Network")
ax.hlines(y=.59, xmin=0, xmax=32, color='k', linestyles='dashed', label="P2 Uncompressed Reference Network")
ax.plot(ratios_l, test_errs_l, "bo-", label="Part 1 Low-Rank")
ax.plot(l_compression_ratios, l_test_errors, "b.--", label="Part 2 Low-Rank")
ax.set_xlabel("Compression Ratio")
ax.set_title('Test Error (%) vs. Compression Ratio for Low-Rank')
ax.set_ylabel('Test Error (%)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()


# fig = plt.figure()
# ax = plt.gca()
# ax.hlines(y=train_test_errs[curr_subset], xmin=0, xmax=65, color='k', linestyles='dashed', label="Uncompressed Reference Network")
# ax.plot(codebook_size, errs, 'ro-', label="Quantization")
# ax.set_xlabel("Codebook Size")
# ax.set_title(y_axis_labels[curr_subset] + ' Error vs. Codebook Size (Quantization)')
# ax.set_ylabel(y_axis_labels[curr_subset] + ' Error')
# ax.set_xscale('linear')
# ax.set_yscale('linear')
# ax.legend()


# ### Pruning on Conv layers and Quantization on Linear layers

# In[85]:


# With this compression scheme, we applied pruning to the convolutional layers, and quantization to the linear layers. Our initial
# findings were that the number of parameters of the first layer and the codebook size of both linear layers could be pretty
# low without much harm to the model. This compression scheme gave us one of our highest compression ratio to error ratios, with
# a test error of .51% and compression ratio of approx 26.4. This was given by compression with 104 parameters in the first convolutional 
# layer, 2505 parameters in the second conv layer, and a codebook size of 2 for both linear layers. We tried increasing the compression 
# a little bit more from that, and saw that it was impacting error enough as to not be worth pushing further. It looks like we got 
# lucky with these particular compression parameters, as the test error for the model mentioned above is lower than those of 
# models compressed with the same scheme but with more parameters in the second layer. A plot of this compression type's test error vs
# number of parameters in the second layer is given in Figure X.

# layer 1 kappa =  104
# layer 2 kappa = 12525
# layer 3 k = 2
# layer 4 k = 2
# Train err: 0.01%, train loss: 0.0014465828186132072
# TEST ERR: 0.53%, test loss: 0.020963299728488143
# Compressed_params: 415289
# Compression_ratio: 15.891025551894028

# layer 1 kappa =  104
# layer 2 kappa = 5010
# layer 3 k = 2
# layer 4 k = 2
# Train err: 0.00%, train loss: 0.001723080134206964
# TEST ERR: 0.55%, test loss: 0.01985923921677964
# Compressed_params: 407618
# Compression_ratio: 22.635991067346463

            # layer 1 kappa =  104
            # layer 2 kappa = 2505
            # layer 3 k = 2
            # layer 4 k = 2
            # Train err: 0.03%, train loss: 0.0024527820672789343
            # TEST ERR: 0.51%, test loss: 0.02095057955792589
            # Compressed_params: 405113
            # Compression_ratio: 26.42542655375396

# layer 1 kappa =  104
# layer 2 kappa = 1250
# layer 3 k = 2
# layer 4 k = 2
# Train err: 0.06%, train loss: 0.003353178411787159
# TEST ERR: 0.57%, test loss: 0.0208197200697502
# Compressed_params: 403858
# Compression_ratio: 28.963396605759566
test_errors = [0.53, 0.55, 0.51, 0.57]
compression_ratios = [15.891025551894028, 22.635991067346463, 26.42542655375396, 28.963396605759566]

fig = plt.figure()
ax = plt.gca()
ax.hlines(y=.59, xmin=compression_ratios[0]-.5, xmax=38, color='k', linestyles='dashed', label="Uncompressed Reference Network")
ax.plot(compression_ratios, test_errors, "b.-", label="Pruning on Conv layers and Quantization on Linear layers")
ax.plot(37.73248519798384, 0.53, "ro", label="Model with best Compression Ratio to Test Error Ratio")
ax.set_xlabel('Compression Ratio')
ax.set_title('Compression Ratio  vs. Test Error')
ax.set_ylabel('Test Error (%)')
ax.set_xscale('linear')
ax.set_yscale('linear')
ax.set_ylim([.5, .65])
ax.legend()


# ### Quantization of Conv layers and Low-Rank on Linear Layers

# In[ ]:


# For this compression scheme, we saw that we were able to get a pretty high compression while maintaining good accuracy. We wanted to see if 
# reducing the amount of compression would allow the model to perform better with a still relatively high compression. When we 
# lowered the compression by a factor of 3, we saw the error hardly decreased, and we had already seen a model that had higher
# compression with less error than the second run. Because of this, we decided to stop here, but the first run of this compression
# scheme still ended up giving our highest compression ratio with not terrible accuracy.

# alpha = 2.625e-9
# k = 2
# Train err: 0.10%, train loss: 0.0036937077804071496
# TEST ERR: 0.67%, test loss: 0.021679462028518778
# Compressed_params: 33229
# Compression_ratio: 47.09162580521104

# alpha = 1e-9
# k = 2
# Train err: 0.01%, train loss: 0.001493711800444438
# TEST ERR: 0.63%, test loss: 0.020371556772654287
# Compressed_params: 54029
# Compression_ratio: 14.333192590517836


# ### Low-rank on Conv Layers and Quantization on Linear Layers

# In[86]:


# With this compression scheme, we decided to keep the codebook size as 2 for the quantization of linear layers for all runs,
# since we have noticed that models tend to work pretty well with that type of compression. Here, we varied the amount of low-rank
# compression on the convolutional layers by changing the alpha value. There were no stand-out compressed models from this compression
# scheme, and the results are shown in Figure X

#  alpha = 1e-9
#  k = 2
#  Train err: 0.00%, train loss: 0.0007940481794554047
#  TEST ERR: 0.55%, test loss: 0.018662374498653996
#  Compressed_params: 434179
#  Compression_ratio: 9.559551326197454

#  alpha = 2.5e-9
#  k = 2
#  Train err: 0.00%, train loss: 0.0007536800849421109
#  TEST ERR: 0.65%, test loss: 0.02101732165169862
#  Compressed_params: 422279
#  Compression_ratio: 13.01388841442816

#  alpha = 5e-9
#  k = 2
#  Train err: 0.00%, train loss: 0.0006984795862447846
#  TEST ERR: 0.59%, test loss: 0.020446154859168397
#  Compressed_params: 410379
#  Compression_ratio: 20.37716112851174

#  alpha = 1e-8
#  k = 2
#  Train err: 0.00%, train loss: 0.0009552586485285225
#  TEST ERR: 0.67%, test loss: 0.020718296900665834
#  Compressed_params: 407229
#  Compression_ratio: 23.966670627791718

#  alpha = 2.5e-8
#  k = 2
#  Train err: 0.01%, train loss: 0.0013898124918792858
#  TEST ERR: 0.74%, test loss: 0.02361567228240226
#  Compressed_params: 406179
#  Compression_ratio: 25.461728688445458

#  alpha = 5e-8
#  k = 2
#  Train err: 78.45%, train loss: 1.60869908425235
#  TEST ERR: 78.98%, test loss: 1.6089411411792467
#  Compressed_params: 404114
# Compression_ratio: 29.02223689445305

test_errors = [0.55, 0.65, 0.59, 0.67, 0.74, 78.98]
compression_ratios = [9.559551326197454, 13.01388841442816, 20.37716112851174, 23.966670627791718, 25.461728688445458, 29.02223689445305]

fig = plt.figure()
ax = plt.gca()
ax.hlines(y=.59, xmin=compression_ratios[0]-.5, xmax=38, color='k', linestyles='dashed', label="Uncompressed Reference Network")
ax.plot(compression_ratios, test_errors, "b.-", label="Pruning on Conv layers and Quantization on Linear layers")
ax.plot(37.73248519798384, 0.53, "ro", label="Model with best Compression Ratio to Test Error Ratio")
ax.set_xlabel('Compression Ratio')
ax.set_title('Compression Ratio  vs. Test Error')
ax.set_ylabel('Test Error (%)')
ax.set_xscale('linear')
ax.set_yscale('linear')
ax.set_ylim([.5, .65])
ax.legend()


# ### Quantization on Conv Layers and Pruning on Linear Layers

# In[ ]:


# With this compression scheme, we again keep our quantization codebook at 2 for both layers, but apply quantization to the convolutional
# layers only. On the linear layers, we apply pruning with different numbers of parameters proportional to the number of starting
# parameters for each layer. We saw strange results here, with lower compression ratios giving higher error. The best compressed 
# model that we got with this compression scheme had relatively high error of .78%, with a compression ratio of only approx. 17.
# We noticed that this scheme gave some of our worst results, and is the inverse of the scheme that gave some of our best results 
# (quantization on linear layers and pruning on conv layers). This leads us to believe that, for this model, quantization is more
# effective on linear layers and pruning is more effective on convolutional layers.

# layer 0 k = 2
# layer 1 k = 2
# layer 2 kappa = 100000
# layer 3 kappa = 1000
# Train err: 0.11%, train loss: 0.004736287437032822
# TEST ERR: 0.84%, test loss: 0.02545553339999512
# Compressed_params: 126504
# Compression_ratio: 3.766573718079878

# layer 0 k = 2
# layer 1 k = 2
# layer 2 kappa = 80000
# layer 3 kappa = 500
# Train err: 0.14%, train loss: 0.005909851407126863
# TEST ERR: 0.82%, test loss: 0.024643742107235824
# Compressed_params: 106004
# Compression_ratio: 4.660869919845843

# layer 0 k = 2
# layer 1 k = 2
# layer 2 kappa = 40000
# layer 3 kappa = 250
# Train err: 0.10%, train loss: 0.004272402206060383
# TEST ERR: 0.84%, test loss: 0.02374215452476635
# Compressed_params: 65754
# Compression_ratio: 8.876197675713039

# layer 0 k = 2
# layer 1 k = 2
# layer 2 kappa = 20000
# layer 3 kappa = 125
# Train err: 0.06%, train loss: 0.0037158044668766416
# TEST ERR: 0.78%, test loss: 0.022713467773575725
# Compressed_params: 45629
# Compression_ratio: 16.641104294478527


# ### Low-Rank on Conv Layers and Pruning on Linear Layers

# In[ ]:


# With this compression scheme, we again observed that pruning was not working very well on the linear layers. For very low 
# compression ratios, we were already seeing error rates higher than we had seen before with much higher compression ratios. When
# we pushed the model to have a compression ratio close to 20, it started giving garbage output, so we decided to stop there. 

# alpha = 1e-9
# layer 2 kappa = 80000
# layer 3 kappa = 500
# Train err: 0.00%, train loss: 0.000249883399526372
# TEST ERR: 0.63%, test loss: 0.022979293916001155
# Compressed_params: 111125
# Compression_ratio: 3.5195712376729418

# alpha = 2.625e-9
# layer 2 kappa = 40000
# layer 3 kappa = 250
# Train err: 0.00%, train loss: 0.0003304009178703896
# TEST ERR: 0.57%, test loss: 0.023337179302011287
# Compressed_params: 56875
# Compression_ratio: 6.693464472083884

# alpha = 2.625e-9
# layer 2 kappa = 15000
# layer 3 kappa = 250
# Train err: 80.15%, train loss: 1.6093867913697117
# TEST ERR: 80.41%, test loss: 1.6099159806296381
# Compressed_params: 17875
# Compression_ratio: 19.249415404136105


# ### Pruning on Conv Layers and Low-Rank on Linear Layers

# In[89]:


# This compression scheme gave us our best compression ratio to error ratio. With 104 parameters in the first layer, 2505 parameters
# in the second layer, and an alpha of $2.5*10^{-9}$ for low-rank compression applied to the linear layers. The resulting error
# was .53%, and the compression ratio was 37.7. We have noticed a trend that some of our best models were compressed with pruning
# on the convolutional layers, so when we tried to combine all three compression types into one model, we decided to fix the 
# compression on the first two layers to be pruning with 104 parameters for the first layer and 2505 parameters for the second
# layer.

# layer 0 kappa = 104
# layer 1 kappa = 5010
# alpha = 1e-9
# Train err: 0.00%, train loss: 0.00023444037877254245
# TEST ERR: 0.57%, test loss: 0.023048661883867103
# Compressed_params: 41439
# Compression_ratio: 10.04371385404743

# layer 0 kappa = 104
# layer 1 kappa = 5010
# alpha = 2.5e-9
# Train err: 0.00%, train loss: 0.00023940207609047666
# TEST ERR: 0.61%, test loss: 0.022994919822138023
# Compressed_params: 12839
# Compression_ratio: 30.45465244869227

            # layer 0 kappa = 104
            # layer 1 kappa = 2505
            # alpha = 2.5e-9
            # Train err: 0.00%, train loss: 0.00026679373799296197
            # TEST ERR: 0.53%, test loss: 0.023120258530651378
            # Compressed_params: 10334
            # Compression_ratio: 37.73248519798384

# layer 0 kappa = 104
# layer 1 kappa = 2505
# alpha = 2.75e-9
# Train err: 0.00%, train loss: 0.0002578537257355746
# TEST ERR: 0.57%, test loss: 0.02328361051884653
# Compressed_params: 10334
# Compression_ratio: 37.72926725264447

test_errors = [0.57, 0.61, 0.53, 0.57]
compression_ratios = [10.04371385404743, 30.45465244869227, 37.73248519798384, 37.72926725264447]

fig = plt.figure()
ax = plt.gca()
ax.hlines(y=.59, xmin=compression_ratios[0]-.5, xmax=38, color='k', linestyles='dashed', label="Uncompressed Reference Network")
ax.plot(compression_ratios, test_errors, "b.-", label="Pruning on Conv Layers and Low-Rank on Linear Layers")
ax.plot(37.73248519798384, 0.53, "ro", label="Model with best Compression Ratio to Test Error Ratio")
ax.set_xlabel('Compression Ratio')
ax.set_title('Compression Ratio  vs. Test Error')
ax.set_ylabel('Test Error (%)')
ax.set_xscale('linear')
ax.set_yscale('linear')
ax.set_ylim([.5, .65])
ax.legend()


# ### Mix of Pruning, Low Rank, and Quantization

# In[28]:


# layer 0 kappa = 104
# layer 1 kappa = 2505
# layer 2 k = 2
# layer 3 alpha = 1e-9
# Train err: 0.06%, train loss: 0.0024381824390053057
# TEST ERR: 0.74%, test loss: 0.025427111382986627
# Compressed_params: 405136
# Compression_ratio: 22.971771039394586

# layer 0 kappa = 104
# layer 1 kappa = 2505
# layer 2 k = 2
# layer 3 alpha = 2.5e-9
# Train err: 0.09%, train loss: 0.0027759067967801187
# TEST ERR: 0.67%, test loss: 0.025797346877296033
# Compressed_params: 405136
# Compression_ratio: 22.973618130436893

# layer 0 kappa = 104
# layer 1 kappa = 2505
# layer 2 alpha = 2e-9
# layer 3 k = 2
# Train err: 0.19%, train loss: 0.005692896033211272
# TEST ERR: 1.17%, test loss: 0.03771758408024755
# Compressed_params: 10311
# Compression_ratio: 48.08948577239336

# layer 0 kappa = 104
# layer 1 kappa = 2505
# layer 2 alpha = 2.5e-9
# layer 3 k = 2
# Train err: 0.04%, train loss: 0.0023015917632584425
# TEST ERR: 0.80%, test loss: 0.026643005014989517
# Compressed_params: 10311
# Compression_ratio: 48.08661930842

# layer 0 kappa = 104
# layer 1 kappa = 2505
# layer 2 alpha = 3e-9
# layer 3 k = 2
# Train err: 0.09%, train loss: 0.0029669747408118475
# TEST ERR: 0.80%, test loss: 0.029288866328254556
# Compressed_params: 10311
# Compression_ratio: 48.08594489596993

