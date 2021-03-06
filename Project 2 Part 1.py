# -*- coding: utf-8 -*-
"""My Copy of LC_algorithm_demo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ajz-zJZxk-tLkIewmIKJGqtBtBqoHpqb
"""

! git clone https://github.com/UCMerced-ML/LC-model-compression

! pip3 install -e ./LC-model-compression

"""## IMPORTANT!
At this point you need to restart the runtime by doing "Runtime => Restart Runtime"
"""

import lc
from lc.torch import ParameterTorch as Param, AsVector, AsIs
from lc.compression_types import ConstraintL0Pruning, LowRank, RankSelection, AdaptiveQuantization
from lc.models.torch import lenet300_classic, lenet300_modern_drop, lenet300_modern

import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import datasets
torch.set_num_threads(4)

"""## Data
We use a subset of the MNIST dataset. The dataset contains subtracted 28x28 grayscale images with digits 0, 2, 5, 6, and 7. The images are normalized to have grayscale value 0 to 1 and then mean is subtracted.
"""

from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = [10, 5]
def show_MNIST_images():
    train_data_th = datasets.MNIST(root='./datasets', download=True, train=True)
    data_train = np.array(train_data_th.data[:])
    targets = np.array(train_data_th.targets)
    images_to_show = 5
    random_indexes = np.random.randint(data_train.shape[0], size=images_to_show)
    for i,ind in enumerate(random_indexes):
        plt.subplot(1,images_to_show,i+1)
        plt.imshow(data_train[ind], cmap='gray')
        plt.xlabel(targets[ind])
        plt.xticks([])
        plt.yticks([])
show_MNIST_images()

def data_loader(batch_size=2048, n_workers=4):
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

    data_train = np.array(train_data[:]).reshape([-1, 28 * 28]).astype(np.float32)
    data_test = np.array(test_data[:]).reshape([-1, 28 * 28]).astype(np.float32)

    data_train = (data_train / 255)
    dtrain_mean = data_train.mean(axis=0)
    data_train -= dtrain_mean
    data_test = (data_test / 255).astype(np.float32)
    data_test -= dtrain_mean

    train_data = TensorDataset(torch.from_numpy(data_train), train_targets)

    # Create validation set
    val_split = int(0.3 * len(train_data))
    train_data, val_data = random_split(train_data, [len(train_data) - val_split, val_split], generator=torch.Generator().manual_seed(1778))

    test_data = TensorDataset(torch.from_numpy(data_test), test_targets)

    train_loader = DataLoader(train_data, num_workers=n_workers, batch_size=batch_size, shuffle=True,)
    val_loader = DataLoader(val_data, num_workers=n_workers, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, num_workers=n_workers, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

"""## Our Subset of MNIST"""

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = [10, 5]
label_map = {0: 0, 1: 2, 2: 5, 3: 6, 4: 7}
def show_subset_MNIST_images(images_to_show):
    first_batch, labels = next(iter(data_loader(images_to_show)[0]))
    data_train = np.array(first_batch[:]).reshape(-1, 28, 28)
    targets = np.array(labels)
    for i in range(images_to_show):
        plt.subplot(1,images_to_show,i+1)
        plt.imshow(data_train[i], cmap='gray')
        plt.xlabel(str(label_map[targets[i]]) + " (label: " + str(targets[i]) + ")")
        plt.xticks([])
        plt.yticks([])
show_subset_MNIST_images(5)

"""## Helper Functions"""

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
    compressed_model_bits = lc_alg.count_param_bits() + (300+100+5)*32
    uncompressed_model_bits = (784*300+300*100+100*5 + 300 + 100 + 5)*32
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
  
def train_val_acc_eval_f(net):
    train_loader, val_loader, _ = data_loader()
    def forward_func(x, target):
        y = net(x)
        return y, net.loss(y, target)
    with torch.no_grad():
        acc_train, loss_train = compute_acc_loss(forward_func, train_loader)
        acc_val, loss_val = compute_acc_loss(forward_func, val_loader)

    return acc_train, loss_train, acc_val, loss_val

def visualize_errs(errs, subset, parameters):
    epochs = np.arange(len(errs))
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(epochs * 10, errs)
    ax.set_xlabel('SGD Step')
    ax.set_title(subset + ' Accuracy per SGD step')
    ax.set_ylabel(subset + ' Accuracy')
    ax.set_xscale('log')
    ax.set_yscale('log')

def visualize_params(train_error, test_error, num_params):
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(num_params, train_error)
    ax.set_xlabel('Number of Compressed Parameters')
    ax.set_title('Number of Compressed Parameters vs. Train Error')
    ax.set_ylabel('Train Error')
    ax.set_xscale('log')
    ax.set_yscale('log')
    # test
    fig1 = plt.figure()
    ax1 = plt.gca()
    ax1.plot(num_params, test_error)
    ax1.set_xlabel('Number of Compressed Parameters')
    ax1.set_title('Number of Compressed Parameters vs. Test Error')
    ax1.set_ylabel('Test Error')
    ax1.set_xscale('log')
    ax1.set_yscale('log')

def visualize_ratios(train_error, test_error, ratios):
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(ratios, train_error)
    ax.set_xlabel('Compression Ratio')
    ax.set_title('Compression Ratio  vs. Train Error')
    ax.set_ylabel('Train Error')
    ax.set_xscale('linear')
    ax.set_yscale('log')
    # test
    fig1 = plt.figure()
    ax1 = plt.gca()
    ax1.plot(ratios, test_error)
    ax1.set_xlabel('Compression Ratio')
    ax1.set_title('Compression Ratio  vs. Test Error')
    ax1.set_ylabel('Test Error')
    ax1.set_xscale('linear')
    ax1.set_yscale('log')

"""##Reference Network
We use cuda capable GPU for our experiments. The network has 3 fully-connected layers with dimensions 784x300, 300x100, and 100x5, and the total of 266105 parameters (which includes biases).

"""

device = torch.device('cuda')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 5)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_net(net, parameters):
    train_errs = []
    val_errs = []

    max_val_acc = 0
    #min_val_loss = np.inf
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
            if (i % 10 == 9):
                # -------------------------------------------------------------------------------
                acc_train, loss_train, acc_val, loss_val = train_val_acc_eval_f(net.eval())
                #print(f"\tTrain err: {100-acc_train*100:.2f}%, train loss: {loss_train}")
                #print(f"\tValidation err: {100-acc_val*100:.5f}%, validation loss: {loss_val}\n")
                train_errs.append(acc_train)
                val_errs.append(acc_val)
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
        
        # ------------------------------------------------------------------------------- 
        #if (min_val_loss - early_stop_thresh > loss_val):
        if (epoch % epochs_per_early_stop_check == epochs_per_early_stop_check - 1):
            if (max_val_acc + early_stop_thresh < acc_val):
                #min_val_loss = loss_val
                max_val_acc = acc_val
                early_stop_patience = intitial_early_stop_patience
            else:
                early_stop_patience -= 1
            if (early_stop_patience == 0):
                break;
    
    visualize_errs(train_errs, "Train", parameters)
    visualize_errs(val_errs, "Validation", parameters)

"""##Parameter Selection
We use Stochastic Gradient Descent and try to find the optimal combination of momentum, batch size, learning rate, and l2 regularization.
"""

# Batch size: powers of 2 (2^0 (batch-size 1) to 2^16 (gradient descent)) # very small batch sizes take a long time for each epoch, very large batch sizes take longer to converge
# likewise, patience too high for very small batch size
# lr: small positive number to value at which train loss oscillates up and down from very start of training (perhaps try 1e-5 - 1e4)
# gamma (factor by which learning rate decreases): small positive number to 1
# step size (how many epochs pass between each lr scheduler step): integers from 1 up
# momentum (addition of previous SGD step vector to current SGD step vector): 0 - 1 
# weight decay (penalty to loss for magnitude of weights): 0 to very large number (train error doesn't decrease because CELoss term dwarfed by weight magnitude loss term)
# nesterov (something to do with SGD, similar to momentum but looking at next update instead of previous): True or False

# Notes: 
# Wanted to find a way so that we didn't have to train every model for a long time regardless of how fast it converged, but still be able to show that validation error either increases
# or flattens out before giving up on a particular selection of hyperparameters
#   Decided to set a maximum number of epochs and use early stopping with a threshold of 1e-5 on validation accuracy to determine when to stop
#   Result is that (as long as the model didn't take longer than 100 epochs to converge) the validation accuracy curve will at least be flat for some time before stopping training
#   This introduced a new hyperparameter, epochs_per_early_stop_check, since some selections of hyperparameters will naturally take longer to converge than others
#     Training loop will only check to see if validation accuracy hasn't improved every <epochs_per_early_stop_check> epochs, allowing us to give more time to models
#     that naturally take longer to improve significantly
#     If validation accuracy hasn't improved by at least 1e-5 (.001%) since its last check, a counter is decremented. If this occurs three times, the training loop breaks
#
# Started by seeing if I could find a smaller, reasonable range of batch sizes
# for batch sizes 2^0, 2^1, 2^2 epochs each took a while and train error barely decreased, sometimes going up (lr may need to be lower in these cases)
# for batch size 2^16 (gradient descent), early stopping settings were too stringent and the model wasn't given time to converge - patience should be higher for higher batch size
# Changing learning rate to 0.01 on very small batch sizes worked very well, going to check progressively larger batch sizes to see if the learning rate is better in general
#   Lower batch sizes make epochs take longer, but the model converges to a good accuracy in fewer epochs - initial run kept going for too long after min validation error reached
#     so I reduced epochs per early stop check and trained again: validation error to beat 1.05144% (batch_size 8, lr: .01, epochs_per_early_stop_check: 3)
#   batch size 16 not doing better on val set than batch size 8 with same lr - validation accuracy flattens
#   batch sizes up to 256 weren't doing better than bs 8 with lr 0.01, so I tried increasing lr for larger batch sizes up to 0.05 and allowed model to train longer
#     models trained in this way consistently reached a minimal train and validation error and almost completely stopped moving from that point on
#     played around a bit with gamma and step size, but model error would consistently bottom out and not oscillate, so I assumed the values were good as they were
# Found that learning rates around 1 either don't work at all for smaller batch sizes, or oscillate a bit at the start then start behaving as lr_scheduler decreases it over time for larger 
#     batch sizes
# For gradient descent (batch_size 2**16), learning rate starting at about 1 works well, but a learning rate as high as 2 doesn't even work well on a very large batch size
#   Decreasing gamma helped with this, but it still underperforms compared to starting at a lower lr
# Decided to set learning rate range to be small positive number to 2. 
# 
# Best results so far have been found by setting lr relatively low and having batch size very low. In effect we are taking a very large amount of small steps towards
#   the minimizer, but I wonder if increasing the number of training epochs with a lower gamma and higher batch size may have the same effect but faster. Note: by the end of the first
#   epoch with very small batch size, the training error is typically already around 1% and validation error less than 2%
#   I have also noticed that training error will very quickly go down to 0%, and once there, the validation accuracy barely changes. 
#   I also wonder if decreasing the gamma a bit with small batch size and lr may allow us to reach a lower minimum, since I've seen the validation error reach a lower value but then
#   increase and settle at a slightly higher value later in training
# Tried decreasing gamma on lower batch size, best model is batch_size 4, lr .005, gamma .9, epochs per early stop check 2, with val accuracy of 0.94969%
# Trying to get results comparable to low batch size with a batch size of 256, by varying lr, gamma, and step size
#   While trying to do the above, we were kicked out of google colab and unable to get access to a GPU the same day
#   When we went back the next day, we were unable to get the above results with as low of validation error as we initially could, but we were able to get a model with a more
#   reasonable batch size of 128 that had validation accuracy almost as low as the other one. Since we knew that we could get kicked out of google colab again with little time
#   left to finish the project, we decided to stick with this as our final model
parameters={"batch_size": 2**8, "lr": .2, "gamma":.9, "step_size": 2, "momentum": 0.9, "weight_decay": 0, "nesterov": True, "epochs_per_early_stop_check": 3}
net = Net()

train_net(net.to(device), parameters)

PATH = "/content/state_dicts/" + str(parameters) + ".pt"

torch.save(net.state_dict(), PATH)

net = Net()
net.load_state_dict(torch.load(PATH))

parameters={"batch_size": 2**8, "lr": 0.2, "gamma":0.9, "step_size": 2, "momentum": 0.9, "weight_decay": 0, "nesterov": True}
net = Net()
net.load_state_dict(torch.load("/content/state_dicts/" + str(parameters) + ".pt"))
acc_train, loss_train, acc_val, loss_val = train_val_acc_eval_f(net.cuda().eval())
print(f"\tTrain err: {100-acc_train*100:.5f}%, train loss: {loss_train}")
print(f"\tValidation err: {100-acc_val*100:.5f}%, validation loss: {loss_val}\n")

train_test_acc_eval_f(net)

"""## Compression using the LC toolkit
### Step 1: L step
We will use same L step with same hyperparamters for all our compression examples
"""

# l_step_parameters: keep parameters 
#   learning rate: 0.7*(0.98**step), 
#   epochs per step: 7, 

# quantization parameters: codebook size (k) - powers of 2
# low rank parameters: alpha - higher alpha gives greater compression

# running each type of compression with the l-step parameters as they initially were in the demo gave good results for each compression type, so we decided to keep 
# the l-step parameters as they are and focus on seeing the impact of the mu schedule

def my_l_step(model, lc_penalty, step):
    train_loader, val_loader, test_loader = data_loader()
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    # ------------------- Learning rate parameter
    lr = 0.7*(0.98**step)
    # -------------------------------------------
    optimizer = optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    print(f'L-step #{step} with lr: {lr:.5f}')
    epochs_per_step_ = 7
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

"""### Step 2: Schedule of mu values"""

# mu schedule parameters: mu value should start very low and increase
#   first number is a really small number, demo had it at 9e-5
#   second number is some factor greater than 1 by which the first number should increase on each iteration, demo had it at 1.1
#   result will increase by a certain percentage (1 - second number) compounding on each step
#   third parameter is the number of step, demo had it at 20

# mu_s = [9e-5 * (1.1 ** n) for n in range(20)] test errors: pruning: 1.23%, quantization: 2.37%, low-rank: 1.33%

# First thing I would try is to increase the number of steps, to see if iterating longer leads to better results
#   This will come at the cost of extra computation time, so if the effect is small, we should keep it as it is
#   Looking at the initial run with default mu schedule, it looks like the quanization error was actually lower for earlier 
#   iteration, and happened to be a bit higher for the final iteration (test accuracy, not train accuracy)
#   Maybe try fewer iterations first, see if result is similar to first run and save some time by using fewer iterations
#
#     mu_s = [9e-5 * (1.1 ** n) for n in range(17)] test errors: pruning: 1.45%, quantization: 1.45%, low-rank: 1.43%
#         Error on quantization was lower by nearly a percent, slightly higher on pruning and low rank
#         I'm going to see if we can get lower error on quantization by letting it go for a few more iterations in order to keep the better error rates on the other two

#     mu_s = [9e-5 * (1.1 ** n) for n in range(23)] test errors: pruning: 1.21%, quantization: 1.31%, low-rank: 1.41%
#         This is an improvement on pruning and quantization

#     We decided to select a range of values that seemed reasonable for each parameter, then go through them one by one (keeping the values of the other parameters fixed)
#       and compress with each compression type to compare error rates for each value. We then select the best value out of those that we tried and repeat with the next 
#       parameter. Once we have the best values for the first two parameters, we will set those and see if increasing the number of iterations improves test error.
#     9e-3, 9e-5, 9e-6, 9e-7
#     mu_s = [9e-5 * (1.1 ** n) for n in range(17)] test errors: pruning: 1.45%, quantization: 1.45%, low-rank: 1.43%
#     mu_s = [9e-6 * (1.1  n) for n in range(20)] test errors: pruning: 3.05% , quantization: 5.69%, low-rank: 72.80%
#     mu_s = [9e-7 * (1.1 ** n) for n in range(20)] test errors: pruning: 4.17%, quantization: 5.89%, low-rank: 78.90%
#     mu_s = [9e-7 * (1.1 ** n) for n in range(20)] test errors: pruning: 4.17%, quantization: 5.89%, low-rank: 78.90%

#     1.01, 1.05, 1.1, 1.2
#     mu_s = [9e-5 * (1.05 ** n) for n in range(20)] test errors: pruning: 1.47%, quantization: 1.47%, low-rank: 5.85%
#     mu_s = [9e-5 * (1.1 ** n) for n in range(20)] test errors: pruning: 1.23%, quantization: 2.37%, low-rank: 1.33%
#     mu_s = [9e-5 * (1.2 ** n) for n in range(20)] test errors: pruning: 2.97%, quantization: 10.33%, low-rank: 79.96%

#     Number of iteration:
#       17, 20, 23, 25, 30
#     mu_s = [9e-5 * (1.1 ** n) for n in range(17)] test errors: pruning: 1.45%, quantization: 1.45%, low-rank: 1.43%
#     mu_s = [9e-5 * (1.1 ** n) for n in range(20)] test errors: pruning: 1.23%, quantization: 2.37%, low-rank: 1.33%
#     mu_s = [9e-5 * (1.1 ** n) for n in range(23)] test errors: pruning: 1.21%, quantization: 1.31%, low-rank: 1.41%
#     mu_s = [9e-5 * (1.1  n) for n in range(25)] test errors: pruning: 1.17%, quantization: 1.62%, low-rank: 1.41%
#     mu_s = [9e-5 * (1.1 ** n) for n in range(30)]
#     mu_s = [9e-5 * (1.1 ** n) for n in range(35)]
#     mu_s = [9e-5 * (1.1 ** n) for n in range(40)] test errors: pruning: 1.19%, quantization: 1.43%, low-rank: 1.37%

mu_s = [9e-5 * (1.1 ** n) for n in range(23)] # 0 to infinity
# 20 L-C steps in total
# total training epochs is 7 x 20 = 140

"""### Compression time! Pruning
Let us prune all but 5% of the weights in the network (5% = 13310 weights)
"""

# pruning parameter: kappa - controls how many weights will be in the compressed model (excluding biases)

parameters={'batch_size': 256, 'lr': 0.2, 'gamma': 0.9, 'step_size': 2, 'momentum': 0.9, 'weight_decay': 0, 'nesterov': True}
net = Net().cuda()
net.load_state_dict(torch.load("/content/state_dicts/" + str(parameters) + ".pt"))


layers = [lambda x=x: getattr(x, 'weight') for x in net.modules() if isinstance(x, nn.Linear)]
compression_tasks = {
    Param(layers, device): (AsVector, ConstraintL0Pruning(
            kappa=13285
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

"""### Quantization
Now let us quantize each layer with its own codebook
"""

parameters={'batch_size': 256, 'lr': 0.2, 'gamma': 0.9, 'step_size': 2, 'momentum': 0.9, 'weight_decay': 0, 'nesterov': True}
net = Net().cuda()
net.load_state_dict(torch.load("/content/state_dicts/" + str(parameters) + ".pt"))

layers = [lambda x=x: getattr(x, 'weight') for x in net.modules() if isinstance(x, nn.Linear)]
# k = 2 for each layer gives 30x compression
# k = 4 for each layer gives 15x compression
# k = 8 for each layer gives 10x compression
# k = 16 for each layer gives about 8x compression
# k = 32 for each layer gives about 6.2x compression
# k = 64 for each layer gives about 5.28x compresssion
# k = 128 for each layer gives about 4.5x compression

compression_tasks = {
    Param(layers[0], device): (AsVector, AdaptiveQuantization(k=2), 'layer0_quant'),
    Param(layers[1], device): (AsVector, AdaptiveQuantization(k=2), 'layer1_quant'),
    Param(layers[2], device): (AsVector, AdaptiveQuantization(k=2), 'layer2_quant')
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

"""### Low-rank compression with automatic rank selection"""

parameters={'batch_size': 256, 'lr': 0.2, 'gamma': 0.9, 'step_size': 2, 'momentum': 0.9, 'weight_decay': 0, 'nesterov': True}
net = Net().cuda()
net.load_state_dict(torch.load("/content/state_dicts/" + str(parameters) + ".pt"))

layers = [lambda x=x: getattr(x, 'weight') for x in net.modules() if isinstance(x, nn.Linear)]
# ----------------- alpha - compresssion parameter
alpha=1e-9
alpha=1e-3
# ------------------------------------------------
compression_tasks = {
    Param(layers[0], device): (AsIs, RankSelection(conv_scheme='scheme_1', alpha=alpha, criterion='storage', module=layers[0], normalize=True), "layer1_lr"),
    Param(layers[1], device): (AsIs, RankSelection(conv_scheme='scheme_1', alpha=alpha, criterion='storage', module=layers[1], normalize=True), "layer2_lr"),
    Param(layers[2], device): (AsIs, RankSelection(conv_scheme='scheme_1', alpha=alpha, criterion='storage', module=layers[2], normalize=True), "layer3_lr")
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

"""## Plots"""

# Pruning
train_errs_p = [0, 0, 0, 0, 0, 0, 0, 0]
test_errs_p = [.0125, .0125, .0121, .0121, .0123, .0119, .0117, .0133]
params_p = [46250, 36700, 26700, 16700, 13000, 10000, 8350, 6700]
ratios_p = [5.001867911627809, 6.207001113787051, 8.354945054945055, 12.976122810038508, 16.31688875092455, 
            20.78356715383339, 24.467737857159275, 29.934964951381907]

# Quantization
train_errs_q = [0, 0, 0, 0, 0, 0]
test_errs_q = [.0131, .0131, .0131, .0133, .0137, .0145]
params_q = [265892, 265796, 265748, 265724, 265712, 265706]
codebook_size = [64, 32, 16, 8, 4, 2]
ratios_q = [5.278211669964247, 6.333326391636644, 7.904382825147406, 10.50205468977391, 15.631856431644955, 
            30.537202530374536]

# Low-Rank
train_errs_l = [0, 0, 0, 0, .7845]
test_errs_l = [.0125, .0137, .0137, .0135, .7898]
params_l = [51625, 30037, 23869, 16512, 7840]
ratios_l = [5.114453200076879, 8.7413770448722164, 10.962552525335756, 15.730034876160076, 32.274711946634326]

train_errs = [train_errs_p, train_errs_q, train_errs_l]
test_errs = [test_errs_p, test_errs_q, test_errs_l]

params = [params_p, params_q, params_l]
ratios = [ratios_p, ratios_q, ratios_l]

compression_type = ["Pruning", "Quantization", "Low-Rank"]

x_axis = [params, ratios]
x_axis_labels = ["Number of Compressed Parameters", "Compression Ratio"]
y_axis = [train_errs, test_errs]
y_axis_labels = ["Train", "Test"]
legend = {"Pruning": "bo-", "Quantization": "ro-", "Low-Rank": "go-"}

from matplotlib import pyplot as plt
# hline - reference network error
x_max = [300000, 32]
train_test_errs = [0, .0133]
for curr_subset in range(2): # Train and test
    for curr_x in range(2): # Compressed Params and Compression Ratio
        fig = plt.figure()
        ax = plt.gca()
        ax.hlines(y=train_test_errs[curr_subset], xmin=0, xmax=x_max[curr_x], color='k', linestyles='dashed', label="Uncompressed Reference Network")
        for curr_type in range(3):
            ax.plot(x_axis[curr_x][curr_type], y_axis[curr_subset][curr_type], legend[compression_type[curr_type]], label=compression_type[curr_type])
            ax.set_xlabel(x_axis_labels[curr_x])
            ax.set_title(y_axis_labels[curr_subset] + ' Error vs. ' + x_axis_labels[curr_x])
            ax.set_ylabel(y_axis_labels[curr_subset] + ' Error')
            ax.set_xscale('log')
            ax.set_yscale('logit')
        ax.legend()

for curr_subset, errs in enumerate([train_errs_q, test_errs_q]):
    fig = plt.figure()
    ax = plt.gca()
    ax.hlines(y=train_test_errs[curr_subset], xmin=0, xmax=65, color='k', linestyles='dashed', label="Uncompressed Reference Network")
    ax.plot(codebook_size, errs, 'ro-', label="Quantization")
    ax.set_xlabel("Codebook Size")
    ax.set_title(y_axis_labels[curr_subset] + ' Error vs. Codebook Size (Quantization)')
    ax.set_ylabel(y_axis_labels[curr_subset] + ' Error')
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.legend()

from sklearn.metrics import confusion_matrix

net = Net().cuda()
net.load_state_dict(torch.load("model.pt"))

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