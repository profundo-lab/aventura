from common import print_now, determine_working_root
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


class DigitsData(Dataset):

    def __init__(self, data, labels, transform = None):
        # loading data into memory object
        self.x = torch.as_tensor(data).float()
        self.y = torch.tensor(labels, dtype=torch.int64)
        self.x = self.x.reshape(data.shape[0], 1, 28,28)
        self.n_samples = len(data)

    def __len__(self):
        # return the number of elements in the source data
        return self.n_samples

    def __getitem__(self, index):
        assert index >= 0 and index < self.n_samples, \
            print(f"index must be within the range: 0 <= index < len(data)")
        # data, label
        return (self.x[index], self.y[index])


def train_epoch(model, data_loader,
    data_size, loss_function, optimizer, device='cpu'):

    running_loss = 0.
    running_corrects = 0
    for batch_idx, (data, labels) in enumerate(data_loader):
        data = data.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(data)

        optimizer.zero_grad()
        loss = loss_function(outputs, labels)

        _, predictions = torch.max(outputs.data, 1)       # compute accuracy
        running_corrects += torch.sum(predictions == labels).item()
        running_loss += loss.item() * data.size(0)

        # Backpropagation and perform optimisation
        loss.backward()
        optimizer.step()

        total_loss = running_loss / data_size
        total_acc = 100. * running_corrects / data_size

    return total_acc, total_loss



def examine(model, data_loader, data_size, loss_function, device='cpu'):

    test_loss = 0.
    test_corrects = 0

    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.reshape(-1, 28*28).to(device), labels.to(device)
            outputs = model(data)
            test_loss += loss_function(outputs, labels).item() * data.size(0)
            _, predictions = torch.max(outputs.data, 1)
            test_corrects += torch.sum(predictions == labels).item()

    # print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
    #         test_loss / data_size, test_corrects, data_size,
    #         100. * test_corrects / data_size))

    return 100. * test_corrects / data_size, test_loss / data_size


def prepare_mnist_data(train_batch_size=100, test_batch_size=100):

    working_root = determine_working_root('profundo')
    mnist_pickle = os.path.join(working_root, 'mnist source', 'float_mnist.pkl')

    train_data, train_label, test_data, test_label = \
        pickle.load(open(mnist_pickle, 'rb'))

    train_set = DigitsData(train_data, train_label)
    test_set = DigitsData(test_data, test_label)
    train_loader = DataLoader(train_set, batch_size = train_batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size = test_batch_size, shuffle=False)
    return train_loader, len(train_set), test_loader, len(test_set)


input_size = 28 * 28
output_size = 10
hidden_size = 500
num_classes = 10

class SimpleMLN(nn.Module):

    def __init__(self):
        super(SimpleMLN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class FeedForwardNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        # return x # no activation at the end
        # return F.log_softmax(x, dim = 1)


def show_result(train_acc, test_acc, train_loss, test_loss):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi = 120)

    x_range = np.arange(0, len(train_acc), 1)

    ax[0].plot(x_range, train_acc, c = 'navy', label='train set')
    ax[0].plot(x_range, test_acc, c = 'brown', label='test set accuracy after each train epoch')
    ax[0].set_ylabel('accuracy (%)')
    ax[0].set_xlabel('numer of epochs')
    ax[0].set_xticks(x_range)
    ax[0].set_xticklabels([str(i+1) for i in x_range])
    ax[0].legend(loc='best', frameon=True, shadow=True, facecolor='#dddddd')
    ax[1].plot(x_range, train_loss, c = 'navy', label='train loss after each epch')
    ax[1].plot(x_range, test_loss, c = 'brown', label='test loss')
    ax[1].set_ylabel('epoch loss')
    ax[1].set_xlabel('numer of epochs')
    ax[1].set_xticks(x_range)
    ax[1].set_xticklabels([str(i+1) for i in x_range])
    ax[1].legend(loc='best', frameon=True, shadow=True, facecolor='#dddddd')
    plt.show()


def main():

    train_loader, train_size , test_loader, test_size = \
        prepare_mnist_data()

    num_epochs = 10
    model = FeedForwardNet()
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    acc_list = np.zeros(num_epochs)
    loss_list = np.zeros(num_epochs)
    test_acc = np.zeros(num_epochs)
    test_loss = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        acc_list[epoch], loss_list[epoch] = \
            train_epoch(
                model, train_loader, train_size, loss_fn, optimizer, device
            )
        test_acc[epoch], test_loss[epoch] = \
            examine(model, test_loader, test_size, loss_fn, device)

        print('epoch[{}/{}]: accuracy[{:.2f}% : {:.2f}%]'.format(epoch+1, num_epochs,
                acc_list[epoch], test_acc[epoch]))

    # show_result(acc_list, test_acc, loss_list, test_loss)


if __name__ == '__main__':
    main()
