import torch
import numpy as np
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.autograd import Variable
from mnist_read import read_mnist
import matplotlib.pyplot as plt


class TrainDataset(Dataset):
    def __init__(self):
        mnist_data = read_mnist("data")
        self.train_data = torch.from_numpy(np.float32(mnist_data.train_data))
        self.train_label = torch.from_numpy(np.reshape(mnist_data.train_label, [len(mnist_data.train_label), ]))
        pass

    def __getitem__(self, item):
        return self.train_data[item, :], self.train_label[item]

    def __len__(self):
        return self.train_data.shape[0]


class TestDataset(Dataset):
    def __init__(self):
        mnist_data = read_mnist("data")
        self.test_data = torch.from_numpy(np.float32(mnist_data.test_data))
        self.test_label = torch.from_numpy(np.reshape(mnist_data.test_label, [len(mnist_data.test_label), ]))
        pass

    def __getitem__(self, item):
        return self.test_data[item, :], self.test_label[item]

    def __len__(self):
        return self.test_data.shape[0]


def main():
    train_dataset = TrainDataset()
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_dataset = TestDataset()
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=True)
    net = torch.nn.Sequential(
        torch.nn.BatchNorm1d(784),
        torch.nn.Linear(784, 200),
        torch.nn.Tanh(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(200, 10),
        torch.nn.Tanh(),
        torch.nn.LogSoftmax(dim=1)
    )
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    print(net)
    for epoch in range(10):
        net.train()
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            optimizer.zero_grad()
            output = net(inputs)
            loss = functional.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            # print("Epoch %d, %d: loss = %.4f" % (epoch, i, loss.item()))

        net.eval()
        for i, data in enumerate(test_dataloader):
            inputs, labels = data
            output = net(inputs)
            y_pred = output.argmax(dim=1, keepdim=True)
            correct = y_pred.eq(labels.view_as(y_pred)).sum().item()
            print("Epoch %d,%d: loss %.6f, %d / %d" % (epoch, i, loss.item(), correct, labels.shape[0]))

    torch.save(net, "model/net.pkl")
    #
    # tensor_x = torch.from_numpy(np.float32(mnist_data.test_data[0:1, :]))
    # test_x = Variable(tensor_x).to("cpu")
    # y_prediction = net(test_x)
    # print(y_prediction.argmax(dim=1, keepdim=True))
    # im_data = np.reshape(mnist_data.test_data[0, :], [28, 28])
    # plt.imshow(im_data)
    # plt.show()


if __name__ == "__main__":
    main()
