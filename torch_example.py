import torch
import numpy as np
import torch.nn.functional as functional
from torch.autograd import Variable
from mnist_read import read_mnist
import matplotlib.pyplot as plt


class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.norm_layer = torch.nn.BatchNorm1d(n_input)
        self.hidden_layer = torch.nn.Linear(n_input, n_hidden)
        self.output_layer = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.norm_layer(x)
        x = functional.tanh(self.hidden_layer(x))
        y_prediction = functional.log_softmax(self.output_layer(x))
        return y_prediction


def trans_to_one_shot(label_x):
    rows_num = label_x.shape[0]
    one_x = np.zeros([rows_num, 10])
    for i in range(rows_num):
        one_x[i, int(label_x[i, 0])] = 1
    return one_x


def main():
    mnist_data = read_mnist("data")
    tensor_x = torch.from_numpy(np.float32(mnist_data.train_data))
    tensor_y = torch.from_numpy(np.reshape(mnist_data.train_label, [60000, ]))
    train_x, train_y = Variable(tensor_x).to("cpu"), Variable(tensor_y).to("cpu")
    # net = Net(784, 200, 10)
    net = torch.nn.Sequential(
        torch.nn.BatchNorm1d(784),
        torch.nn.Linear(784, 200),
        torch.nn.Tanh(),
        torch.nn.Linear(200, 10),
        torch.nn.LogSoftmax(dim=1)
    )
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    print(net)
    for i in range(100):
        net.train()
        optimizer.zero_grad()
        output = net(train_x)
        loss = functional.nll_loss(output, train_y)
        loss.backward()
        optimizer.step()

        net.eval()
        test_x = Variable(torch.from_numpy(np.float32(mnist_data.test_data)))
        test_y = Variable(torch.from_numpy(np.reshape(mnist_data.test_label, [10000, ])))
        output = net(test_x)
        y_pred = output.argmax(dim=1, keepdim=True)
        correct = y_pred.eq(test_y.view_as(y_pred)).sum().item()

        print("Epoch %d, loss %.6f, %d / %d" % (i, loss.item(), correct, test_x.shape[0]))

    tensor_x = torch.from_numpy(np.float32(mnist_data.test_data[0:1, :]))
    test_x = Variable(tensor_x).to("cpu")
    y_prediction = net(test_x)
    print(y_prediction.argmax(dim=1, keepdim=True))
    im_data = np.reshape(mnist_data.test_data[0, :], [28, 28])
    plt.imshow(im_data)
    plt.show()


if __name__ == "__main__":
    main()
