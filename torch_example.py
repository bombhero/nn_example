import torch
import time
import random
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
    calc_device = "cuda"
    mnist_data = read_mnist("data")
    tensor_x = torch.from_numpy(np.float32(mnist_data.train_data))
    tensor_y = torch.from_numpy(np.reshape(mnist_data.train_label, [60000, ]))
    train_x, train_y = Variable(tensor_x).to(calc_device), Variable(tensor_y).to(calc_device)
    # net = Net(784, 200, 10)
    net = torch.nn.Sequential(
        torch.nn.BatchNorm1d(784),
        torch.nn.Linear(784, 200),
        torch.nn.Tanh(),
        torch.nn.Linear(200, 10),
        torch.nn.LogSoftmax(dim=1)
    ).to(calc_device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    print(net)
    for i in range(1000):
        start_ts = time.time()
        net.train()
        optimizer.zero_grad()
        output = net(train_x)
        loss = functional.nll_loss(output, train_y)
        loss.backward()
        optimizer.step()

        net.eval()
        test_x = Variable(torch.from_numpy(np.float32(mnist_data.test_data))).to(calc_device)
        test_y = Variable(torch.from_numpy(np.reshape(mnist_data.test_label, [10000, ]))).to(calc_device)
        output = net(test_x)
        y_pred = output.argmax(dim=1, keepdim=True)
        correct = y_pred.eq(test_y.view_as(y_pred)).sum().item()

        end_ts = time.time()
        spent_time = end_ts - start_ts
        print("Epoch %d, spent %.2f loss %.6f, %d / %d" % (i, spent_time, loss.item(), correct, test_x.shape[0]))

    idx = random.randint(0, 1000)
    print('idx = {}'.format(idx))
    tensor_x = torch.from_numpy(np.float32(mnist_data.test_data[idx:(idx+1), :]))
    y = mnist_data.test_label[idx, :]
    test_x = Variable(tensor_x).to(calc_device)
    y_prediction = net(test_x)
    print('Test: y_pred={}, y={}'.format(y_prediction.argmax(dim=1, keepdim=True), y))
    torch.save(net, 'tmp/torch_example.pkl')

    verify_net = torch.load('tmp/torch_example.pkl')
    verify_net.to(calc_device)
    verify_net.eval()
    y_pred = verify_net(test_x)
    print('Verify: y_pred={}, y={}'.format(y_pred.argmax(dim=1, keepdim=True), y))

    im_data = np.reshape(mnist_data.test_data[idx, :], [28, 28])
    plt.imshow(im_data)
    plt.show()


if __name__ == "__main__":
    main()
