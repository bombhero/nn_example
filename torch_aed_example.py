import torch
import numpy as np
from torch.autograd import Variable
from mnist_read import read_mnist
import matplotlib.pyplot as plt


class AedNet(torch.nn.Module):
    def __init__(self, n_input, n_middle):
        super(AedNet, self).__init__()
        self.norm_layer = torch.nn.BatchNorm1d(n_input)
        self.encode_layer = torch.nn.Linear(in_features=n_input, out_features=n_middle)
        self.encode_active = torch.nn.Tanh()
        self.decode_layer = torch.nn.Linear(in_features=n_middle, out_features=n_input)

    def forward(self, x):
        x = self.norm_layer(x)
        middle_output = self.encode_active(self.encode_layer(x))
        y_prediction = self.decode_layer(middle_output)
        return y_prediction, x


def main():
    mnist_data = read_mnist("data")
    train_x = Variable(torch.from_numpy(np.float32(mnist_data.train_data))).to("cpu")
    net = AedNet(784, 100)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()
    print(net)
    fig = plt.figure()
    ax_l = fig.add_subplot(1, 2, 1)
    ax_r = fig.add_subplot(1, 2, 2)
    for _ in range(20):
        for i in range(10):
            net.train()
            optimizer.zero_grad()
            y_pred, x = net(train_x)
            loss = loss_func(y_pred, x)
            loss.backward()
            optimizer.step()
            print("%d: loss = %f" % (i, loss.item()))

        rand_line = np.random.randint(0, 10000)
        net.eval()
        test_x = Variable(torch.from_numpy(np.float32(mnist_data.test_data[rand_line:(rand_line+1), :]))).to("cpu")
        y_prediction, x_output = net(test_x)
        matrix_l = torch.reshape(y_prediction, [28, 28]).data.numpy()
        matrix_r = torch.reshape(x_output, [28, 28]).data.numpy()
        ax_l.imshow(matrix_l)
        ax_r.imshow(matrix_r)
        plt.ion()
        plt.show()
        plt.pause(1)


if __name__ == "__main__":
    main()
