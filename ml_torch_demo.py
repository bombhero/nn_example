import matplotlib.pyplot as plt
import numpy as np
import math
import random
import time
import torch

calc_device = 'cuda'


def generate_sample():
    x = np.arange(0, 7, 0.01)
    y = np.zeros([x.shape[0]])
    for i in range(x.shape[0]):
        y[i] = math.cos(x[i]) + random.uniform(-0.2, 0.2)
    return x.reshape((x.shape[0], 1)), y.reshape((y.shape[0], 1))


def generate_test_sample():
    x = np.arange(7, 8, 0.05)
    y = np.zeros([x.shape[0]])
    for i in range(x.shape[0]):
        y[i] = math.cos(x[i]) + random.uniform(-0.2, 0.2)
    return x.reshape((x.shape[0], 1)), y.reshape((y.shape[0], 1))


def build_sample_model():
    nn_model = torch.nn.Sequential(
        torch.nn.Linear(1, 1)
    )
    print(nn_model)
    return nn_model


def build_complex_model():
    nn_model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(1),
        torch.nn.Linear(1, 20),
        torch.nn.Tanh(),
        torch.nn.Linear(20, 1),
    )
    print(nn_model)
    return nn_model


def main():
    train_x, train_y = generate_sample()
    test_x, test_y = generate_test_sample()
    print(train_x.shape)
    print(train_y.shape)
    # nn_model = build_sample_model()
    nn_model = build_complex_model().to(calc_device)
    optimize = torch.optim.Adam(nn_model.parameters(), lr=0.01)
    loss_func = torch.nn.L1Loss()

    plt.ion()
    plt.show()
    plt.cla()
    plt.scatter(train_x, train_y, color="green")
    plt.scatter(test_x, test_y, color="orange")
    plt.pause(1)

    for i in range(3000):
        nn_model.train()
        optimize.zero_grad()
        output = nn_model(torch.from_numpy(np.float32(train_x)).to(calc_device))
        loss = loss_func(output, torch.from_numpy(np.float32(train_y)).to(calc_device))
        loss.backward()
        optimize.step()

        if i % 100 == 0:
            nn_model.eval()
            predict_y = nn_model(torch.from_numpy(np.float32(train_x)).to(calc_device)).cpu().detach().numpy()
            verify_y = nn_model(torch.from_numpy(np.float32(test_x)).to(calc_device)).cpu().detach().numpy()
            plt.cla()
            plt.scatter(train_x, train_y, color="green")
            plt.scatter(test_x, test_y, color="orange")
            plt.plot(train_x, predict_y, "b-")
            plt.plot(test_x, verify_y, "r-")
            plt.pause(0.1)
            print("i = {}".format(i))
    print("Done")
    time.sleep(15)


if __name__ == "__main__":
    main()
