import torch
import numpy as np
import torch.nn.functional as functional
import cv2
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.autograd import Variable
from mnist_read import read_mnist
import matplotlib.pyplot as plt


class TrainDataset(Dataset):
    def __init__(self):
        mnist_data = read_mnist("data")
        self.train_data = np.uint8(mnist_data.train_data)
        self.train_label = np.reshape(mnist_data.train_label, [len(mnist_data.train_label), ])
        pass

    def __getitem__(self, item):
        out_data = np.reshape(self.train_data[item, :], [28, 28])
        ret, out_data = cv2.threshold(out_data, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        out_data = cv2.bitwise_not(out_data)
        out_data = np.array([out_data / 255.0])
        out_label = np.array(self.train_label[item])
        return Variable(torch.from_numpy(np.float32(out_data))), Variable(torch.from_numpy(out_label))

    def __len__(self):
        return self.train_data.shape[0]


class TestDataset(Dataset):
    def __init__(self):
        mnist_data = read_mnist("data")
        self.test_data = np.uint8(mnist_data.test_data)
        self.test_label = np.reshape(mnist_data.test_label, [len(mnist_data.test_label), ])

    def __getitem__(self, item):
        out_data = np.reshape(self.test_data[item, :], [28, 28])
        ret, out_data = cv2.threshold(out_data, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        out_data = cv2.bitwise_not(out_data)
        out_data = np.array([out_data / 255.0])
        out_label = np.array(self.test_label[item])
        return Variable(torch.from_numpy(np.float32(out_data))), Variable(torch.from_numpy(out_label))

    def __len__(self):
        return self.test_data.shape[0]


class KaggleCNN(torch.nn.Module):
    def __init__(self):
        super(KaggleCNN, self).__init__()
        self.conv_l1 = torch.nn.Sequential(
            # output: 64 * 28 *28
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.ReLU(),
            # output: 64 * 14 * 14
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv_l2 = torch.nn.Sequential(
            # output: 128 * 14 * 14
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU(),
            # output: 128 * 7 * 7
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.linear_l1 = torch.nn.Linear(32 * 7 * 7, 10)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv_l1(x)
        x = self.conv_l2(x)
        x = x.view(x.size(0), -1)
        x = self.linear_l1(x)
        x = self.log_softmax(x)
        return x


def build_first_nn(device):
    net = KaggleCNN().to(device)
    return net


def sum_list(li_number):
    sum_number = 0
    for num in li_number:
        sum_number += num
    return sum_number


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_dataset = TrainDataset()
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=2048, shuffle=True)
    test_dataset = TestDataset()
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=2048, shuffle=True)
    net = build_first_nn(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    print(net)
    loss_record = []
    early_stop_count = 0
    for i in range(10000):
        net.train()
        loss_val = 0
        for j, data in enumerate(train_dataloader):
            train_x, train_y = data
            optimizer.zero_grad()
            output = net(train_x.to(device))
            loss = functional.nll_loss(output, train_y.to(device))
            loss_val += loss.data.cpu()
            # print("{}:{}:{}".format(i, j, loss.data.cpu()))
            loss.backward()
            optimizer.step()
        print("{}:{}".format(i, loss_val))
        if i > 10:
            loss_record.append(loss_val)
            if len(loss_record) > 10:
                del (loss_record[0])
                if loss_val > (sum_list(loss_record) / len(loss_record)):
                    early_stop_count += 1
                    print("Early stop ready: count=%d, current_loss=%.8f, average=%.8f" %
                          (early_stop_count, loss_val, (sum_list(loss_record) / len(loss_record))))
                if early_stop_count > 1 and loss_val < (sum_list(loss_record) / len(loss_record)):
                    print("Early stop done: current_loss=%.8f, average=%.8f" %
                          (loss_val, (sum_list(loss_record) / len(loss_record))))
                    break

    result_y = None
    net.eval()
    correct = 0
    total = 0
    for i, data in enumerate(test_dataloader):
        test_x, test_y = data
        tensor_x = test_x.to(device)
        predict_y = np.argmax(net(tensor_x).cpu().detach().numpy(), axis=1)
        correct += predict_y.eq(test_y.view_as(predict_y)).sum().item()
        total += test_y.shape[0]
        print("Testing {}/{}".format(correct, total))


if __name__ == "__main__":
    main()