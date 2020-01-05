import torch
import math
import random
import numpy as np
import matplotlib.pyplot as plt


class TrainDataSet(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __getitem__(self, item):
        position = float(item) / 100
        x = math.sin(math.pi * position)
        y = math.cos(math.pi * position) + random.uniform(-0.1, 0.1)
        return x, y

    def __len__(self):
        return 100000


class DemoRNN(torch.nn.Module):
    def __init__(self):
        super(DemoRNN, self).__init__()
        self.rnn = torch.nn.RNN(input_size=1, hidden_size=10, num_layers=1, batch_first=True)
        self.output_layer = torch.nn.Linear(in_features=10, out_features=1)

    def forward(self, x, hidden):
        output, h_out = self.rnn(x, hidden)
        linear_input = output[0, :, :]
        output = self.output_layer(linear_input)
        return output, h_out


def main():
    train_data = TrainDataSet()
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=1000, shuffle=False)
    nn_model = DemoRNN()
    print(nn_model)
    optimize = torch.optim.Adam(nn_model.parameters(), lr=0.01)
    loss_func = torch.nn.L1Loss()
    h_output = None

    plt.ion()
    plt.show()
    plt.cla()
    for i, data in enumerate(train_loader):
        plt.cla()
        x, y = data
        x = torch.from_numpy(np.float32(x.numpy()[np.newaxis, :, np.newaxis]))
        y = torch.from_numpy(np.float32(y.numpy()[:, np.newaxis]))
        row_idx = np.arange(0, x.shape[1], 1)
        x_array = x.numpy()[0, :, 0]
        y_array = y.numpy()[:, 0]
        nn_model.train()
        optimize.zero_grad()
        output, h_output = nn_model(x, h_output)
        h_output = h_output.data
        loss = loss_func(output, y)
        loss.backward()
        optimize.step()

        nn_model.eval()
        predict_y, h = nn_model(x, None)
        predict_y = predict_y.cpu().detach().numpy()[:, 0]

        plt.cla()
        plt.plot(row_idx, x_array, "b-")
        plt.plot(row_idx, y_array, "g-")
        plt.plot(row_idx, predict_y, "r-")
        plt.pause(0.1)


if __name__ == "__main__":
    main()
