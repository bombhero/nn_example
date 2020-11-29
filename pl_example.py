import torch
import time
import random
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from mnist_read import read_mnist
import matplotlib.pyplot as plt


class LiteNet(pl.LightningModule):
    def __init__(self, n_input, n_hidden, n_output):
        super(LiteNet, self).__init__()
        self.norm_layer = torch.nn.BatchNorm1d(n_input)
        self.hidden_layer = torch.nn.Linear(n_input, n_hidden)
        self.hidden_func = torch.nn.Tanh()
        self.output_layer = torch.nn.Linear(n_hidden, n_output)
        self.output_func = torch.nn.LogSoftmax(dim=1)
        self.loss_func = torch.nn.NLLLoss()

    def forward(self, x):
        x = self.norm_layer(x)
        x = self.hidden_func(self.hidden_layer(x))
        y_prediction = self.output_func(self.output_layer(x))
        return y_prediction

    def training_step(self, branch, branch_idx):
        x, y = branch
        output = self(x)
        loss = self.loss_func(output, y)
        self.log('train_loss', loss)
        return loss

    def test_step(self, branch, branch_idx):
        x, y = branch
        output = self(x)
        loss = self.loss_func(output, y)
        pred_y = torch.argmax(output, axis=1)
        pred = torch.sum(pred_y == y)
        self.log('val_loss', loss)
        return {'val_loss': loss, 'pred': pred}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class MnistDataSet(Dataset):
    def __init__(self, mnist, for_train=True):
        if for_train:
            self.data = torch.from_numpy(np.float32(mnist.train_data))
            self.label = torch.from_numpy(mnist.train_label[:, 0])
        else:
            self.data = torch.from_numpy(np.float32(mnist.test_data))
            self.label = torch.from_numpy(mnist.test_label[:, 0])

    def __getitem__(self, item):
        return self.data[item, :], self.label[item]

    def __len__(self):
        return self.data.shape[0]


def main():
    from pytorch_lightning.callbacks import EarlyStopping

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    mnist_data = read_mnist("data")
    lite_net = LiteNet(784, 200, 10)
    print(lite_net)
    train_dataset = MnistDataSet(mnist_data, for_train=True)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=500)
    test_dataset = MnistDataSet(mnist_data, for_train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))
    early_stop = EarlyStopping('train_loss')
    if device == 'cuda':
        trainer = pl.Trainer(gpus=1, callbacks=[early_stop])
    else:
        trainer = pl.Trainer(callbacks=[early_stop])
    start_ts = time.time()
    trainer.fit(lite_net, train_loader)
    end_ts = time.time()
    spend_time = end_ts - start_ts
    print('Spend %fs' % spend_time)
    trainer.test(lite_net, test_loader)
    idx = random.randint(0, (len(test_dataset)-1))
    test_x, test_y = test_dataset[idx]
    lite_net.eval()
    input_x = test_x.unsqueeze(0).to(device)
    pred_y = torch.argmax(lite_net(input_x), axis=1).cpu().detach().numpy()
    print('Test: pred_y = {}, y = {}'.format(pred_y, test_y))
    trainer.save_checkpoint('tmp/pl_example.ckpt')

    # verify_net = LiteNet(784, 200, 10).to(device)
    verify_net = LiteNet.load_from_checkpoint('tmp/pl_example.ckpt', n_input=784, n_hidden=200, n_output=10)
    verify_net.to(device)
    verify_net.eval()
    pred_y = torch.argmax(verify_net(input_x), axis=1).cpu().detach().numpy()
    print('Verify: pred_y = {}, y = {}'.format(pred_y, test_y))

    im_data = np.reshape(test_x.detach().numpy(), [28, 28])
    plt.imshow(im_data)
    plt.show()


if __name__ == "__main__":
    main()
