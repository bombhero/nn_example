import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt


class MnistData:
    def __init__(self, path):
        self.train_image_file = path + "/train-images-idx3-ubyte.gz"
        self.train_label_file = path + "/train-labels-idx1-ubyte.gz"
        self.test_image_file = path + "/t10k-images-idx3-ubyte.gz"
        self.test_label_file = path + "/t10k-labels-idx1-ubyte.gz"
        self.train_data = self._read_image_file(self.train_image_file)
        self.train_label = self._read_label_file(self.train_label_file)
        self.test_data = self._read_image_file(self.test_image_file)
        self.test_label = self._read_label_file(self.test_label_file)

    @staticmethod
    def _read_image_file(gz_file):
        fid = gzip.open(gz_file)
        magic_number = struct.unpack("!i", fid.read(4))[0]
        image_count = struct.unpack("!i", fid.read(4))[0]
        data = fid.read(4)
        image_width = struct.unpack("!i", data)[0]
        data = fid.read(4)
        image_height = struct.unpack("!i", data)[0]
        image_size = image_width * image_height
        image_x = np.zeros([image_count, image_size])
        for i in range(image_count):
            print("Reading image %d" % i, end="\r")
            data = fid.read(image_size)
            new_line = np.array([list(data)])
            image_x[i, :] = new_line
        fid.close()
        print("")
        return image_x

    @staticmethod
    def _read_label_file(gz_file):
        fid = gzip.open(gz_file)
        magic_number = struct.unpack("!i", fid.read(4))[0]
        label_count = struct.unpack("!i", fid.read(4))[0]
        label_x = np.zeros([label_count, 1], dtype="int64")
        for i in range(label_count):
            print("Reading label %d" % i, end="\r")
            data = fid.read(1)
            label_x[i, 0] = int(data[0])
        fid.close()
        print("")
        return label_x


def read_mnist(path):
    return MnistData(path)


if __name__ == "__main__":
    mnist_data = read_mnist("data")
    print(mnist_data.train_label[0, 0])
    number = mnist_data.train_data[0, :]
    img = np.reshape(number, [28, 28])
    plt.imshow(img)
    plt.show()

