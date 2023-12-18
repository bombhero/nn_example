import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from mnist_read import read_mnist


def build_model(verbose=0):
    input_data = keras.Input(shape=(784,), name="Input")
    x = keras.layers.BatchNormalization()(input_data)
    y = keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs=input_data, outputs=y, name="MnistModel")

    model.summary()

    model.compile(loss=keras.metrics.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.01),
                  metrics=['accuracy'])
    return model


def draw_number(ax, x):
    matrix = np.reshape(x, [28, 28])
    ax.imshow(matrix)


def trans_to_one_shot(X):
    rows_num = len(X)
    one_x = np.zeros([rows_num, 10])
    for i in range(rows_num):
        one_x[i, X[i]] = 1
    return one_x


def main():
    mnist = read_mnist("data")
    nn_model = build_model(0)

    ts = time.time()
    for i in range(1):
        # print("%d round." % i)
        batch_x, batch_y = mnist.train_data, mnist.train_label
        one_y = trans_to_one_shot(batch_y)

        batch_size = 256
        nb_epoch = 5

        hist = nn_model.fit(batch_x, one_y, batch_size=batch_size, epochs=nb_epoch, verbose=1)
        new_ts = time.time()
        spend_time = new_ts - ts

        test_x, test_y = mnist.test_data, mnist.test_label
        test_y = trans_to_one_shot(test_y)
        loss, acc = nn_model.evaluate(test_x, test_y, batch_size=256, verbose=1)
        print('Test loss {:>.3f}, Accuracy {:.3f}'.format(loss, acc))


if __name__ == "__main__":
    main()
