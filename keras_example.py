import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from mnist_read import read_mnist
from tensorflow.keras import layers


def build_model(verbose=0):
    input_data = tf.keras.Input(shape=(784,), name="Input")
    x = layers.BatchNormalization()(input_data)
    y = layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs=input_data, outputs=y, name="MnistModel")

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=0.01, decay=0.01),
                  metrics=['accuracy'])
    return model


def draw_number(ax, X):
    matrix = np.reshape(X, [28, 28])
    ax.imshow(matrix)


def trans_to_one_shot(X):
    rows_num = len(X)
    oneX = np.zeros([rows_num, 10])
    for i in range(rows_num):
        oneX[i, X[i]] = 1
    return oneX


def main():
    mnist = read_mnist("data")
    nn_model = build_model(0)

    ts = time.time()
    for i in range(1):
        # print("%d round." % i)
        batch_x, batch_y = mnist.train_data, mnist.train_label
        one_y = trans_to_one_shot(batch_y)

        batch_size = batch_x.shape[0]
        nb_epoch = 5

        # early_stopping = EarlyStopping(monitor='loss', patience=2)
        hist = nn_model.fit(batch_x, one_y, batch_size=batch_size, epochs=nb_epoch)
        new_ts = time.time()
        spend_time = new_ts - ts

        test_x, test_y = mnist.test_data, mnist.test_label
        predict_y = np.argmax(nn_model.predict(test_x), axis=1)
        correct = (test_y.reshape([test_y.shape[0]]) == predict_y).sum()
        print("{}/{}".format(correct, test_y.shape[0]))


if __name__ == "__main__":
    main()
