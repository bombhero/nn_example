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
    Matrix = np.reshape(X, [28, 28])
    ax.imshow(Matrix)


def trans_to_one_shot(X):
    rows_num = len(X)
    oneX = np.zeros([rows_num, 10])
    for i in range(rows_num):
        oneX[i, X[i]] = 1
    return oneX


def main():
    mnist = read_mnist("data")
    nn_model = build_model(0)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ts = time.time()
    for i in range(1000):
        # print("%d round." % i)
        batch_x, batch_y = mnist.train.next_batch(128)
        one_y = trans_to_one_shot(batch_y)

        batch_size = batch_x.shape[0]
        nb_epoch = 100

        early_stopping = EarlyStopping(monitor='loss', patience=2)
        hist = nn_model.fit(batch_x, one_y, batch_size=batch_size, epochs=nb_epoch,
                            verbose=0, callbacks=[early_stopping])
        if i % 50 == 0:
            new_ts = time.time()
            spend_time = new_ts - ts
            ts = new_ts
            print('loss = %f, spend %f.' % (hist.history["loss"][-1], spend_time))
            draw_number(ax, batch_x[0:1, :])
            result = nn_model.predict(batch_x[0:1, :])
            result = np.argmax(nn_model.predict(batch_x[0:1, :]))
            plt.title(result)
            plt.ion()
            plt.show()
            plt.pause(1)


if __name__ == "__main__":
    main()
