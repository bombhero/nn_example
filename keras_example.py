import numpy as np
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import matplotlib.pyplot as plt


def build_model(verbose=0):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import Adam

    model = Sequential()
    model.add(BatchNormalization(input_shape=(784,)))
    model.add(Dense(10, input_shape=(784,)))
    model.add(Activation('sigmoid'))

    if verbose == 1:
        model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.01, decay=0.01),
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
    from keras.callbacks import EarlyStopping

    mnist = read_data_sets("./data")
    nn_model = build_model(0)

    return
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
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
            print('loss = %f' % hist.history["loss"][-1])
            draw_number(ax, batch_x[0:1, :])
            result = nn_model.predict(batch_x[0:1, :])
            result = np.argmax(nn_model.predict(batch_x[0:1, :]))
            plt.title(result)
            plt.ion()
            plt.show()
            plt.pause(1)


if __name__ == "__main__":
    main()
