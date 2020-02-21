import matplotlib.pyplot as plt
import numpy as np
import math
import random
import time


def generate_sample():
    x = np.arange(0, 10, 0.1)
    y = np.zeros([x.shape[0]])
    for i in range(x.shape[0]):
        y[i] = math.cos(x[i]) + random.uniform(-0.2, 0.2)
    return x.reshape((x.shape[0], 1)), y.reshape((y.shape[0], 1))


def generate_test_sample():
    x = np.arange(10, 12, 0.1)
    y = np.zeros([x.shape[0]])
    for i in range(x.shape[0]):
        y[i] = math.cos(x[i]) + random.uniform(-0.2, 0.2)
    return x.reshape((x.shape[0], 1)), y.reshape((y.shape[0], 1))


def build_sample_model():
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation
    from keras.optimizers import Adam

    nn_model = Sequential()
    nn_model.add(Dense(1, input_shape=(1,)))
    nn_model.summary()

    nn_model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=0.01, decay=0.01),
                  metrics=['accuracy'])
    return nn_model


def build_complex_model():
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import Adam

    n_hidden = 10
    nn_model = Sequential()
    nn_model.add(BatchNormalization(input_shape=(1,)))
    nn_model.add(Dense(n_hidden, input_shape=(1,)))
    nn_model.add(Activation('tanh'))
    nn_model.add(BatchNormalization(input_shape=(n_hidden,)))
    nn_model.add(Dense(1, input_shape=(n_hidden,)))
    nn_model.summary()

    nn_model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=0.01, decay=0.01))
    return nn_model


def main():
    from keras.callbacks import EarlyStopping

    train_x, train_y = generate_sample()
    test_x, test_y = generate_test_sample()
    print(train_x.shape)
    print(train_y.shape)
    # nn_model = build_sample_model()
    nn_model = build_complex_model()

    plt.ion()
    plt.show()
    for i in range(1500):
        early_stopping = EarlyStopping(monitor='loss', patience=50)
        hist = nn_model.fit(train_x, train_y, batch_size=train_x.shape[0], epochs=10,
                            verbose=0, callbacks=[early_stopping])
        predict_y = nn_model.predict(train_x)
        verify_y = nn_model.predict(test_x)

        if i % 100 == 0:
            plt.cla()
            plt.scatter(train_x, train_y, color="green")
            plt.scatter(test_x, test_y, color="orange")
            plt.plot(train_x, predict_y, "r-")
            plt.plot(test_x, verify_y, "r-")
            plt.pause(0.1)
            print("i = {}".format(i))
    print("Done")
    time.sleep(15)


if __name__ == "__main__":
    main()
