import numpy as np
import time
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import matplotlib.pyplot as plt


def build_resnet_core(input_node):
    from keras.models import Model
    from keras.layers import Dense, Activation, Add
    from keras.optimizers import Adam
    from keras.utils import plot_model

    step_1 = Dense(784, activation='relu')(input_node)
    step_2 = Dense(784, activation='linear')(step_1)
    step_3 = Add()([input_node, step_2])
    resnet_out = Activation('relu')(step_3)
    resnet = Model(inputs=input_node, outputs=resnet_out)
    resnet.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01, decay=0.01), metrics=['accuracy'])

    plot_model(resnet, to_file='core.png')

    return resnet, resnet_out


def build_model(verbose=0):
    from keras.models import Model
    from keras.layers import Dense, Activation, Input
    from keras.optimizers import Adam
    from keras.utils import plot_model

    net_in = Input(shape=(784, ))
    data_in = net_in
    for _ in range(50):
        layer_in = Input(shape=(784,))
        resnet_core, data_out = build_resnet_core(layer_in)
        data_out = resnet_core(data_in)
        data_in = data_out

    net_out = Dense(10, activation='softmax')(data_out)

    model = Model(inputs=net_in, outputs=net_out)

    if verbose == 1:
        model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.01, decay=0.01),
                  metrics=['accuracy'])
    plot_model(model, to_file='model.png')
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
