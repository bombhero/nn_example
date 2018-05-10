import numpy as np
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import matplotlib.pyplot as plt
import time


def trans_to_one_shot(X):
    rows_num = len(X)
    oneX = np.zeros([rows_num, 10])
    for i in range(rows_num):
        oneX[i, X[i]] = 1
    return oneX


def set_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def set_bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def draw_number(ax, X):
    Matrix = np.reshape(X, [28, 28])
    ax.imshow(Matrix)


def main():
    mnist = read_data_sets("./data")
    trainX = tf.placeholder("float", [None, mnist.train.images.shape[1]])
    W = tf.Variable(set_weight_variable([mnist.train.images.shape[1], 10]))
    b = tf.Variable(set_bias_variable([10]))
    Y = tf.nn.softmax(tf.matmul(trainX, W) + b)
    trainY = tf.placeholder("float", [None, 10])
    cross_entropy = -tf.reduce_sum(trainY * tf.log(Y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    saver = tf.train.Saver()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ts = time.time()
    for i in range(1000):
        # print("%d round." % i)
        batchX, batchY = mnist.train.next_batch(128)
        oneY = trans_to_one_shot(batchY)
        sess.run(train_step, feed_dict={trainX: batchX, trainY: oneY})
        if i % 50 == 0:
            new_ts = time.time()
            spend_time = new_ts - ts
            ts = new_ts
            print("%f, spend %f" % (sess.run(cross_entropy, feed_dict={trainX: batchX, trainY: oneY}), spend_time))
            result = np.argmax(sess.run(Y, feed_dict={trainX: batchX[0:1, :], trainY: oneY[0:1, :]}))
            draw_number(ax, batchX[0:1, :])
            plt.title(result)
            plt.ion()
            plt.show()
            plt.pause(10)

    model_dir = "./model"
    model_file = "%s/mnist_bomb" % model_dir
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    saver.save(sess, model_file)


if __name__ == "__main__":
    main()
