import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Activation, Input, Reshape
from keras.layers import Conv1D, Flatten, Dropout
from keras.optimizers import SGD, Adam


def sample_data(n_samples=10000, x_vals=np.arange(0, 5, .1), max_offset=1000, mul_range=[1, 2]):
    vectors = []
    for i in range(n_samples):
        offset = np.random.random() * max_offset
        mul = mul_range[0] + np.random.random() * (mul_range[1] - mul_range[0])
        vectors.append(np.sin(offset + x_vals * mul) / 2 + .5)

    return np.array(vectors)


if __name__ == "__main__":
    XT = sample_data()
    pass
