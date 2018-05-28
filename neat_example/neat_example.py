import os
import neat
from visualize import draw_net
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

def main():
    config_path = os.path.dirname(__file__)
    config_file = os.path.join(config_path, "neat.conf")
    mnist = read_data_sets("../data")
    pass


if if __name__ == '__main__':
    main()






