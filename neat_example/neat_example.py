import os
import neat
import numpy as np
from visualize import draw_net
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

mnist = read_data_sets("../data")


def trans_to_one_shot(x):
    rows_num = len(x)
    one_hot = np.zeros([rows_num, 10])
    for i in range(rows_num):
        one_hot[i, x[i]] = 1
    return one_hot


def calc_fitness(genomes, config):
    global mnist
    for g_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 1280
        batch_x, batch_y = mnist.train.next_batch(128)
        one_y = trans_to_one_shot(batch_y)
        for in_x, out_y in zip(batch_x, one_y):
            result_y = np.array(net.activate(in_x))
            fitness -= np.sum(np.abs(result_y - out_y))
        genome.fitness = fitness / 1280


def main():
    config_path = os.path.dirname(__file__)
    config_file = os.path.join(config_path, "neat.conf")
    neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                              neat.DefaultSpeciesSet, neat.DefaultStagnation,
                              config_file)
    p = neat.Population(neat_config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    winner = p.run(calc_fitness, 10)
    draw_net(neat_config, winner, True)


if __name__ == '__main__':
    main()


