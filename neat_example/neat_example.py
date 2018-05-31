import os
import neat
import numpy as np
import multiprocessing
from visualize import draw_net
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

mnist = read_data_sets("../data")


def trans_to_one_shot(x):
    rows_num = len(x)
    one_hot = np.zeros([rows_num, 10])
    for i in range(rows_num):
        one_hot[i, x[i]] = 1
    return one_hot


def calc_fitness(genome, config):
    global mnist
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = 128
    batch_x, batch_y = mnist.train.next_batch(128)
    for in_x, out_y in zip(batch_x, batch_y):
        result_y = np.argmax(net.activate(in_x))
        if result_y != out_y:
            fitness -= 1
    return fitness / 128


def parallel_fitness(genomes, config):
    parallel_calc = neat.ParallelEvaluator(multiprocessing.cpu_count(), calc_fitness)
    parallel_calc.evaluate(genomes, config)


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
    winner = p.run(parallel_fitness, 100)
    draw_net(neat_config, winner, True)


if __name__ == '__main__':
    main()


