import os
import neat
import pickle
import numpy as np
from game import fitness
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                                config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(fitness, 50)
    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
        f.close()

    draw_graphs(stats)
    
    

def replay_genome(config_path, genome_path="winner.pkl"):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                                config_path)

    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    fitness([(1, genome)], config)


def draw_graphs(stats):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle('Fitness Data')

    X = np.arange(1, len(stats.get_fitness_mean()) + 1)
    X = np.reshape(X, (-1, 1))

    y1 = stats.get_fitness_mean()
    y1 = np.reshape(np.array(y1), (-1, 1))
    y2 = stats.get_fitness_median()
    y2 = np.reshape(np.array(y2), (-1, 1))

    ax1.plot(X, y1, label='Fitness Mean', c='b')
    ax2.plot(X, y2, label='Fitness Median', c='b')

    ax1.set_xlabel("Gen")
    ax2.set_xlabel("Gen")
    ax1.set_ylabel("Fitness Mean")
    ax2.set_ylabel("Fitness Median")

    model = LinearRegression()

    model.fit(X, y1)
    X_new = np.array([[1], [len(stats.get_fitness_mean())]])
    y1_pred = model.predict(X_new)
    ax1.plot(X_new, y1_pred, 'g-', label='Linear Regression')

    model.fit(X, y2)
    X_new = np.array([[1], [len(stats.get_fitness_median())]])
    y2_pred = model.predict(X_new)
    ax2.plot(X_new, y2_pred, 'g-', label='Linear Regression')

    plt.show()
    

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
    # replay_genome(config_path)