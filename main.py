import os
import neat
import pickle
from game import fitness

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

def replay_genome(config_path, genome_path="winner.pkl"):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                                config_path)

    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    fitness([(1, genome)], config)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    # run(config_path)
    replay_genome(config_path)