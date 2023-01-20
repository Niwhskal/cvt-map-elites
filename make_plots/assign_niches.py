import os
import neat
import visualize
import numpy as np
import multiprocessing
import pickle
from pathlib import Path
from sim_env import Environment
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree


def __save_archive(archive, gen):
    filename = './archives/archive_' + str(gen)+ str(dims) + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(archive, f, pickle.HIGHEST_PROTOCOL)


def make_hashable(array):
    return tuple(map(float, array))

def __add_to_archive(s, centroid, archive, kdt):
    niche_index = kdt.query(np.array([np.array(centroid)]), k=1)[1][0][0]
    niche = kdt.data[niche_index]
    n = make_hashable(niche)
    s.centroid = n
    if n in archive:
        if s.fitness > archive[n].fitness:
            archive[n] = s
            return 1
        return 0
    else:
        archive[n] = s
        return 1

def __centroids_filename(k, dim):
    return 'centroids_' + str(k) + '_' + str(dim) + '.dat'

def __write_centroids(centroids):
    k = centroids.shape[0]
    dim = centroids.shape[1]
    filename = __centroids_filename(k, dim)
    with open(filename, 'w') as f:
        for p in centroids:
            for item in p:
                f.write(str(item) + ' ')
            f.write('\n')

def cvt(k, dim, samples, cvt_use_cache=True):
# check if we have cached values
    fname = __centroids_filename(k, dim)
    if cvt_use_cache:
        if Path(fname).is_file():
            print("WARNING: using cached CVT:", fname)
            return np.loadtxt(fname)
# otherwise, compute cvt
    print("Computing CVT (this can take a while...):", fname)

    if dim==2:
        x = np.random.rand(samples, dim)
    else:
        raise NotImplementedError

    k_means = KMeans(init='k-means++', n_clusters=k,
                     n_init=1, verbose=1)#,algorithm="full")
    k_means.fit(x)
    __write_centroids(k_means.cluster_centers_)

    return k_means.cluster_centers_

class Species:
    def __init__(self, x, desc, fitness, centroid=None):
        self.x = x
        self.desc = desc
        self.fitness = fitness
        self.centroid = centroid


def eval_genome(genomes, config):
    print('Simulating Khera...')
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        env = Environment(dims)
        perf, desc = env.simulate(net) # run robot for 3k timesteps
        print(f'Performance of Genome : {-perf}, terminated at {desc}')
        s = Species(genome, desc, -perf)
        __add_to_archive(s, desc, archive, kdt)
        # genome.fitness = -perf
        global genCnt
        genCnt += 1
        print(f'Saving Archive at gen: {genCnt%2000}, {dims}')
        __save_archive(archive, genCnt)
        del env
    return 0
    # return performances


def run(config_file):
    # Load configuration.
    print('Starting NEAT ...')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)

    # Run for up to 990 generations.
    # winner = p.run(pe.evaluate, 1)
    p = neat.Checkpointer.restore_checkpoint(os.path.join('../','neat-checkpoint-30'))
    genomes = p.population.items()
    _ = eval_genome(genomes, config)

        # np.save(f'./perf_{ckpt}', np.array(performances))

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner))

    # # Show output of the most fit genome against training data.
    # print('\nOutput:')
    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # # perf = env.simulate(winner_net)
    # print(f"Distance to goal : {perf}")

    # # node_names = {-1: 'LeftRadar', -2: 'CenterRadar', -3: 'RightRadar', -4: "Slice1", -5: "Slice2", -6: "Slice3", -7: "Slice4", 0: 'Lvel', 1: "Rvel" }
    # visualize.draw_net(config, winner, True)#, node_names=node_names)
    # visualize.draw_net(config, winner, True, prune_unused=True)#node_names=node_names,
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.

    N_niches = 5000
    dim_map = 2
    samples = int(1000e03)
    global genCnt
    genCnt = 0

    local_dir = '../'#os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')

    for ctf in centroid_files:
        # creating centroids
        # c = cvt(N_niches, dim_map, samples)

        c = np.load(ctf)
        print('Creating KDTree...')
        global dims
        dims =
        kdt = KDTree(c, leaf_size = 30, metric='euclidean')
        run(config_path)

    # # create an empty archive
    # print('Creating empty archive')
    # archive = {}


