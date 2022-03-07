import logging
from random import shuffle
from time import time

import numba as nb
import numpy as np
import torch as th
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

from dlmcol import memetic_algorithm, tabuCol_numba
from dlmcol.memetic_algorithm import insertion_pop
from dlmcol.NNetWrapper import NNetWrapper as nnetwrapper


def main_GCP(
    instance, k, seed, alpha, nb_neighbors, nb_iter_tabu, test, device, name_expe
):
    # Load graph
    filepath = "instances/gcp/"

    with open(filepath + instance + ".col", "r", encoding="utf8") as f:
        for line in f:
            x = line.split(sep=" ")
            if x[0] == "p":
                size = int(x[2])
                break

        logging.info(f"size {size}")

        graph = np.zeros((size, size), dtype=np.int16)

        for line in f:
            x = line.split(sep=" ")
            if x[0] == "e":
                graph[int(x[1]) - 1, int(x[2]) - 1] = 1
                graph[int(x[2]) - 1, int(x[1]) - 1] = 1

    beginTime = time()

    # load adjacency matrix to device
    A_global_mem = cuda.to_device(graph)

    # Parameters

    min_dist_insertion = size / 10

    if nb_iter_tabu == -1:
        nb_iter_tabu = int(size * 128)

    if alpha == -1:
        alpha = 0.6
        
    batch_size = 100
    size_pop = 4096 * 5

    if nb_neighbors == -1:
        nb_neighbors = 16
        
        
    if test:
        logging.info("TEST")

        nb_iter_tabu = 10
        size_pop = 5
        nb_neighbors = 3

    best_score = 99999

    logging.info(f"k : {k}")

    # Init tables

    offsprings_pop = np.zeros(
        (size_pop, size), dtype=np.int32
    )  # new colors generated after offspring

    fitness_pop = (
        np.ones((size_pop), dtype=np.int32) * 9999
    )  # vector of fitness of the population
    fitness_offsprings = np.zeros(
        (size_pop), dtype=np.int32
    )  # vector of fitness of the offsprings

    matrice_crossovers_already_tested = np.zeros((size_pop, size_pop), dtype=np.uint8)

    # Big Distance matrix with all individuals in pop and all offsprings at each generation
    matrixDistanceAll = np.zeros((2 * size_pop, 2 * size_pop), dtype=np.int16)

    matrixDistanceAll[:size_pop, :size_pop] = (
        np.ones((size_pop, size_pop), dtype=np.int16) * 9999
    )

    matrixDistance1 = np.zeros(
        (size_pop, size_pop), dtype=np.int16
    )  # Matrix with distances between individuals in pop and offsprings
    matrixDistance2 = np.zeros(
        (size_pop, size_pop), dtype=np.int16
    )  # Matrix with distances between all offsprings

    tabuTenure = np.zeros((size_pop, size, k), dtype=np.int32)
    tabuTenure_gpu_memory = cuda.to_device(tabuTenure)

    offsprings_pop_gpu_memory = cuda.to_device(offsprings_pop)
    fitness_offsprings_gpu_memory = cuda.to_device(fitness_offsprings)

    matrixDistance1_gpu_memory = cuda.to_device(matrixDistance1)
    matrixDistance2_gpu_memory = cuda.to_device(matrixDistance2)

    crossovers_np = np.zeros((size_pop, size), dtype=np.int32)
    crossovers_gpu_memory = cuda.to_device(crossovers_np)

    memetic_algorithm.size = size
    memetic_algorithm.k = k
    tabuCol_numba.size = size
    tabuCol_numba.k = k
    memetic_algorithm.size_pop = size_pop

    # Configure the different sizes of blocks for numba
    threadsperblock = 64
    blockspergrid1 = (size_pop + (threadsperblock - 1)) // threadsperblock
    blockspergrid2 = (size_pop * size_pop + (threadsperblock - 1)) // threadsperblock

    # Init numba random generator
    rng_states = create_xoroshiro128p_states(
        threadsperblock * blockspergrid1, seed=int(seed)
    )

    # Init pop

    logging.info("initPopGCP")
    memetic_algorithm.initPopGCP[blockspergrid1, threadsperblock](
        rng_states, size_pop, offsprings_pop_gpu_memory
    )

    colors_pop = offsprings_pop_gpu_memory.copy_to_host()

    # Build neural network
    nnet = nnetwrapper(
        size,
        k,
        dropout=0.0,
        remix=True,
        verbose=False,
        nbEpochTraining=5,
        layers_size=[
            size,
            size * 10,
            size * 5,
            size * 2,
            size * 2,
            size * 2,
            size * 2,
            size,
            size,
            size // 2,
            1,
        ],
    )
    nnet.set_to_device(device)

    for epoch in range(100000):
        # First step : local search
        # Start tabu

        logging.info("############################")
        logging.info("Start TABU")
        logging.info("############################")

        startEpoch = time()
        start = time()

        offsprings_pop_after_tabu = np.zeros((size_pop, size), dtype=np.int32)
        fitness_offsprings_after_tabu = np.ones((size_pop), dtype=np.int32) * 99999

        # Collect the starting points of the local search and convert it into torch tensor - X's of the training dataset
        train_examples = offsprings_pop_gpu_memory.copy_to_host()
        graph_example = th.FloatTensor(train_examples.astype(np.float32))
        zeros = th.zeros((graph_example.size()[0], size, k))
        ones = th.ones((graph_example.size()[0], size, k))
        graphs = zeros.scatter_(2, graph_example.unsqueeze(2).long(), ones)
        listGraphs = th.split(graphs, int(size_pop), 0)

        tabuCol_numba.tabuGCP[blockspergrid1, threadsperblock](
            rng_states,
            size_pop,
            nb_iter_tabu,
            A_global_mem,
            offsprings_pop_gpu_memory,
            fitness_offsprings_gpu_memory,
            alpha,
            tabuTenure_gpu_memory,
        )

        nb.cuda.synchronize()

        offsprings_pop = offsprings_pop_gpu_memory.copy_to_host()
        fitness_offsprings_after_tabu = fitness_offsprings_gpu_memory.copy_to_host()
        offsprings_pop_after_tabu = offsprings_pop

        logging.info(f"Tabucol duration : {time() - start}")

        best_score_pop = np.min(fitness_offsprings_after_tabu)
        worst_score_pop = np.max(fitness_offsprings_after_tabu)
        avg_pop = np.mean(fitness_offsprings_after_tabu)

        logging.info(f"Epoch : {epoch}")
        logging.info(
            f"Pop best : {best_score_pop}_worst : {worst_score_pop}_avg : {avg_pop}"
        )

        logging.info("end tabu")

        # Get and log results

        logging.info("############################")
        logging.info("Results TabuCol")
        logging.info("############################")

        best_current_score = min(fitness_offsprings_after_tabu)

        if best_current_score < best_score:

            best_score = best_current_score

            logging.info("Save best solution")

            solution = offsprings_pop_after_tabu[
                np.argmin(fitness_offsprings_after_tabu)
            ]

            np.savetxt(
                f"solutions/Solutions_GCP_{instance}_score_{best_current_score}_epoch_{epoch}.csv",
                solution.astype(int),
                fmt="%i",
            )

        with open("evol/" + name_expe, "a", encoding="utf8") as f:
            f.write(f"{best_score},{best_current_score},{epoch},{time() - beginTime}\n")

        if best_score == 0:
            break

        # Second step : insertion of offsprings in pop according to diversity/fit criterion

        logging.info("Keep best with diversity/fit tradeoff")

        logging.info("start matrix distance")

        start = time()

        offsprings_pop_gpu_memory = cuda.to_device(offsprings_pop_after_tabu)

        colors_pop_gpu_memory = cuda.to_device(colors_pop)

        memetic_algorithm.computeMatrixDistance_PorumbelApprox[
            blockspergrid2, threadsperblock
        ](
            size_pop,
            size_pop,
            matrixDistance1_gpu_memory,
            colors_pop_gpu_memory,
            offsprings_pop_gpu_memory,
        )

        matrixDistance1 = matrixDistance1_gpu_memory.copy_to_host()

        memetic_algorithm.computeSymmetricMatrixDistance_PorumbelApprox[
            blockspergrid2, threadsperblock
        ](size_pop, matrixDistance2_gpu_memory, offsprings_pop_gpu_memory)

        matrixDistance2 = matrixDistance2_gpu_memory.copy_to_host()

        # Aggregate all the matrix in order to obtain a full 2*size_pop matrix with all the distances between individuals in pop and in offspring
        matrixDistanceAll[:size_pop, size_pop:] = matrixDistance1
        matrixDistanceAll[size_pop:, :size_pop] = matrixDistance1.transpose(1, 0)
        matrixDistanceAll[size_pop:, size_pop:] = matrixDistance2

        logging.info(f"Matrix distance duration : {time() - start}")

        offsprings_pop_gpu_memory = None
        nb.cuda.synchronize()

        logging.info("end  matrix distance")
        #####################################

        logging.info("start insertion in pop")
        start = time()

        results = insertion_pop(
            size_pop,
            matrixDistanceAll,
            colors_pop,
            offsprings_pop_after_tabu,
            fitness_pop,
            fitness_offsprings_after_tabu,
            matrice_crossovers_already_tested,
            min_dist_insertion,
        )

        matrixDistanceAll[:size_pop, :size_pop] = results[0]
        fitness_pop = results[1]
        colors_pop = results[2]
        matrice_crossovers_already_tested = results[3]

        logging.info(f"Insertion in pop : {time() - start}")

        logging.info("end insertion in pop")

        logging.info("After keep best info")

        best_score_pop = np.min(fitness_pop)
        worst_score_pop = np.max(fitness_pop)
        avg_score_pop = np.mean(fitness_pop)

        logging.info(
            f"Pop _best : {best_score_pop}_worst : {worst_score_pop}_avg : {avg_score_pop}"
        )
        logging.info(fitness_pop)
        matrix_distance_pop = matrixDistanceAll[:size_pop, :size_pop]
        max_dist = np.max(matrix_distance_pop)
        min_dist = np.min(matrix_distance_pop + np.eye(size_pop) * 9999)
        avg_dist = np.sum(matrix_distance_pop) / (size_pop * (size_pop - 1))
        logging.info(
            f"Avg dist : {avg_dist} min dist : {min_dist} max dist : {max_dist}"
        )
        # Third step : selection of best crossovers to generate new offsprings
        logging.info("############################")
        logging.info("start crossover")
        logging.info("############################")
        # Collect the fitness after the local search and convert it into torch tensor - Y's of the training dataset
        target_cs = th.FloatTensor(fitness_offsprings_after_tabu.astype(np.float32))
        target_cs = target_cs.view(-1, 1)
        listFs = th.split(target_cs, int(size_pop), 0)
        trainExamples = list(zip(listGraphs[0], listFs[0]))
        # Train neural network
        logging.info(f"len(trainExamples) = {len(trainExamples)}")
        shuffle(trainExamples)
        nnet.train(trainExamples, batch_size)
        start = time()
        logging.info("Start crossovers")
        bestColor_global_mem = cuda.to_device(colors_pop)
        # Remove conflicts before crossover
        memetic_algorithm.removeConflicts[blockspergrid1, threadsperblock](
            rng_states, size_pop, A_global_mem, bestColor_global_mem
        )
        dist_neighbors = np.where(
            matrice_crossovers_already_tested == 1,
            9999,
            matrixDistanceAll[:size_pop, :size_pop],
        )
        dist_neighbors = np.where(dist_neighbors == 0, 999, dist_neighbors)
        closest_individuals = np.argsort(dist_neighbors, axis=1)[:, :nb_neighbors]
        best_expected_fit = np.ones((size_pop)) * 9999
        best_solutions = np.zeros((size_pop, size))
        best_idx = np.zeros((size_pop))
        for n in range(nb_neighbors):
            closest_individuals_gpu_memory = cuda.to_device(
                np.ascontiguousarray(closest_individuals[:, n])
            )
            memetic_algorithm.computeClosestCrossover[blockspergrid1, threadsperblock](
                rng_states,
                size_pop,
                bestColor_global_mem,
                crossovers_gpu_memory,
                closest_individuals_gpu_memory,
            )
            crossovers_np = crossovers_gpu_memory.copy_to_host()
            graph_example = th.FloatTensor(crossovers_np.astype(np.float32))
            zeros = th.zeros((graph_example.size()[0], size, k))
            ones = th.ones((graph_example.size()[0], size, k))
            graphs = zeros.scatter_(2, graph_example.unsqueeze(2).long(), ones)
            listGraphs = th.split(graphs, int(size_pop), 0)
            expected_fit = nnet.predict_batch(graphs, batch_size)
            expected_fit = np.asarray(expected_fit)
            if n == 0:
                best_expected_fit = expected_fit
                best_solutions = crossovers_np
                best_idx = closest_individuals[:, n]
            else:
                for i in range(size_pop):
                    if expected_fit[i] <= best_expected_fit[i]:
                        best_solutions[i, :] = crossovers_np[i, :]
                        best_expected_fit[i] = expected_fit[i]
                        best_idx[i] = closest_individuals[i, n]
        logging.info("best_expected_fit")
        logging.info(np.min(best_expected_fit))
        logging.info(np.max(best_expected_fit))
        logging.info(np.mean(best_expected_fit))
        for i in range(size_pop):
            matrice_crossovers_already_tested[i, best_idx[i]] = 1
        offsprings_pop_gpu_memory = cuda.to_device(best_solutions.astype(np.int32))
        logging.info(
            f"nb cross already tested in pop : {np.sum(matrice_crossovers_already_tested)}"
        )
        logging.info(f"Crossover duration : {time() - start}")
        logging.info(f"generation duration : {time() - startEpoch}")
