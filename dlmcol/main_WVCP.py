import logging
import random
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


def test_solution(A, W, S, k):
    nb_conflict = 0
    max_weight = np.zeros((k))
    nb_uncolored = 0
    for v in range(A.shape[0]):
        colorV = S[v]
        if colorV < k:
            if W[v] > max_weight[colorV]:
                max_weight[colorV] = W[v]
            for i in range(A.shape[1]):
                if A[v, i] == 1:
                    colorVprim = S[i]
                    if colorVprim == colorV:
                        nb_conflict += 1
        else:
            nb_uncolored += 1
    score = np.sum(max_weight)
    logging.info(f"nb_conflict : {nb_conflict / 2}")
    logging.info(f"score : {score}")
    logging.info(f"nb_uncolored : {nb_uncolored}")
    return score, nb_conflict / 2


def determine_k(graph):
    size = graph.shape[0]
    vertices = list(range(size))
    max_k = 0
    gamma = np.empty((size, size), dtype=int)
    colors = np.empty(size, dtype=int)
    for _ in range(10):
        random.shuffle(vertices)
        gamma.fill(0)
        colors.fill(-1)
        nb_colors = 0
        for vertex in vertices:
            possible_colors = [
                color for color in range(nb_colors) if gamma[color, vertex] == 0
            ]
            color = random.choice(possible_colors) if possible_colors else nb_colors
            if color == nb_colors:
                nb_colors += 1
            colors[vertex] = color
            for v2 in range(size):
                if graph[vertex, v2] == 1:
                    gamma[color, v2] += 1
        max_k = max(max_k, nb_colors)
    return max_k


def main_WVCP(
    instance, seed, alpha, nb_neighbors, nb_iter_tabu, test, device, name_expe
):

    # Load graph
    filepath = "instances/wvcp_reduced/"
    weights = np.loadtxt(filepath + instance + ".col.w")

    heavyWeights = np.max(weights) >= 128

    size = int(weights.shape[0])

    graph = np.zeros((size, size), dtype=bool)

    with open(filepath + instance + ".col", "r", encoding="utf8") as f:
        for line in f:
            x = line.split(sep=" ")
            if x[0] == "e":
                graph[int(x[1]) - 1, int(x[2]) - 1] = 1
                graph[int(x[2]) - 1, int(x[1]) - 1] = 1

    beginTime = time()

    # Parameters

    max_iter_ITS = 10
    min_dist_insertion = size / 10

    if nb_iter_tabu == -1:
        nb_iter_tabu = int(size * 10)

    if nb_neighbors == -1:
        nb_neighbors = 32

    if alpha == -1:
        alpha = 0.2
        
        
    bigSize = size > 500
    if bigSize:
        size_pop = 4096 * 2
        batch_size = 10
    else:
        size_pop = 4096 * 5
        batch_size = 100

    # TODO : to change
    if instance == "DSJC500.9":
        size_pop = 4096 * 4
        batch_size = 10

    k = determine_k(graph)
    logging.info(f"k : {k}")

    if test:
        logging.info("TEST")
        nb_iter_tabu = 10
        size_pop = 5
        nb_neighbors = 3

    best_score = 99999

    # Init tables

    # new colors generated after offspring
    offsprings_pop = np.zeros((size_pop, size), dtype=np.int32)

    # vector of fitness of the population
    fitness_pop = np.ones((size_pop), dtype=np.int32) * 9999

    # vector of fitness of the offsprings
    fitness_offsprings = np.zeros((size_pop), dtype=np.int32)

    matrice_crossovers_already_tested = np.zeros((size_pop, size_pop), dtype=np.uint8)

    # Big Distance matrix with all individuals in pop and all offsprings at each generation
    matrixDistanceAll = np.zeros((2 * size_pop, 2 * size_pop), dtype=np.int16)
    matrixDistanceAll[:size_pop, :size_pop] = (
        np.ones((size_pop, size_pop), dtype=np.int16) * 9999
    )

    # Consider renaming these matrices
    # Matrix with ditances between individuals in pop and offsprings
    matrixDistance1 = np.zeros((size_pop, size_pop), dtype=np.int16)
    # Matrix with ditances between all offsprings
    matrixDistance2 = np.zeros((size_pop, size_pop), dtype=np.int16)

    offsprings_pop_gpu_memory = cuda.to_device(offsprings_pop)
    fitness_offsprings_gpu_memory = cuda.to_device(fitness_offsprings)

    matrixDistance1_gpu_memory = cuda.to_device(matrixDistance1)
    matrixDistance2_gpu_memory = cuda.to_device(matrixDistance2)

    vect_score = np.zeros((size_pop), dtype=np.float32)
    vect_conflicts = np.zeros((size_pop), dtype=np.float32)

    vect_score_gpu_memory = cuda.to_device(vect_score)
    vect_conflicts_gpu_memory = cuda.to_device(vect_conflicts)

    crossovers_np = np.zeros((size_pop, size), dtype=np.int32)
    crossovers_gpu_memory = cuda.to_device(crossovers_np)

    # Specify size of arrays for cuda kernels
    memetic_algorithm.size = size
    memetic_algorithm.k = k
    memetic_algorithm.size_pop = size_pop
    tabuCol_numba.size = size
    tabuCol_numba.k = k

    if "inithx" in instance or "zeroin" in instance:
        tabuCol_numba.N = int(size // k * 20)
    else:
        tabuCol_numba.N = int(size // k * 5)

    # Configure the different sizes of blocks for numba
    threadsperblock = 64
    blockspergrid1 = (size_pop + (threadsperblock - 1)) // threadsperblock
    blockspergrid2 = (size_pop * size_pop + (threadsperblock - 1)) // threadsperblock

    # Init numba random generator
    rng_states = create_xoroshiro128p_states(
        threadsperblock * blockspergrid1, seed=int(seed)
    )

    A_global_mem = cuda.to_device(graph)  # load adjacency matrix to device
    W_global_mem = cuda.to_device(weights)  # load weights to device

    # Init pop
    if bigSize:
        gamma_tabucol = np.zeros((size_pop, size, k), dtype=np.int8)
        gamma_tabucol_gpu_memory = cuda.to_device(gamma_tabucol)
        memetic_algorithm.initGreedyOrderPopWVCP_bigSize[
            blockspergrid1, threadsperblock
        ](
            rng_states,
            A_global_mem,
            W_global_mem,
            size_pop,
            offsprings_pop_gpu_memory,
            fitness_offsprings_gpu_memory,
            gamma_tabucol_gpu_memory,
        )
    else:
        memetic_algorithm.initGreedyOrderPopWVCP[blockspergrid1, threadsperblock](
            rng_states,
            A_global_mem,
            W_global_mem,
            size_pop,
            offsprings_pop_gpu_memory,
            fitness_offsprings_gpu_memory,
        )

    phi = int(k / size * np.max(weights)) / 2.0
    if phi == 0:
        phi = 0.5

    vect_phi = np.ones((size_pop), dtype=np.float32) * phi

    colors_pop = offsprings_pop_gpu_memory.copy_to_host()
    score, nb_conflict = test_solution(graph, weights, colors_pop[0], k)

    logging.info(f"score test : {score}")
    logging.info(f"nb_conflict test : {nb_conflict}")

    # Build neural network

    nnet = nnetwrapper(
        size,
        k,
        dropout=0.0,
        remix=True,
        verbose=False,
        nbEpochTraining=20,
        layers_size=[
            size,
            size * 5,
            size * 2,
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

        # Collect the starting points and convert it into torch tensor - X's of the training dataset
        train_examples = offsprings_pop_gpu_memory.copy_to_host()
        graph_example = th.FloatTensor(train_examples.astype(np.float32))
        zeros = th.zeros((graph_example.size()[0], size, k))
        ones = th.ones((graph_example.size()[0], size, k))
        graphs = zeros.scatter_(2, graph_example.unsqueeze(2).long(), ones)
        listGraphs = th.split(graphs, int(size_pop), 0)
        offsprings_pop_after_tabu = np.zeros((size_pop, size), dtype=np.int32)
        fitness_offsprings_after_tabu = np.ones((size_pop), dtype=np.int32) * 9999
        for i in range(max_iter_ITS):
            logging.info(f"iter tabu : {i}")
            logging.info("stats phi")
            logging.info(f"min {np.min(vect_phi)}")
            logging.info(f"mean {np.mean(vect_phi)}")
            logging.info(f"max {np.max(vect_phi)}")
            if i == max_iter_ITS - 1:
                vect_phi_global_mem = cuda.to_device(
                    np.ones((size_pop), dtype=np.float32) * np.max(weights) * 2
                )
            else:
                vect_phi_global_mem = cuda.to_device(vect_phi)

            if bigSize:
                tabuCol_numba.tabuWVCP_NoRandom_AFISA_bigSize[
                    blockspergrid1, threadsperblock
                ](
                    rng_states,
                    size_pop,
                    nb_iter_tabu,
                    A_global_mem,
                    W_global_mem,
                    offsprings_pop_gpu_memory,
                    fitness_offsprings_gpu_memory,
                    vect_score_gpu_memory,
                    vect_conflicts_gpu_memory,
                    alpha,
                    vect_phi_global_mem,
                    gamma_tabucol_gpu_memory,
                )
            else:
                if heavyWeights:
                    tabuCol_numba.tabuWVCP_NoRandom_AFISA_heavyWeights[
                        blockspergrid1, threadsperblock
                    ](
                        rng_states,
                        size_pop,
                        nb_iter_tabu,
                        A_global_mem,
                        W_global_mem,
                        offsprings_pop_gpu_memory,
                        fitness_offsprings_gpu_memory,
                        vect_score_gpu_memory,
                        vect_conflicts_gpu_memory,
                        alpha,
                        vect_phi_global_mem,
                    )
                else:
                    tabuCol_numba.tabuWVCP_NoRandom_AFISA[
                        blockspergrid1, threadsperblock
                    ](
                        rng_states,
                        size_pop,
                        nb_iter_tabu,
                        A_global_mem,
                        W_global_mem,
                        offsprings_pop_gpu_memory,
                        fitness_offsprings_gpu_memory,
                        vect_score_gpu_memory,
                        vect_conflicts_gpu_memory,
                        alpha,
                        vect_phi_global_mem,
                    )

            offsprings_pop = offsprings_pop_gpu_memory.copy_to_host()
            fitness_offsprings = fitness_offsprings_gpu_memory.copy_to_host()

            vect_conflicts = vect_conflicts_gpu_memory.copy_to_host()
            vect_score = vect_score_gpu_memory.copy_to_host()

            logging.info(f"vect_score {vect_score}")
            logging.info(f"vect_conflicts {vect_conflicts}")

            for d in range(size_pop):
                if vect_conflicts[d] == 0:
                    vect_phi[d] = vect_phi[d] / 2
                    if fitness_offsprings[d] < fitness_offsprings_after_tabu[d]:
                        fitness_offsprings_after_tabu[d] = fitness_offsprings[d]
                        offsprings_pop_after_tabu[d, :] = offsprings_pop[d, :]
                else:
                    vect_phi[d] = vect_phi[d] * 2

            nb.cuda.synchronize()

        logging.info(f"Tabucol duration : {time() - start}")

        best_score_pop = np.min(fitness_offsprings_after_tabu)
        worst_score_pop = np.max(fitness_offsprings_after_tabu)
        avg_pop = np.mean(fitness_offsprings_after_tabu)

        logging.info(f"Epoch : {epoch}")
        logging.info(
            f"Pop best : {best_score_pop}"
            f"_worst : {worst_score_pop}"
            f"_avg : {avg_pop}"
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
                f"solutions/Solution_WVCP_{instance}_score_{best_current_score}_epoch_{epoch}.csv",
                solution.astype(int),
                fmt="%i",
            )

        with open("evol/" + name_expe, "a", encoding="utf8") as fichier:
            fichier.write(
                f"\n{best_score},{best_current_score},{epoch},{time() - beginTime}"
            )

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

        # Train neural network
        logging.info("Train neural network")

        target_cs = th.FloatTensor(fitness_offsprings_after_tabu.astype(np.float32))
        target_cs = target_cs.view(-1, 1)
        listFs = th.split(target_cs, int(size_pop), 0)
        trainExamples = list(zip(listGraphs[0], listFs[0]))

        logging.info(f"len(trainExamples) {len(trainExamples)}")

        shuffle(trainExamples)
        nnet.train(trainExamples, batch_size)

        start = time()

        logging.info("start crossover")
        bestColor_global_mem = cuda.to_device(colors_pop)
        dist_neighbors = np.where(
            matrice_crossovers_already_tested == 1,
            9999,
            matrixDistanceAll[:size_pop, :size_pop],
        )
        dist_neighbors = np.where(dist_neighbors == 0, 999, dist_neighbors)
        closest_individuals = np.argsort(dist_neighbors, axis=1)[:, :nb_neighbors]
        best_expected_fit = np.ones((size_pop)) * 999
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
        logging.info("end crossover")
        logging.info(f"generation duration : {time() - startEpoch}")
