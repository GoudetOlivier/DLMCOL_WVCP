import logging
import math

import numba as nb
import numpy as np
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32

size = -1
k = -1
size_pop = -1
E = -1


@cuda.jit
def initGreedyOrderPopWVCP(rng_states, A, W, D, tColor, fit):
    d = cuda.grid(1)
    if d < D:
        gamma = nb.cuda.local.array((size, k), nb.int8)
        for x in range(size):
            for y in range(k):
                gamma[x, y] = 0
        f = 0
        nb_max_col = 0
        for x in range(size):
            c = 0
            found = False
            startCol = int(nb_max_col * xoroshiro128p_uniform_float32(rng_states, d))
            while not found and c < nb_max_col:
                v = (startCol + c) % nb_max_col
                if gamma[x, v] == 0:
                    found = True
                    tColor[d, x] = v
                    for y in range(size):
                        if A[x, y] == 1:
                            gamma[y, v] += 1
                c = c + 1
            if not found:
                if nb_max_col < k:
                    f += W[x]
                    tColor[d, x] = nb_max_col
                    for y in range(size):
                        if A[x, y] == 1:
                            gamma[y, nb_max_col] += 1
                    nb_max_col += 1
                else:
                    r = int(k * xoroshiro128p_uniform_float32(rng_states, d))
                    tColor[d, x] = r
        fit[d] = f


@cuda.jit
def initGreedyOrderPopWVCP_bigSize(rng_states, A, W, D, tColor, fit, gamma):
    d = cuda.grid(1)
    if d < D:
        for x in range(size):
            for y in range(k):
                gamma[d, x, y] = 0
        f = 0
        nb_max_col = 0
        for x in range(size):
            c = 0
            found = False
            startCol = int(nb_max_col * xoroshiro128p_uniform_float32(rng_states, d))
            while not found and c < nb_max_col:
                v = (startCol + c) % nb_max_col
                if gamma[d, x, v] == 0:
                    found = True
                    tColor[d, x] = v
                    for y in range(size):
                        if A[x, y] == 1:
                            gamma[d, y, v] += 1
                c = c + 1
            if not found:
                if nb_max_col < k:
                    f += W[x]
                    tColor[d, x] = nb_max_col
                    for y in range(size):
                        if A[x, y] == 1:
                            gamma[d, y, nb_max_col] += 1
                    nb_max_col += 1
                else:
                    r = int(k * xoroshiro128p_uniform_float32(rng_states, d))
                    tColor[d, x] = r
        fit[d] = f


@cuda.jit
def initPopGCP(rng_states, D, tColor):
    d = cuda.grid(1)
    if d < D:
        for i in range(size):
            r = int(k * xoroshiro128p_uniform_float32(rng_states, d))
            if r >= k:
                r = k - 1
            tColor[d, i] = r


# Cuda kernel allowing to remove conflicts of D solutions in parallel
@cuda.jit
def removeConflicts(rng_states, D, A, tColor):
    d = cuda.grid(1)
    if d < D:
        startNode = int(size * xoroshiro128p_uniform_float32(rng_states, d))
        for i in range(size):
            x = (startNode + i) % size
            if tColor[d, x] < k:
                for y in range(size):
                    if A[x, y] == 1 and tColor[d, x] == tColor[d, y]:
                        tColor[d, x] = -1


# CUDA kernel : compute symmetric distance matrix between solutions of the same set for each pop - O(S) Daniel Approximation
@cuda.jit
def computeSymmetricMatrixDistance_PorumbelApprox(size_sub_pop, matrixDistance, tColor):
    d = cuda.grid(1)
    if d < (size_sub_pop * (size_sub_pop - 1) / 2):
        # Get upper triangular matrix indices from thread index !
        idx1 = int(
            size_sub_pop
            - 2
            - int(
                math.sqrt(-8.0 * d + 4.0 * size_sub_pop * (size_sub_pop - 1) - 7) / 2.0
                - 0.5
            )
        )
        idx2 = int(
            d
            + idx1
            + 1
            - size_sub_pop * (size_sub_pop - 1) / 2
            + (size_sub_pop - idx1) * ((size_sub_pop - idx1) - 1) / 2
        )
        ttNbSameColor = nb.cuda.local.array((k, k), nb.uint8)
        M = nb.cuda.local.array((k), nb.int16)
        sigma = nb.cuda.local.array((k), nb.int16)
        for j in range(k):
            M[j] = 0
            sigma[j] = 0
        for x in range(size):
            ttNbSameColor[int(tColor[int(idx1), x]), int(tColor[int(idx2), x])] = 0
        for x in range(size):
            i = int(tColor[int(idx1), x])
            j = int(tColor[int(idx2), x])
            ttNbSameColor[i, j] += 1
            if ttNbSameColor[i, j] > M[i]:
                M[i] = ttNbSameColor[i, j]
                sigma[i] = j
        proxi = 0
        for i in range(k):
            proxi += ttNbSameColor[i, sigma[i]]
        matrixDistance[int(idx1), int(idx2)] = size - proxi
        matrixDistance[int(idx2), int(idx1)] = size - proxi


# CUDA kernel : compute distance matrix between two set of solutions for each pop
@cuda.jit
def computeMatrixDistance_PorumbelApprox(
    size_sub_pop, size_sub_pop2, matrixDistance, tColor1, tColor2
):
    d = cuda.grid(1)
    if d < size_sub_pop * size_sub_pop2:
        idx1 = int(d // size_sub_pop2)
        idx2 = int(d % size_sub_pop2)
        ttNbSameColor = nb.cuda.local.array((k, k), nb.uint8)
        M = nb.cuda.local.array((k), nb.int16)
        sigma = nb.cuda.local.array((k), nb.int16)
        for i in range(k):
            M[i] = 0
            sigma[i] = 0
        for x in range(size):
            ttNbSameColor[int(tColor1[int(idx1), x]), int(tColor2[int(idx2), x])] = 0
        for x in range(size):
            i = int(tColor1[int(idx1), x])
            j = int(tColor2[int(idx2), x])
            ttNbSameColor[i, j] += 1
            if ttNbSameColor[i, j] > M[i]:
                M[i] = ttNbSameColor[i, j]
                sigma[i] = j
        proxi = 0
        for i in range(k):
            proxi += ttNbSameColor[i, sigma[i]]
        matrixDistance[int(idx1), int(idx2)] = size - proxi


@cuda.jit
def computeClosestCrossover(rng_states, size_pop, tColor, allCrossovers, indices):
    d = cuda.grid(1)
    nbParent = 2
    if d < size_pop:
        idx1 = int(d)
        idx2 = int(indices[idx1])
        parents = nb.cuda.local.array((nbParent, size), nb.int16)
        current_child = nb.cuda.local.array((size), nb.int16)
        for j in range(size):
            parents[0, j] = tColor[idx1, j]
            parents[1, j] = tColor[idx2, j]
        for j in range(size):
            current_child[j] = -1
        tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)
        for i in range(nbParent):
            for j in range(k):
                tSizeOfColors[i, j] = 0
            for j in range(size):
                if parents[i, j] > -1:
                    tSizeOfColors[i, parents[i, j]] += 1
        for i in range(k):
            indiceParent = i % 2
            valMax = -1
            colorMax = -1
            startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d))
            for j in range(k):
                color = (startColor + j) % k
                currentVal = tSizeOfColors[indiceParent, color]
                if currentVal > valMax:
                    valMax = currentVal
                    colorMax = color
            for j in range(size):
                if parents[int(indiceParent), j] == colorMax and current_child[j] < 0:
                    current_child[j] = i
                    for l in range(nbParent):
                        if parents[l, j] > -1:
                            tSizeOfColors[l, parents[l, j]] -= 1
        for j in range(size):
            if current_child[j] < 0:
                r = int(k * xoroshiro128p_uniform_float32(rng_states, d))
                if r >= k:
                    r = k - 1
                current_child[j] = r
        for j in range(size):
            allCrossovers[d, j] = current_child[j]


def insertion_pop(
    size_pop,
    matrixDistanceAll,
    colors_pop,
    offsprings_pop_after_tabu,
    fitness_pop,
    fitness_offsprings_after_tabu,
    matrice_crossovers_already_tested,
    min_dist,
):
    all_scores = np.hstack((fitness_pop, fitness_offsprings_after_tabu))
    matrice_crossovers_already_tested_new = np.zeros(
        (size_pop * 2, size_pop * 2), dtype=np.uint8
    )
    matrice_crossovers_already_tested_new[
        :size_pop, :size_pop
    ] = matrice_crossovers_already_tested
    idx_best = np.argsort(all_scores)
    idx_selected = []
    cpt = 0
    for i in range(0, size_pop * 2):
        idx = idx_best[i]
        if len(idx_selected) > 0:
            dist = np.min(matrixDistanceAll[idx, idx_selected])
        else:
            dist = 9999
        if dist >= min_dist:
            idx_selected.append(idx)
            if idx >= size_pop:
                cpt += 1
        if len(idx_selected) == size_pop:
            break
    logging.info(f"len(idx_selected) {len(idx_selected)}")
    if len(idx_selected) != size_pop:
        for i in range(0, size_pop * 2):
            idx = idx_best[i]
            if idx not in idx_selected:
                dist = np.min(matrixDistanceAll[idx, idx_selected])
                if dist >= 0:
                    idx_selected.append(idx)
            if len(idx_selected) == size_pop:
                break
    logging.info(f"Nb insertion {cpt}")
    new_matrix = matrixDistanceAll[idx_selected, :][:, idx_selected]
    stack_all = np.vstack((colors_pop, offsprings_pop_after_tabu))
    colors_pop_v2 = stack_all[idx_selected]
    fitness_pop_v2 = all_scores[idx_selected]
    matrice_crossovers_already_tested_v2 = matrice_crossovers_already_tested_new[
        idx_selected, :
    ][:, idx_selected]
    return (
        new_matrix,
        fitness_pop_v2,
        colors_pop_v2,
        matrice_crossovers_already_tested_v2,
    )
