import numba as nb
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32

size = -1
k = -1
E = -1
N = -1


@cuda.jit
def tabuWVCP_NoRandom_AFISA(
    rng_states,
    D,
    max_iter,
    A,
    W,
    tColor,
    vect_fit,
    vect_score,
    vect_conflicts,
    alpha,
    phi,
):
    # vect_nb_vois, voisin
    d = cuda.grid(1)
    if d < D:
        f = 0
        tColor_local = nb.cuda.local.array((size), nb.int16)
        gamma = nb.cuda.local.array((size, k), nb.int8)
        gammaDepart = nb.cuda.local.array((size), nb.int8)
        gammaArrive = nb.cuda.local.array((size, k), nb.int8)
        max_weight = nb.cuda.local.array((k), nb.int8)
        secondmax_weight = nb.cuda.local.array((k), nb.int8)
        tabuTenure = nb.cuda.local.array((size), nb.int32)
        tGroup_color = nb.cuda.local.array((k, N), nb.int16)
        for c in range(k):
            max_weight[c] = 0
            secondmax_weight[c] = 0
            for i in range(N):
                tGroup_color[c, i] = -1
        for x in range(size):
            gammaDepart[x] = 0
            for y in range(k):
                gamma[x, y] = 0
                gammaArrive[x, y] = 0
            tColor_local[x] = int(tColor[d, x])
            idx = 0
            while tGroup_color[tColor_local[x], idx] != -1:
                idx += 1
            tGroup_color[tColor_local[x], idx] = x
            tabuTenure[x] = -1
        nb_conflicts = 0
        score_wvcp = 0
        for x in range(size):
            for y in range(x):
                if A[x, y] == 1:
                    gamma[x, tColor_local[y]] += 1
                    gamma[y, tColor_local[x]] += 1
                    if tColor_local[y] == tColor_local[x]:
                        nb_conflicts += 1
            if W[x] >= max_weight[tColor_local[x]]:
                score_wvcp += W[x] - max_weight[tColor_local[x]]
                secondmax_weight[tColor_local[x]] = max_weight[tColor_local[x]]
                max_weight[tColor_local[x]] = int(W[x])
            elif W[x] > secondmax_weight[tColor_local[x]]:
                secondmax_weight[tColor_local[x]] = int(W[x])
        for x in range(size):
            for c in range(k):
                if W[x] > max_weight[c]:
                    gammaArrive[x, c] = W[x] - max_weight[c]
                else:
                    gammaArrive[x, c] = 0
        for x in range(size):
            if W[x] == max_weight[tColor_local[x]]:
                gammaDepart[x] = secondmax_weight[tColor_local[x]] - W[x]
            else:
                gammaDepart[x] = 0
        f = score_wvcp + phi[d] * nb_conflicts
        f_best = f
        score_best = score_wvcp
        nb_conflicts_best = nb_conflicts
        for iter_ in range(max_iter):
            best_delta = 9999
            best_delta_conflicts = -1
            best_delta_score = -1
            best_x = -1
            best_v = -1
            for x in range(size):
                v_x = tColor_local[x]
                for v in range(k):
                    if v != v_x:
                        delta_score = gammaArrive[x, v] + gammaDepart[x]
                        delta_conflicts = gamma[x, v] - gamma[x, v_x]
                        delta = delta_score + phi[d] * delta_conflicts
                        if tabuTenure[x] <= iter_ or delta + f < f_best:
                            if delta < best_delta:
                                best_x = x
                                best_v = v
                                best_delta = delta
                                best_delta_conflicts = delta_conflicts
                                best_delta_score = delta_score
            f += best_delta
            score_wvcp += best_delta_score
            nb_conflicts += best_delta_conflicts
            old_color = tColor_local[best_x]
            for y in range(size):
                if A[best_x, y] == 1:
                    gamma[y, old_color] -= 1
                    gamma[y, best_v] += 1
            tColor_local[best_x] = best_v
            old_max_old_color = max_weight[old_color]
            old_second_max_old_color = secondmax_weight[old_color]
            max_weight[old_color] = 0
            secondmax_weight[old_color] = 0
            for idx in range(N):
                x = tGroup_color[old_color, idx]
                if x == best_x:
                    tGroup_color[old_color, idx] = -1
                elif x != -1:
                    if W[x] >= max_weight[old_color]:
                        secondmax_weight[old_color] = max_weight[old_color]
                        max_weight[old_color] = int(W[x])
                    elif W[x] > secondmax_weight[old_color]:
                        secondmax_weight[old_color] = int(W[x])
            idx = 0
            while tGroup_color[best_v, idx] != -1:
                idx += 1
            tGroup_color[best_v, idx] = best_x
            old_max_best_v = max_weight[best_v]
            if W[best_x] >= max_weight[best_v]:
                secondmax_weight[best_v] = max_weight[best_v]
                max_weight[best_v] = int(W[best_x])
            elif W[best_x] > secondmax_weight[best_v]:
                secondmax_weight[best_v] = int(W[best_x])
            if max_weight[old_color] != old_max_old_color:
                for x in range(size):
                    if W[x] >= max_weight[old_color]:
                        gammaArrive[x, old_color] = W[x] - max_weight[old_color]
                    else:
                        gammaArrive[x, old_color] = 0
            if max_weight[best_v] != old_max_best_v:
                for x in range(size):
                    if W[x] >= max_weight[best_v]:
                        gammaArrive[x, best_v] = W[x] - max_weight[best_v]
                    else:
                        gammaArrive[x, best_v] = 0
            if (
                old_second_max_old_color != secondmax_weight[old_color]
                or max_weight[old_color] != old_max_old_color
            ):
                for idx in range(N):
                    x = tGroup_color[old_color, idx]
                    if x != -1:
                        if W[x] == max_weight[old_color]:
                            gammaDepart[x] = secondmax_weight[old_color] - W[x]
                        else:
                            gammaDepart[x] = 0
            for idx in range(N):
                x = tGroup_color[best_v, idx]
                if x != -1:
                    if W[x] == max_weight[best_v]:
                        gammaDepart[x] = secondmax_weight[best_v] - W[x]
                    else:
                        gammaDepart[x] = 0
            tabuTenure[best_x] = (
                int(alpha * size)
                + int(10 * xoroshiro128p_uniform_float32(rng_states, d))
                + iter_
            )
            if f < f_best:
                f_best = f
                score_best = score_wvcp
                nb_conflicts_best = nb_conflicts
                for a in range(size):
                    tColor[d, a] = tColor_local[a]
        vect_fit[d] = f_best
        vect_score[d] = score_best
        vect_conflicts[d] = nb_conflicts_best


@cuda.jit
def tabuGCP(rng_states, D, max_iter, A, tColor, vect_fit, alpha, tabuTenure):
    # vect_nb_vois, voisin
    d = cuda.grid(1)
    if d < D:
        f = 0
        tColor_local = nb.cuda.local.array((size), nb.int16)
        gamma = nb.cuda.local.array((size, k), nb.int8)
        for x in range(size):
            for y in range(k):
                gamma[x, y] = 0
                tabuTenure[d, x, y] = -1
            tColor_local[x] = int(tColor[d, x])
        f = 0
        for x in range(size):
            for y in range(x):
                if A[x, y] == 1:
                    gamma[x, tColor_local[y]] += 1
                    gamma[y, tColor_local[x]] += 1
                    if tColor_local[y] == tColor_local[x]:
                        f += 1
        f_best = f
        for iter_ in range(max_iter):
            best_delta = 9999
            best_x = -1
            best_v = -1
            nbcfl = 0
            for x in range(size):
                v_x = tColor_local[x]
                if gamma[x, v_x] > 0:
                    nbcfl += 1
                    for v in range(k):
                        if v != v_x:
                            delta = gamma[x, v] - gamma[x, v_x]
                            if tabuTenure[d, x, v] <= iter_ or delta + f < f_best:
                                if delta < best_delta:
                                    best_x = x
                                    best_v = v
                                    best_delta = delta
            f += best_delta
            old_color = tColor_local[best_x]
            for y in range(size):
                if A[best_x, y] == 1:
                    gamma[y, old_color] -= 1
                    gamma[y, best_v] += 1
            tColor_local[best_x] = best_v
            tabuTenure[d, best_x, old_color] = (
                int(alpha * nbcfl)
                + int(10 * xoroshiro128p_uniform_float32(rng_states, d))
                + iter_
            )
            if f < f_best:
                f_best = f
                for a in range(size):
                    tColor[d, a] = tColor_local[a]
        vect_fit[d] = f_best


@cuda.jit
def tabuWVCP_NoRandom_AFISA_bigSize(
    rng_states,
    D,
    max_iter,
    A,
    W,
    tColor,
    vect_fit,
    vect_score,
    vect_conflicts,
    alpha,
    phi,
    gamma,
):
    d = cuda.grid(1)
    if d < D:
        f = 0
        tColor_local = nb.cuda.local.array((size), nb.int16)
        gammaDepart = nb.cuda.local.array((size), nb.int8)
        gammaArrive = nb.cuda.local.array((size, k), nb.int8)
        max_weight = nb.cuda.local.array((k), nb.int8)
        secondmax_weight = nb.cuda.local.array((k), nb.int8)
        tabuTenure = nb.cuda.local.array((size), nb.int32)
        tGroup_color = nb.cuda.local.array((k, N), nb.int16)
        for c in range(k):
            max_weight[c] = 0
            secondmax_weight[c] = 0
            for i in range(N):
                tGroup_color[c, i] = -1
        for x in range(size):
            gammaDepart[x] = 0
            for y in range(k):
                gamma[d, x, y] = 0
                gammaArrive[x, y] = 0
            tColor_local[x] = int(tColor[d, x])
            idx = 0
            while tGroup_color[tColor_local[x], idx] != -1:
                idx += 1
            tGroup_color[tColor_local[x], idx] = x
            tabuTenure[x] = -1
        nb_conflicts = 0
        score_wvcp = 0
        for x in range(size):
            for y in range(x):
                if A[x, y] == 1:
                    gamma[d, x, tColor_local[y]] += 1
                    gamma[d, y, tColor_local[x]] += 1
                    if tColor_local[y] == tColor_local[x]:
                        nb_conflicts += 1
            if W[x] >= max_weight[tColor_local[x]]:
                score_wvcp += W[x] - max_weight[tColor_local[x]]
                secondmax_weight[tColor_local[x]] = max_weight[tColor_local[x]]
                max_weight[tColor_local[x]] = int(W[x])
            elif W[x] > secondmax_weight[tColor_local[x]]:
                secondmax_weight[tColor_local[x]] = int(W[x])
        for x in range(size):
            for c in range(k):
                if W[x] > max_weight[c]:
                    gammaArrive[x, c] = W[x] - max_weight[c]
                else:
                    gammaArrive[x, c] = 0
        for x in range(size):
            if W[x] == max_weight[tColor_local[x]]:
                gammaDepart[x] = secondmax_weight[tColor_local[x]] - W[x]
            else:
                gammaDepart[x] = 0
        f = score_wvcp + phi[d] * nb_conflicts
        f_best = f
        score_best = score_wvcp
        nb_conflicts_best = nb_conflicts
        for iter_ in range(max_iter):
            best_delta = 9999
            best_delta_conflicts = -1
            best_delta_score = -1
            best_x = -1
            best_v = -1
            for x in range(size):
                v_x = tColor_local[x]
                for v in range(k):
                    if v != v_x:
                        delta_score = gammaArrive[x, v] + gammaDepart[x]
                        delta_conflicts = gamma[d, x, v] - gamma[d, x, v_x]
                        delta = delta_score + phi[d] * delta_conflicts
                        if tabuTenure[x] <= iter_ or delta + f < f_best:
                            if delta < best_delta:
                                best_x = x
                                best_v = v
                                best_delta = delta
                                best_delta_conflicts = delta_conflicts
                                best_delta_score = delta_score
            f += best_delta
            score_wvcp += best_delta_score
            nb_conflicts += best_delta_conflicts
            old_color = tColor_local[best_x]
            for y in range(size):
                if A[best_x, y] == 1:
                    gamma[d, y, old_color] -= 1
                    gamma[d, y, best_v] += 1
            tColor_local[best_x] = best_v
            old_max_old_color = max_weight[old_color]
            old_second_max_old_color = secondmax_weight[old_color]
            max_weight[old_color] = 0
            secondmax_weight[old_color] = 0
            for idx in range(N):
                x = tGroup_color[old_color, idx]
                if x == best_x:
                    tGroup_color[old_color, idx] = -1
                elif x != -1:
                    if W[x] >= max_weight[old_color]:
                        secondmax_weight[old_color] = max_weight[old_color]
                        max_weight[old_color] = int(W[x])
                    elif W[x] > secondmax_weight[old_color]:
                        secondmax_weight[old_color] = int(W[x])
            idx = 0
            while tGroup_color[best_v, idx] != -1:
                idx += 1
            tGroup_color[best_v, idx] = best_x
            old_max_best_v = max_weight[best_v]
            if W[best_x] >= max_weight[best_v]:
                secondmax_weight[best_v] = max_weight[best_v]
                max_weight[best_v] = int(W[best_x])
            elif W[best_x] > secondmax_weight[best_v]:
                secondmax_weight[best_v] = int(W[best_x])
            if max_weight[old_color] != old_max_old_color:
                for x in range(size):
                    if W[x] >= max_weight[old_color]:
                        gammaArrive[x, old_color] = W[x] - max_weight[old_color]
                    else:
                        gammaArrive[x, old_color] = 0
            if max_weight[best_v] != old_max_best_v:
                for x in range(size):
                    if W[x] >= max_weight[best_v]:
                        gammaArrive[x, best_v] = W[x] - max_weight[best_v]
                    else:
                        gammaArrive[x, best_v] = 0
            if (
                old_second_max_old_color != secondmax_weight[old_color]
                or max_weight[old_color] != old_max_old_color
            ):
                for idx in range(N):
                    x = tGroup_color[old_color, idx]
                    if x != -1:
                        if W[x] == max_weight[old_color]:
                            gammaDepart[x] = secondmax_weight[old_color] - W[x]
                        else:
                            gammaDepart[x] = 0
            for idx in range(N):
                x = tGroup_color[best_v, idx]
                if x != -1:
                    if W[x] == max_weight[best_v]:
                        gammaDepart[x] = secondmax_weight[best_v] - W[x]
                    else:
                        gammaDepart[x] = 0
            tabuTenure[best_x] = (
                int(alpha * size)
                + int(10 * xoroshiro128p_uniform_float32(rng_states, d))
                + iter_
            )
            if f < f_best:
                f_best = f
                score_best = score_wvcp
                nb_conflicts_best = nb_conflicts
                for a in range(size):
                    tColor[d, a] = tColor_local[a]
        vect_fit[d] = f_best
        vect_score[d] = score_best
        vect_conflicts[d] = nb_conflicts_best


@cuda.jit
def tabuWVCP_NoRandom_AFISA_heavyWeights(
    rng_states,
    D,
    max_iter,
    A,
    W,
    tColor,
    vect_fit,
    vect_score,
    vect_conflicts,
    alpha,
    phi,
):
    # vect_nb_vois, voisin
    d = cuda.grid(1)
    if d < D:
        f = 0
        tColor_local = nb.cuda.local.array((size), nb.int16)
        gamma = nb.cuda.local.array((size, k), nb.int8)
        gammaDepart = nb.cuda.local.array((size), nb.int16)
        gammaArrive = nb.cuda.local.array((size, k), nb.int16)
        max_weight = nb.cuda.local.array((k), nb.int16)
        secondmax_weight = nb.cuda.local.array((k), nb.int16)
        tabuTenure = nb.cuda.local.array((size), nb.int32)
        tGroup_color = nb.cuda.local.array((k, N), nb.int16)
        for c in range(k):
            max_weight[c] = 0
            secondmax_weight[c] = 0
            for i in range(N):
                tGroup_color[c, i] = -1
        for x in range(size):
            gammaDepart[x] = 0
            for y in range(k):
                gamma[x, y] = 0
                gammaArrive[x, y] = 0
            tColor_local[x] = int(tColor[d, x])
            idx = 0
            while tGroup_color[tColor_local[x], idx] != -1:
                idx += 1
            tGroup_color[tColor_local[x], idx] = x
            tabuTenure[x] = -1
        nb_conflicts = 0
        score_wvcp = 0
        for x in range(size):
            # nb_vois = int(vect_nb_vois[x])
            # for i in range(nb_vois):
            # y = int(voisin[i])
            for y in range(x):
                if A[x, y] == 1:
                    gamma[x, tColor_local[y]] += 1
                    gamma[y, tColor_local[x]] += 1
                    if tColor_local[y] == tColor_local[x]:
                        nb_conflicts += 1
            if W[x] >= max_weight[tColor_local[x]]:
                score_wvcp += W[x] - max_weight[tColor_local[x]]
                secondmax_weight[tColor_local[x]] = max_weight[tColor_local[x]]
                max_weight[tColor_local[x]] = int(W[x])
            elif W[x] > secondmax_weight[tColor_local[x]]:
                secondmax_weight[tColor_local[x]] = int(W[x])
        for x in range(size):
            for c in range(k):
                if W[x] > max_weight[c]:
                    gammaArrive[x, c] = W[x] - max_weight[c]
                else:
                    gammaArrive[x, c] = 0
        for x in range(size):
            if W[x] == max_weight[tColor_local[x]]:
                gammaDepart[x] = secondmax_weight[tColor_local[x]] - W[x]
            else:
                gammaDepart[x] = 0
        f = score_wvcp + phi[d] * nb_conflicts
        f_best = f
        score_best = score_wvcp
        nb_conflicts_best = nb_conflicts
        for iter_ in range(max_iter):
            best_delta = 99999
            best_delta_conflicts = -1
            best_delta_score = -1
            best_x = -1
            best_v = -1
            for x in range(size):
                v_x = tColor_local[x]
                for v in range(k):
                    if v != v_x:
                        delta_score = gammaArrive[x, v] + gammaDepart[x]
                        delta_conflicts = gamma[x, v] - gamma[x, v_x]
                        delta = delta_score + phi[d] * delta_conflicts
                        if tabuTenure[x] <= iter_ or delta + f < f_best:
                            if delta < best_delta:
                                best_x = x
                                best_v = v
                                best_delta = delta
                                best_delta_conflicts = delta_conflicts
                                best_delta_score = delta_score
            f += best_delta
            score_wvcp += best_delta_score
            nb_conflicts += best_delta_conflicts
            old_color = tColor_local[best_x]
            for y in range(size):
                if A[best_x, y] == 1:
                    gamma[y, old_color] -= 1
                    gamma[y, best_v] += 1
            tColor_local[best_x] = best_v
            old_max_old_color = max_weight[old_color]
            old_second_max_old_color = secondmax_weight[old_color]
            max_weight[old_color] = 0
            secondmax_weight[old_color] = 0
            for idx in range(N):
                x = tGroup_color[old_color, idx]
                if x == best_x:
                    tGroup_color[old_color, idx] = -1
                elif x != -1:
                    if W[x] >= max_weight[old_color]:
                        secondmax_weight[old_color] = max_weight[old_color]
                        max_weight[old_color] = int(W[x])
                    elif W[x] > secondmax_weight[old_color]:
                        secondmax_weight[old_color] = int(W[x])
            idx = 0
            while tGroup_color[best_v, idx] != -1:
                idx += 1
            tGroup_color[best_v, idx] = best_x
            old_max_best_v = max_weight[best_v]
            if W[best_x] >= max_weight[best_v]:
                secondmax_weight[best_v] = max_weight[best_v]
                max_weight[best_v] = int(W[best_x])
            elif W[best_x] > secondmax_weight[best_v]:
                secondmax_weight[best_v] = int(W[best_x])
            if max_weight[old_color] != old_max_old_color:
                for x in range(size):
                    if W[x] >= max_weight[old_color]:
                        gammaArrive[x, old_color] = W[x] - max_weight[old_color]
                    else:
                        gammaArrive[x, old_color] = 0
            if max_weight[best_v] != old_max_best_v:
                for x in range(size):
                    if W[x] >= max_weight[best_v]:
                        gammaArrive[x, best_v] = W[x] - max_weight[best_v]
                    else:
                        gammaArrive[x, best_v] = 0
            if (
                old_second_max_old_color != secondmax_weight[old_color]
                or max_weight[old_color] != old_max_old_color
            ):
                for idx in range(N):
                    x = tGroup_color[old_color, idx]
                    if x != -1:
                        if W[x] == max_weight[old_color]:
                            gammaDepart[x] = secondmax_weight[old_color] - W[x]
                        else:
                            gammaDepart[x] = 0
            for idx in range(N):
                x = tGroup_color[best_v, idx]
                if x != -1:
                    if W[x] == max_weight[best_v]:
                        gammaDepart[x] = secondmax_weight[best_v] - W[x]
                    else:
                        gammaDepart[x] = 0
            tabuTenure[best_x] = (
                int(alpha * size)
                + int(10 * xoroshiro128p_uniform_float32(rng_states, d))
                + iter_
            )
            if f < f_best:
                f_best = f
                score_best = score_wvcp
                nb_conflicts_best = nb_conflicts

                for a in range(size):
                    tColor[d, a] = tColor_local[a]
        vect_fit[d] = f_best
        vect_score[d] = score_best
        vect_conflicts[d] = nb_conflicts_best
