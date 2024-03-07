import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.cluster.hierarchy import fclusterdata


def find_roots_1d(f,
                  lower,
                  upper,
                  *args,
                  gp=1000,
                  res=1e-3,
                  **kwargs):
    # print(args)
    x_axis = np.linspace(lower, upper, gp)
    mask = np.diff(f(x_axis, *args, **kwargs) < 0)
    starters = x_axis[:-1][mask]
    solutions = np.zeros(len(starters))

    def fun(x): return f(x, *args, **kwargs)

    for i, s in enumerate(starters):
        solutions[i] = fsolve(fun, s, maxfev=1000)
    if solutions.size == 0:
        #print('no solutions found')
        return np.array([])
    else:
        return solutions
    # elif solutions.shape[0] == 1:
    #    return solutions

    # else:
    #     cluster = fclusterdata(solutions[:, None], res,
    #                            criterion='distance')
    #     n = int(np.max(cluster))
    #     averaged_solutions = np.zeros(n)
    #     for c in range(n):
    #         averaged_solutions[c] = np.mean(solutions[cluster == c+1])
    #     return averaged_solutions


def find_roots_2d(f, g, x_axis, y_axis,
                  fargs=(),
                  gargs=(),
                  fkwargs={},
                  gkwargs={},
                  res=1e-1):
    xx, yy = np.meshgrid(x_axis, y_axis)

    ff = f(xx, yy, *fargs, **fkwargs)
    gg = g(xx, yy, *gargs, **gkwargs)

    mask_f = ff < 0
    mask_g = gg < 0

    border_x_f = np.diff(mask_f, axis=1)
    border_y_f = np.diff(mask_f, axis=0)
    border_x_g = np.diff(mask_g, axis=1)
    border_y_g = np.diff(mask_g, axis=0)

    border_f = np.logical_or(border_x_f[:-1], border_y_f[:, :-1])
    border_g = np.logical_or(border_x_g[:-1], border_y_g[:, :-1])

    #fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    #ax[0].contourf(xx[:-1, :-1], yy[:-1, :-1], border_f)
    #ax[1].contourf(xx[:-1, :-1], yy[:-1, :-1], border_g)

    border = np.logical_and(border_f, border_g)

    starters = list(zip(xx[:-1, :-1][border], yy[:-1, :-1][border]))

    solutions = []

    def vec_fun(x): return (f(x[0], x[1], *fargs, **fkwargs),
                            g(x[0], x[1], *gargs, **gkwargs))

    for s in starters:

        solutions.append(fsolve(vec_fun, s))
    solutions = np.array(solutions)

    if solutions.size == 0:
        print('no solutions found')
        return np.array([])

    else:
        return solutions
    # elif solutions.size == 2:
    #     return solutions

    # else:
    #     cluster = fclusterdata(solutions, res, criterion='distance')
    #     n = np.max(cluster)
    #     averaged_solutions = np.zeros((n, 2))
    #     for i in range(n):
    #         averaged_solutions[i] = np.mean(solutions[cluster == i+1],
    #                                         axis=0)

    #     return averaged_solutions
