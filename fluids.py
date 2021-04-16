import math

import imageio as imageio
import matplotlib.pyplot as plt
import numpy as np

size = 100
dt = 0.2
gauss_seidel_iterations = 2

diffusion = 0.0000
viscosity = 0.0000

density_prev = np.full((size, size), 0, dtype=float)
density_curr = np.full((size, size), 0, dtype=float)

velocity_prev = np.full((size, size, 2), 0, dtype=float)
velocity_curr = np.full((size, size, 2), 0, dtype=float)


def gauss_seidel_solve(x, x0, a, c):
    for iteration in range(0, gauss_seidel_iterations):
        x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (x[2:, 1:-1] + x[:-2, 1:-1] + x[1:-1, 2:] + x[1:-1, :-2])) * 1 / c
        set_boundaries(x)


def set_boundaries(table):
    if len(table.shape) > 2:
        table[:, 0, 0] = table[:, 1, 0]
        table[:, 0, 1] = - table[:, 1, 1]
        table[:, -1, 0] = table[:, -2, 0]
        table[:, -1, 1] = - table[:, -2, 1]

        table[0, :, 0] = - table[1, :, 0]
        table[0, :, 1] = table[1, :, 1]
        table[-1, :, 0] = - table[-2, :, 0]
        table[-1, :, 1] = table[-2, :, 1]

    else:
        table[:, 0] = table[:, 1]
        table[:, -1] = table[:, -2]

        table[0, :] = table[1, :]
        table[-1, :] = table[-2, :]

    table[0, 0] = 0.5 * (table[1, 0] + table[0, 1])
    table[0, -1] = 0.5 * (table[1, -1] + table[0, -2])
    table[-1, 0] = 0.5 * (table[-2, 0] + table[- 1, 1])
    table[-1, -1] = 0.5 * (table[-2, -1] + table[-1, -2])


def step():
    diffuse(velocity_curr, velocity_prev, viscosity)

    project(velocity_curr[:, :, 0], velocity_curr[:, :, 1], velocity_prev[:, :, 0], velocity_prev[:, :, 1])

    advect(velocity_prev[:, :, 0], velocity_curr[:, :, 0], velocity_curr)
    advect(velocity_prev[:, :, 1], velocity_curr[:, :, 1], velocity_curr)

    project(velocity_prev[:, :, 0], velocity_prev[:, :, 1], velocity_curr[:, :, 0], velocity_curr[:, :, 1])

    diffuse(density_prev, density_curr, diffusion)

    advect(density_curr, density_prev, velocity_prev)


def diffuse(x, x0, diff):
    if diff != 0:
        a = dt * diff * (size - 2) * (size - 2)
        gauss_seidel_solve(x, x0, a, 1 + 6 * a)
    else:
        x[:, :] = x0[:, :]


def project(velocity_x, velocity_y, p, div):
    div[1:-1, 1:-1] = -0.5 * (
            velocity_x[2:, 1:-1] -
            velocity_x[:-2, 1:-1] +
            velocity_y[1:-1, 2:] -
            velocity_y[1:-1, :-2]) \
                      / size
    p[:, :] = 0
    set_boundaries(div)
    set_boundaries(p)
    gauss_seidel_solve(p, div, 1, 6)
    velocity_x[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) * size
    velocity_y[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) * size
    set_boundaries(velocity_prev)


def advect(d, d0, velocity):
    dtx = dt * (size - 2)
    dty = dt * (size - 2)
    for j in range(1, size - 1):
        for i in range(1, size - 1):
            x = i - dtx * velocity[i, j, 0]
            y = j - dty * velocity[i, j, 1]
            if x < 0.5:
                x = 0.5
            if x > size + 0.5:
                x = size + 0.5
            i0 = math.floor(x)
            if y < 0.5: y = 0.5
            if y > size + 0.5: y = size + 0.5
            j0 = math.floor(y)
            s1 = x - i0
            t1 = y - j0
            t0 = 1.0 - t1
            i0i = int(i0)
            i1i = int(i0 + 1.0)
            j0i = int(j0)
            j1i = int(j0 + 1.0)
            d[i, j] = 1.0 - s1 * (t0 * d0[i0i, j0i] + t1 * d0[i0i, j1i]) + \
                      s1 * (t0 * d0[i1i, j0i] + t1 * d0[i1i, j1i])
    set_boundaries(d)


if __name__ == "__main__":
    frames = 150


    def calculate_frame(frame):
        density_curr[4:7, 4:7] += 100
        if frame % 2 == 0:
            velocity_prev[5, 5] += [10, 0]
        else:
            velocity_prev[5, 5] += [0, 10]

        step()


    video = np.full((frames, size, size), 0, dtype=float)

    for i in range(frames):
        calculate_frame(i)
        video[i] = density_curr

    plt.imshow(density_curr, cmap='hot', vmax=100, interpolation='bilinear')
    plt.show()

    imageio.mimsave('./video.gif', video.astype('uint8'))
