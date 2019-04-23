#! /usr/bin/env python2

import random
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("pdf")

#optimization
import gaussnewton as gn
import parallel_nearest_neighbor as nn

def generate_transformation():
    scale = 50
    degrees = np.random.randint(0, 360)
    print("theta: %d"%degrees)
    theta = np.radians(degrees)
    c = np.cos(theta)
    s = np.sin(theta)
    x = np.random.randint(-scale, scale)
    y = np.random.randint(-scale, scale)
    return np.matrix([[c, -s, x], [s, c, y], [0, 0, 1]])

def generate_dataset(transformation):
    "Randomly generates a dataset."
    points = 100
    scale = 100
    target = np.matrix([np.random.random(points).astype(np.float64) * scale,
                        np.random.random(points).astype(np.float64) * scale,
                        np.ones(points)])
    #return (target[:,0:points/2], transformation * target[:,points/8:(points/8 + points/2)])
    return (target, transformation * target)

def show_dataset(dataset, name="dataset", subplot = False):

    if subplot:
        a, b, r = dataset
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.scatter(np.squeeze(np.asarray(a[0, :])), np.squeeze(np.asarray(a[1, :])))
            plt.scatter(np.squeeze(np.asarray(r[i][0, :])), np.squeeze(np.asarray(r[i][1, :])))
            plt.axis((-200, 200, -200, 200))
    else:
        a, b = dataset
        plt.scatter(np.squeeze(np.asarray(a[0, :])), np.squeeze(np.asarray(a[1, :])))
        plt.scatter(np.squeeze(np.asarray(b[0, :])), np.squeeze(np.asarray(b[1, :])))
        plt.axis((-200, 200, -200, 200))

    plt.title(name)
    plt.savefig(name + ".png")
    plt.show()


def optimize(dataset, nearest_neighbors=None, error=None):
    a, b = dataset
    r, Y_match, sol, its = gn.solve(a, b)
    print("  Iterations : {}".format(its))
    print("  Calculated : {}".format(sol))
    return (a, Y_match, r)

if __name__ == '__main__':
    random_seed = random.randint(1, 262571)
    parser = argparse.ArgumentParser(description="Cuda based correspondence grouping")
    parser.add_argument('--seed', dest='seed', type=int, default=random_seed,
                        help="Seed value for the random generator")
    args = parser.parse_args()

    print("Seed: %d"%args.seed)
    random.seed(a=args.seed)
    np.random.seed(seed=args.seed)

    transformation = generate_transformation()

    print("Transformation matrix:")
    print(transformation)

    dataset = generate_dataset(transformation)

    show_dataset(dataset)

    opt_dataset = optimize(dataset)

    show_dataset(opt_dataset, name = "result", subplot = True)
