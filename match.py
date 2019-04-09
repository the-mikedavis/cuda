#! /usr/bin/env python3

import random
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("pdf")

def generate_transformation():
    scale = 20
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
    points = 3
    scale = 10
    target = np.matrix([np.random.random(points) * scale,
                        np.random.random(points) * scale,
                        np.ones(points)])
    return (target, transformation * target)

def show_dataset(dataset, name="dataset"):
    a, b = dataset
    plt.scatter(np.squeeze(np.asarray(a[0, :])), np.squeeze(np.asarray(a[1, :])))
    plt.scatter(np.squeeze(np.asarray(b[0, :])), np.squeeze(np.asarray(b[1, :])))
    plt.title(name)
    plt.savefig(name + ".png")
    plt.show()

def nearest_neighbors(dataset):
    "Find the nearest neighbors for each point in a and b"
    return None

def calculate_error(dataset, nearest_neighbors):
    return None

def optimize(dataset, nearest_neighbors, error):
    return None

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

    # while True:
        # nn = nearest_neighbors(dataset)
        # error = calculate_error(dataset, nn)
        # optimization = optimize(dataset, nn, error)
        # TODO adjust the dataset by the optimization
        # show_dataset(updated_dataset, name="dataset_%d"%iteration)
    # show_dataset(solution, name="final")