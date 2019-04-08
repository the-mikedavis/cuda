#! /usr/bin/env python

import random
import argparse

if __name__ == '__main__':
    random_seed = random.randint(1, 262571)
    parser = argparse.ArgumentParser(description="Cuda based correspondence grouping")
    parser.add_argument('--seed', dest='seed', type=int, default=random_seed,
                        help="Seed value for the random generator")
    args = parser.parse_args()

    random.seed(args.seed)
