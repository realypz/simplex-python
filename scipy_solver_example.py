# -*- coding: utf-8 -*-
import unittest
import numpy as np
from scipy.optimize import linprog
import json
from util import LoadSingleTestExample

if __name__ == "__main__":

    with open("test_examples.json", "r") as f:
        test_data = json.load(f)

    c, A, b = LoadSingleTestExample(test_data[0])  # Todo: change the test example
    n = len(c)

    x_bounds = [(0, None) for i in range(n)]
    res = linprog(c, A_eq=A, b_eq=b, bounds=x_bounds)

    print(res)

    pass