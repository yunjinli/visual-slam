#!/usr/bin/env python3

import json
import sys

o1 = json.load(open(sys.argv[1]))
o2 = json.load(open(sys.argv[2]))

def approx_equal_json(x, y, tol=1e-6):
    if isinstance(x, dict):
        equal = x.keys() == y.keys()
        if equal:
            for i in x.keys():
                equal = equal and approx_equal_json(x[i], y[i])
        return equal
    elif isinstance(x, list):
        equal = len(x) == len(y)
        if equal:
            for i in range(len(x)):
                equal = equal and approx_equal_json(x[i], y[i])
        return equal
    elif isinstance(x, float):
        return abs(x-y) < tol
    else:
        return x == y

if not approx_equal_json(o1, o2):
    print('JSON files are not the same')
    sys.exit(-1)
