import random
import time

from montage.core.rover_linear import rover_merge_original as rover_merge


def synthetic(n=10_000):
    return [[(i*0.1, i*0.1+0.05, str(i), random.random()) for i in range(n)]]

def test_perf():
    start = time.time()
    rover_merge(synthetic())
    assert time.time() - start < 0.2          # ≤200 ms on M4 Max
