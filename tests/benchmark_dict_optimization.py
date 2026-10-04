import timeit
import collections
import random

class MockModel:
    def __init__(self, model_type):
        self.model_type = model_type

def run_benchmark(pool_size, iterations):
    model_pool = [MockModel(f"type_{i % 5}") for i in range(pool_size)]

    def current_impl():
        by_type = {}
        for m in model_pool:
            if m.model_type not in by_type:
                by_type[m.model_type] = []
            by_type[m.model_type].append(m)
        return by_type

    def defaultdict_impl():
        by_type = collections.defaultdict(list)
        for m in model_pool:
            by_type[m.model_type].append(m)
        return by_type

    def setdefault_impl():
        by_type = {}
        for m in model_pool:
            by_type.setdefault(m.model_type, []).append(m)
        return by_type

    t1 = timeit.timeit(current_impl, number=iterations)
    t2 = timeit.timeit(defaultdict_impl, number=iterations)
    t3 = timeit.timeit(setdefault_impl, number=iterations)

    print(f"Pool size: {pool_size}, Iterations: {iterations}")
    print(f"  Current (if not in):      {t1:.6f}s")
    print(f"  defaultdict(list):        {t2:.6f}s")
    print(f"  dict.setdefault([], ...): {t3:.6f}s")

    best = min(t1, t2, t3)
    if best == t1: best_name = "Current"
    elif best == t2: best_name = "defaultdict"
    else: best_name = "setdefault"

    print(f"  Best: {best_name}")
    print("-" * 30)

if __name__ == "__main__":
    run_benchmark(4, 100000)
    run_benchmark(10, 100000)
    run_benchmark(100, 10000)
