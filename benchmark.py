import time
import importlib
import statistics
from pathlib import Path
from tabulate import tabulate

# Which scripts to benchmark
IMPLEMENTATIONS = {
    "CPU Matrix Multiplication": "matrix_multiplication_CPU",
    "GPU Matrix Multiplication": "matrix_multiplication_GPU",
}

# How many runs to average
NUM_RUNS = 3

def run_module(module_name: str):
    """Run the module's main() function and return execution time."""
    mod = importlib.import_module(module_name)
    t, _ = mod.main()  # Must have a main() function in the module
    return t

def main():
    results = []
    for label, module_name in IMPLEMENTATIONS.items():
        print(f"\nBenchmarking {label}...")
        times = []
        for i in range(NUM_RUNS):
            print(f"  Run {i+1}/{NUM_RUNS}...")
            t = run_module(module_name)
            times.append(t)
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        results.append([label, f"{avg_time:.2f}", f"{std_dev:.2f}"])

    print("\nBenchmark Results:")
    print(tabulate(results, headers=["Implementation", "Avg Time (s)", "Std Dev (s)"], tablefmt="github"))

if __name__ == "__main__":
    main()
