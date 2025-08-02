import numpy as np
import time

def main():
    # Matrix size (tweak to stress system)
    N = 20000

    # Create random matrices
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    # Warm-up
    _ = np.dot(A, B)

    # Timed run
    start = time.perf_counter()
    C = np.dot(A, B)
    end = time.perf_counter()

    elapsed = end - start
    gflops = (2 * N**3) / (elapsed * 1e9)
    return elapsed, gflops

if __name__ == "__main__":
    elapsed, gflops = main()
    print(f"GPU time: {elapsed:.4f} s")
    print(f"Performance: {gflops:.2f} GFLOP/s")
