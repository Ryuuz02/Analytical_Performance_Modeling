import cupy as cp
import time

def main():

    # Matrix size (same as CPU test)
    N = 20000

    # Create random matrices on GPU
    A_gpu = cp.random.rand(N, N, dtype=cp.float32)
    B_gpu = cp.random.rand(N, N, dtype=cp.float32)

    # Warm-up (ensures kernels are compiled)
    _ = cp.dot(A_gpu, B_gpu)

    # Timed run
    cp.cuda.Device(0).synchronize()  # make sure GPU is ready
    start = time.perf_counter()
    C_gpu = cp.dot(A_gpu, B_gpu)
    cp.cuda.Device(0).synchronize()  # wait for GPU to finish
    end = time.perf_counter()

    elapsed = end - start
    gflops = (2 * N**3) / (elapsed * 1e9)
    return elapsed, gflops

if __name__ == "__main__":
    elapsed, gflops = main()
    print(f"GPU time: {elapsed:.4f} s")
    print(f"Performance: {gflops:.2f} GFLOP/s")
