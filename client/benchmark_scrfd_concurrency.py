import time
import multiprocessing

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

# ===============================
# Configuration
# ===============================
MODEL_NAME = "scrfd"
HTTP_URL = "triton:8000"
GRPC_URL = "triton:8001"

BATCH_SIZES = [1, 4, 8]
NUM_ITERATIONS = 1000
NUM_WARMUP = 10
NUM_PROCESSES = 3


# ===============================
# Test Data
# ===============================
def create_test_data(batch_size=1):
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img = img.astype(np.float32)
    img = img.transpose(2, 0, 1)[np.newaxis, ...]
    return np.repeat(img, batch_size, axis=0)


# ===============================
# Worker Process Function
# ===============================
def worker_process(protocol, batch_size, num_iterations, num_warmup, process_id, result_queue):
    """Worker process that runs benchmark with its own client"""
    try:
        # Create client for this process
        if protocol == "HTTP":
            client = httpclient.InferenceServerClient(url=HTTP_URL)
            infer_input_cls = httpclient.InferInput
            infer_output_cls = httpclient.InferRequestedOutput
        else:  # gRPC
            client = grpcclient.InferenceServerClient(url=GRPC_URL)
            infer_input_cls = grpcclient.InferInput
            infer_output_cls = grpcclient.InferRequestedOutput

        # Prepare test data
        img_batch = create_test_data(batch_size)
        inputs = [infer_input_cls("input.1", img_batch.shape, "FP32")]
        inputs[0].set_data_from_numpy(img_batch)
        outputs = [infer_output_cls("score_8")]

        # Warmup
        for _ in range(num_warmup):
            client.infer(MODEL_NAME, inputs, outputs=outputs)

        # Run benchmark
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            client.infer(MODEL_NAME, inputs, outputs=outputs)
            latencies.append((time.perf_counter() - start) * 1000)

        # Close client
        client.close()

        # Send results back
        result_queue.put((process_id, np.array(latencies)))

    except Exception as e:
        result_queue.put((process_id, f"Error: {str(e)}"))


# ===============================
# Run Concurrent Benchmarks
# ===============================
def run_concurrent_benchmark(protocol, batch_size, num_iterations, num_warmup, num_processes):
    """Run benchmark with multiple processes"""
    result_queue = multiprocessing.Queue()
    processes = []

    # Start all processes
    for i in range(num_processes):
        p = multiprocessing.Process(
            target=worker_process,
            args=(protocol, batch_size, num_iterations, num_warmup, i, result_queue)
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Collect results from all processes
    all_latencies = []
    for _ in range(num_processes):
        process_id, result = result_queue.get()
        if isinstance(result, np.ndarray):
            all_latencies.extend(result.tolist())
        else:
            print(f"  Process {process_id} failed: {result}")

    if all_latencies:
        latencies_array = np.array(all_latencies)
        return latencies_array
    else:
        return np.array([])


# ===============================
# Main Function
# ===============================
def main():
    batch_sizes = BATCH_SIZES
    num_iterations = NUM_ITERATIONS
    num_warmup = NUM_WARMUP
    num_processes = NUM_PROCESSES

    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Running concurrent benchmarks for batch size: {batch_size}")
        print(f"  Processes: {num_processes} | Iterations per process: {num_iterations}")
        print(f"{'='*60}")

        # Run HTTP benchmark with 3 processes
        print(f"  Running HTTP benchmark with {num_processes} processes...")
        http_lat = run_concurrent_benchmark(
            "HTTP", batch_size, num_iterations, num_warmup, num_processes
        )

        # Run gRPC benchmark with 3 processes
        print(f"  Running gRPC benchmark with {num_processes} processes...")
        grpc_lat = run_concurrent_benchmark(
            "gRPC", batch_size, num_iterations, num_warmup, num_processes
        )

        # Print results
        print(f"\n  Batch Size {batch_size} Results (Concurrent - {num_processes} processes):")
        if len(http_lat) > 0:
            print(f"    HTTP  - Latency: mean={http_lat.mean():.2f} ms, p95={np.percentile(http_lat, 95):.2f} ms")
        else:
            print(f"    HTTP  - No valid results")
        
        if len(grpc_lat) > 0:
            print(f"    gRPC  - Latency: mean={grpc_lat.mean():.2f} ms, p95={np.percentile(grpc_lat, 95):.2f} ms")
        else:
            print(f"    gRPC  - No valid results")

    print(f"\n{'='*60}")
    print("Benchmark completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

