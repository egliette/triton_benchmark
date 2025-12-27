import time

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

# ===============================
# Configuration
# ===============================
MODEL_NAME = "arcface"
HTTP_URL = "triton:8000"
GRPC_URL = "triton:8001"

BATCH_SIZES = [1, 4, 8]
NUM_ITERATIONS = 1000
NUM_WARMUP = 10


# ===============================
# Test Data
# ===============================
def create_test_data(batch_size=1):
    img = np.zeros((112, 112, 3), dtype=np.uint8)
    img = img.astype(np.float32)
    img = img.transpose(2, 0, 1)[np.newaxis, ...]
    return np.repeat(img, batch_size, axis=0)


# ===============================
# Generic Benchmark Function
# ===============================
def benchmark(client, infer_input_cls, infer_output_cls, batch_size, num_iterations, num_warmup):
    img_batch = create_test_data(batch_size)

    inputs = [infer_input_cls("input.1", img_batch.shape, "FP32")]
    inputs[0].set_data_from_numpy(img_batch)
    outputs = [infer_output_cls("683")]

    # Warmup
    for _ in range(num_warmup):
        client.infer(MODEL_NAME, inputs, outputs=outputs)

    latencies = []

    for _ in range(num_iterations):
        start = time.perf_counter()
        client.infer(MODEL_NAME, inputs, outputs=outputs)
        latencies.append((time.perf_counter() - start) * 1000)

    return np.array(latencies)


# ===============================
# Run Benchmarks
# ===============================
def main():
    batch_sizes = BATCH_SIZES
    num_iterations = NUM_ITERATIONS
    num_warmup = NUM_WARMUP

    http_client = httpclient.InferenceServerClient(url=HTTP_URL)
    grpc_client = grpcclient.InferenceServerClient(url=GRPC_URL)

    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Running benchmarks for batch size: {batch_size}")
        print(f"{'='*60}")


        # Run HTTP benchmark
        print(f"  Running HTTP benchmark...")
        http_lat = benchmark(
            http_client,
            httpclient.InferInput,
            httpclient.InferRequestedOutput,
            batch_size,
            num_iterations,
            num_warmup,
        )

        # Run gRPC benchmark
        print(f"  Running gRPC benchmark...")
        grpc_lat = benchmark(
            grpc_client,
            grpcclient.InferInput,
            grpcclient.InferRequestedOutput,
            batch_size,
            num_iterations,
            num_warmup,
        )

        # Print results
        print(f"\n  Batch Size {batch_size} Results:")
        print(f"    HTTP  - Latency: mean={http_lat.mean():.2f} ms, p95={np.percentile(http_lat, 95):.2f} ms")
        print(f"    gRPC  - Latency: mean={grpc_lat.mean():.2f} ms, p95={np.percentile(grpc_lat, 95):.2f} ms")

    print(f"\n{'='*60}")
    print("Benchmark completed!")
    print(f"{'='*60}\n")

    http_client.close()
    grpc_client.close()


if __name__ == "__main__":
    main()
