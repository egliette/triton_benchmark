# CLI Commands (example for batch size 1):
# HTTP: perf_analyzer -m scrfd -u triton:8000 -i http \
#   --warmup-request-count=100 --request-count=100 \
#   --max-threads=3 --shape=input.1:3,640,640
#
# gRPC: perf_analyzer -m scrfd -u triton:8001 -i grpc \
#   --warmup-request-count=100 --request-count=100 \
#   --max-threads=3 --shape=input.1:3,640,640


import re
import subprocess
import csv
from datetime import datetime

MODEL_NAME = "scrfd"
HTTP_URL = "triton:8000"
GRPC_URL = "triton:8001"
WARMUP_REQUESTS = 10
REQUEST_COUNT = 100
MAX_THREADS = 3
BATCH_SIZES = [1, 4, 8]
NUM_ITERATIONS = 20
CSV_FILE = "scrfd_perf_results.csv"


def run_perf_analyzer(protocol, url, batch_size):
    """Run perf_analyzer and extract throughput and avg latency.

    Args:
        protocol: Protocol to use, either "http" or "grpc".
        url: URL of the Triton inference server.
        batch_size: Batch size for the inference requests.

    Returns:
        tuple: A tuple of (throughput, avg_latency_ms) where throughput is in infer/sec
            and avg_latency_ms is in milliseconds. Returns (None, None) on error.
    """
    shape = f"input.1:3,640,640"
    cmd = [
        "perf_analyzer",
        "-m", MODEL_NAME,
        "-u", url,
        "-i", protocol,
        "--warmup-request-count", str(WARMUP_REQUESTS),
        "--request-count", str(REQUEST_COUNT),
        "--max-threads", str(MAX_THREADS),
        "--shape", shape,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout

        # Extract throughput
        throughput_match = re.search(r"Throughput:\s+([\d.]+)\s+infer/sec", output)
        throughput = float(throughput_match.group(1)) if throughput_match else None

        # Extract avg latency (in usec)
        latency_match = re.search(r"Avg latency:\s+(\d+)\s+usec", output)
        avg_latency_usec = int(latency_match.group(1)) if latency_match else None
        avg_latency_ms = avg_latency_usec / 1000.0 if avg_latency_usec else None

        return throughput, avg_latency_ms

    except subprocess.CalledProcessError as e:
        print(f"Error running perf_analyzer ({protocol}): {e}")
        return None, None


def save_to_csv(protocol, batch_size, throughput, avg_latency_ms):
    """Save results to CSV file.

    Args:
        protocol: Protocol name (e.g., "HTTP" or "gRPC").
        batch_size: Batch size used for the inference.
        throughput: Throughput value in infer/sec.
        avg_latency_ms: Average latency in milliseconds.

    Returns:
        None.
    """
    file_exists = False
    try:
        with open(CSV_FILE, "r"):
            file_exists = True
    except FileNotFoundError:
        pass

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "protocol", "batch_size", "throughput_infer_per_sec", "avg_latency_ms"])

        writer.writerow([datetime.now().isoformat(), protocol, batch_size, throughput, avg_latency_ms])


def main():
    print(f"Running perf_analyzer for {MODEL_NAME} ({NUM_ITERATIONS} iterations per batch size)...")

    for batch_size in BATCH_SIZES:
        print(f"\n{'='*60}")
        print(f"Batch Size: {batch_size}")
        print(f"{'='*60}")

        # Run HTTP benchmark
        print(f"HTTP (Batch Size {batch_size}, {NUM_ITERATIONS} iterations)...")
        for i in range(NUM_ITERATIONS):
            print(f"  [{i+1}/{NUM_ITERATIONS}]", end=" ", flush=True)
            http_throughput, http_latency = run_perf_analyzer("http", HTTP_URL, batch_size)
            if http_throughput is not None and http_latency is not None:
                print(f"Throughput: {http_throughput:.2f} infer/sec, Latency: {http_latency:.2f} ms")
                save_to_csv("HTTP", batch_size, http_throughput, http_latency)
            else:
                print("Failed")

        # Run gRPC benchmark
        print(f"gRPC (Batch Size {batch_size}, {NUM_ITERATIONS} iterations)...")
        for i in range(NUM_ITERATIONS):
            print(f"  [{i+1}/{NUM_ITERATIONS}]", end=" ", flush=True)
            grpc_throughput, grpc_latency = run_perf_analyzer("grpc", GRPC_URL, batch_size)
            if grpc_throughput is not None and grpc_latency is not None:
                print(f"Throughput: {grpc_throughput:.2f} infer/sec, Latency: {grpc_latency:.2f} ms")
                save_to_csv("gRPC", batch_size, grpc_throughput, grpc_latency)
            else:
                print("Failed")

    print(f"\nResults saved to {CSV_FILE}")


if __name__ == "__main__":
    main()

