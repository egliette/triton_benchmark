import os
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from queue import Queue
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient


@dataclass
class BenchmarkResult:
    protocol: str
    batch_size: int
    avg_latency_ms: float
    test_type: str = "sequential"  # sequential or concurrent
    latencies: List[float] = (
        None  # Store all latency measurements for distribution plots
    )

    def __post_init__(self):
        if self.latencies is None:
            self.latencies = []

    def to_dict(self):
        """Convert to dictionary for pandas DataFrame (exclude latencies list)"""
        d = asdict(self)
        # Don't include latencies in DataFrame (too large)
        d.pop("latencies", None)
        return d


class TritonBenchmark:
    def __init__(
        self, model_name="arcface", grpc_url="triton:8001", http_url="triton:8000"
    ):
        self.model_name = model_name
        self.grpc_url = grpc_url
        self.http_url = http_url

    def create_test_data(self, batch_size=1):
        """Create test image data"""
        img = np.zeros((112, 112, 3), dtype=np.uint8)
        img_float32 = img.astype(np.float32)
        img_single = img_float32.transpose(2, 0, 1)[np.newaxis, ...]

        # Repeat for batch
        if batch_size > 1:
            img_batch = np.repeat(img_single, batch_size, axis=0)
        else:
            img_batch = img_single

        return img_batch

    def _infer(self, client, use_grpc, img_batch):
        """Helper method to perform inference (reduces code duplication)"""
        if use_grpc:
            inputs = [grpcclient.InferInput("input.1", img_batch.shape, "FP32")]
            outputs = [grpcclient.InferRequestedOutput("683")]
        else:
            inputs = [httpclient.InferInput("input.1", img_batch.shape, "FP32")]
            outputs = [httpclient.InferRequestedOutput("683")]

        inputs[0].set_data_from_numpy(img_batch)
        response = client.infer(self.model_name, inputs, outputs=outputs)
        return response.as_numpy("683")

    def warmup(self, client, use_grpc, batch_size=1, num_warmup=50, verbose=False):
        """Warmup the model and client connection"""
        if verbose:
            print(f"  Warming up with {num_warmup} requests...")
        img_batch = self.create_test_data(batch_size)

        for _ in range(num_warmup):
            try:
                self._infer(client, use_grpc, img_batch)
            except Exception as e:
                if verbose:
                    print(f"  Warmup error: {e}")

    def run_sequential_benchmark(
        self, use_grpc=False, batch_size=1, num_requests=1000, verbose=False
    ):
        """Run sequential (synchronous) benchmark"""
        protocol = "gRPC" if use_grpc else "HTTP"
        if verbose:
            print(
                f"  Sequential {protocol} | Batch: {batch_size} | Requests: {num_requests}"
            )

        # Create client
        if use_grpc:
            client = grpcclient.InferenceServerClient(url=self.grpc_url)
        else:
            client = httpclient.InferenceServerClient(url=self.http_url)

        # Warmup
        self.warmup(client, use_grpc, batch_size, verbose=verbose)

        # Prepare test data
        img_batch = self.create_test_data(batch_size)
        latencies = []

        for i in range(num_requests):
            req_start = time.time()

            try:
                self._infer(client, use_grpc, img_batch)
                req_end = time.time()
                latencies.append((req_end - req_start) * 1000)  # Convert to ms

            except Exception as e:
                if verbose:
                    print(f"  Request {i} failed: {e}")

        result = BenchmarkResult(
            protocol=protocol,
            batch_size=batch_size,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            test_type="sequential",
            latencies=latencies,  # Store all latencies for distribution plots
        )

        # Close client
        if hasattr(client, "close"):
            client.close()

        return result

    def worker_thread(
        self, use_grpc, img_batch, num_requests, latencies_queue, errors_queue
    ):
        """Worker thread for concurrent requests - each thread creates its own client"""
        # Create a client for this thread (clients are not thread-safe)
        try:
            if use_grpc:
                client = grpcclient.InferenceServerClient(url=self.grpc_url)
            else:
                client = httpclient.InferenceServerClient(url=self.http_url)
        except Exception as e:
            errors_queue.put(f"Client creation failed: {str(e)}")
            return

        for _ in range(num_requests):
            req_start = time.time()

            try:
                self._infer(client, use_grpc, img_batch)
                req_end = time.time()
                latencies_queue.put((req_end - req_start) * 1000)

            except Exception as e:
                errors_queue.put(str(e))

        # Close the client after all requests
        try:
            if hasattr(client, "close"):
                client.close()
        except:
            pass

    def run_concurrent_benchmark(
        self,
        use_grpc=False,
        batch_size=1,
        num_requests=1000,
        num_threads=10,
        verbose=False,
    ):
        """Run concurrent (multi-threaded) benchmark"""
        protocol = "gRPC" if use_grpc else "HTTP"
        if verbose:
            print(
                f"  Concurrent {protocol} | Batch: {batch_size} | Threads: {num_threads} | Requests: {num_requests}"
            )

        # Create client for warmup only
        if use_grpc:
            warmup_client = grpcclient.InferenceServerClient(url=self.grpc_url)
        else:
            warmup_client = httpclient.InferenceServerClient(url=self.http_url)

        # Warmup
        self.warmup(warmup_client, use_grpc, batch_size, num_warmup=20, verbose=verbose)

        # Close warmup client
        if hasattr(warmup_client, "close"):
            warmup_client.close()

        # Prepare test data
        img_batch = self.create_test_data(batch_size)

        # Queues for results
        latencies_queue = Queue()
        errors_queue = Queue()

        # Calculate requests per thread
        requests_per_thread = num_requests // num_threads

        # Create and start threads - each creates its own client
        threads = []
        for i in range(num_threads):
            t = threading.Thread(
                target=self.worker_thread,
                args=(
                    use_grpc,
                    img_batch,
                    requests_per_thread,
                    latencies_queue,
                    errors_queue,
                ),
            )
            t.start()
            threads.append(t)

        # Wait for all threads
        for t in threads:
            t.join()

        # Collect results
        latencies = []
        while not latencies_queue.empty():
            latencies.append(latencies_queue.get())

        errors = []
        while not errors_queue.empty():
            errors.append(errors_queue.get())

        if errors and verbose:
            print(f"  ⚠ Errors: {len(errors)} requests failed")

        # Calculate metrics
        len(latencies)

        result = BenchmarkResult(
            protocol=protocol,
            batch_size=batch_size,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            test_type="concurrent",
            latencies=latencies,  # Store all latencies for distribution plots
        )

        return result

    @staticmethod
    def create_visualizations(results: List[BenchmarkResult], output_dir: str):
        """Create and save visualization graphs including distribution plots"""
        df = pd.DataFrame([r.to_dict() for r in results])

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # 1. Average Latency comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle(
            "Triton Benchmark Results - Latency Analysis",
            fontsize=16,
            fontweight="bold",
        )

        for test_type in df["test_type"].unique():
            for protocol in df["protocol"].unique():
                subset = df[
                    (df["test_type"] == test_type) & (df["protocol"] == protocol)
                ]
                if not subset.empty:
                    ax.plot(
                        subset["batch_size"],
                        subset["avg_latency_ms"],
                        marker="s",
                        label=f"{protocol} ({test_type})",
                    )
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Average Latency (ms)")
        ax.set_title("Average Latency Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        graph_path = os.path.join(output_dir, "arcface_benchmark_results.png")
        plt.savefig(graph_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Visualization saved to: {graph_path}")

        # 2. Latency Distribution Plots - Merge HTTP and gRPC for comparison
        # Create plots for each batch size (1, 4, 8)
        colors = {"HTTP": "blue", "gRPC": "red"}
        batch_sizes = [1, 4, 8]

        for batch_size in batch_sizes:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(
                f"Latency Distribution Analysis (Batch Size {batch_size})",
                fontsize=16,
                fontweight="bold",
            )

            for test_type_idx, test_type in enumerate(["sequential", "concurrent"]):
                ax = axes[test_type_idx]

                for protocol in ["HTTP", "gRPC"]:
                    # Find matching result
                    result = next(
                        (
                            r
                            for r in results
                            if r.protocol == protocol
                            and r.test_type == test_type
                            and r.batch_size == batch_size
                        ),
                        None,
                    )

                    if result and result.latencies and len(result.latencies) > 0:
                        latencies = np.array(result.latencies)

                        # Histogram
                        ax.hist(
                            latencies,
                            bins=50,
                            alpha=0.5,
                            density=True,
                            label=f"{protocol} (Avg: {result.avg_latency_ms:.2f}ms)",
                            color=colors[protocol],
                        )

                        # Add vertical line for average
                        ax.axvline(
                            result.avg_latency_ms,
                            color=colors[protocol],
                            linestyle="--",
                            linewidth=2,
                            alpha=0.8,
                        )

                ax.set_xlabel("Latency (ms)")
                ax.set_ylabel("Density")
                ax.set_title(f"{test_type.capitalize()} Requests")
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            dist_path = os.path.join(
                output_dir, f"arcface_latency_distributions_batch{batch_size}.png"
            )
            plt.savefig(dist_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(
                f"  Distribution plots (batch size {batch_size}) saved to: {dist_path}"
            )


def run_comprehensive_benchmark():
    """Run comprehensive benchmark suite"""
    benchmark = TritonBenchmark(model_name="arcface")

    print("=" * 60)
    print("TRITON BENCHMARK: HTTP vs gRPC")
    print("=" * 60)

    # Test configurations
    batch_sizes = [1, 4, 8]
    num_requests = 500
    num_threads = 10

    # Collect all results
    all_results = []

    # Sequential benchmarks
    print("\n[Sequential Benchmarks]")
    for batch_size in batch_sizes:
        print(f"  Batch Size: {batch_size}")
        # HTTP
        http_result = benchmark.run_sequential_benchmark(
            use_grpc=False,
            batch_size=batch_size,
            num_requests=num_requests,
            verbose=False,
        )
        all_results.append(http_result)

        # gRPC
        grpc_result = benchmark.run_sequential_benchmark(
            use_grpc=True,
            batch_size=batch_size,
            num_requests=num_requests,
            verbose=False,
        )
        all_results.append(grpc_result)

    # Concurrent benchmarks
    print("\n[Concurrent Benchmarks]")
    for batch_size in batch_sizes:
        print(f"  Batch Size: {batch_size}")
        # HTTP
        http_result = benchmark.run_concurrent_benchmark(
            use_grpc=False,
            batch_size=batch_size,
            num_requests=num_requests,
            num_threads=num_threads,
            verbose=False,
        )
        all_results.append(http_result)

        # gRPC
        grpc_result = benchmark.run_concurrent_benchmark(
            use_grpc=True,
            batch_size=batch_size,
            num_requests=num_requests,
            num_threads=num_threads,
            verbose=False,
        )
        all_results.append(grpc_result)

    # Create visualizations
    output_dir = "benchmark_results"
    benchmark.create_visualizations(all_results, output_dir)

    print(f"\nVisualizations saved to: {os.path.abspath(output_dir)}/")


if __name__ == "__main__":
    run_comprehensive_benchmark()
