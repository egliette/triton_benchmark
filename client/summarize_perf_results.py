import csv
import statistics

ARCFACE_CSV = "arcface_perf_results.csv"
SCRFD_CSV = "scrfd_perf_results.csv"


def read_and_summarize(csv_file, model_name):
    """Read CSV file and calculate average throughput and latency by protocol and batch size.

    Args:
        csv_file: Path to the CSV file containing performance results.
        model_name: Name of the model (e.g., "arcface" or "scrfd").

    Returns:
        dict: Dictionary containing model name and summary statistics by protocol and batch size.
            Returns None if the CSV file is not found.
    """
    results = {}

    try:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                protocol = row["protocol"]
                batch_size = int(row.get("batch_size", 1))  # Default to 1 for backward compatibility
                
                key = (protocol, batch_size)
                if key not in results:
                    results[key] = {"throughput": [], "latency": []}
                
                throughput = float(row["throughput_infer_per_sec"])
                latency = float(row["avg_latency_ms"])
                results[key]["throughput"].append(throughput)
                results[key]["latency"].append(latency)
    except FileNotFoundError:
        print(f"Warning: {csv_file} not found")
        return None

    # Calculate averages grouped by batch size
    summary_by_batch = {}
    for (protocol, batch_size), data in results.items():
        if batch_size not in summary_by_batch:
            summary_by_batch[batch_size] = {}
        
        if data["throughput"]:
            summary_by_batch[batch_size][protocol] = {
                "avg_throughput": statistics.mean(data["throughput"]),
                "avg_latency": statistics.mean(data["latency"]),
                "count": len(data["throughput"]),
            }
        else:
            summary_by_batch[batch_size][protocol] = None

    return {"model": model_name, "summary_by_batch": summary_by_batch}


def print_summary(arcface_data, scrfd_data):
    """Print formatted summary table.

    Args:
        arcface_data: Dictionary containing ArcFace model performance data.
        scrfd_data: Dictionary containing SCRFD model performance data.

    Returns:
        None.
    """
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    def print_model_summary(model_data):
        """Print summary for a single model grouped by batch size."""
        if not model_data:
            return
        
        print(f"\n{model_data['model'].upper()} Model:")
        summary_by_batch = model_data.get("summary_by_batch", {})
        
        # Get all batch sizes and sort them
        batch_sizes = sorted(summary_by_batch.keys())
        
        for batch_size in batch_sizes:
            print(f"\n  Batch Size: {batch_size}")
            print("-" * 80)
            print(f"{'Protocol':<10} {'Avg Throughput (infer/sec)':<30} {'Avg Latency (ms)':<20} {'Count':<10}")
            print("-" * 80)
            
            batch_summary = summary_by_batch[batch_size]
            for protocol in ["HTTP", "gRPC"]:
                if protocol in batch_summary and batch_summary[protocol]:
                    data = batch_summary[protocol]
                    print(
                        f"{protocol:<10} {data['avg_throughput']:<30.2f} {data['avg_latency']:<20.2f} {data['count']:<10}"
                    )
                else:
                    print(f"{protocol:<10} {'N/A':<30} {'N/A':<20} {'0':<10}")

    print_model_summary(arcface_data)
    print_model_summary(scrfd_data)

    print("\n" + "=" * 80)


def main():
    arcface_data = read_and_summarize(ARCFACE_CSV, "arcface")
    scrfd_data = read_and_summarize(SCRFD_CSV, "scrfd")

    print_summary(arcface_data, scrfd_data)


if __name__ == "__main__":
    main()

