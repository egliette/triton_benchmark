import csv
import statistics

ARCFACE_CSV = "arcface_perf_results.csv"
SCRFD_CSV = "scrfd_perf_results.csv"


def read_and_summarize(csv_file, model_name):
    """Read CSV file and calculate average throughput and latency by protocol.

    Args:
        csv_file: Path to the CSV file containing performance results.
        model_name: Name of the model (e.g., "arcface" or "scrfd").

    Returns:
        dict: Dictionary containing model name and summary statistics by protocol.
            Returns None if the CSV file is not found.
    """
    results = {"HTTP": {"throughput": [], "latency": []}, "gRPC": {"throughput": [], "latency": []}}

    try:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                protocol = row["protocol"]
                if protocol in results:
                    throughput = float(row["throughput_infer_per_sec"])
                    latency = float(row["avg_latency_ms"])
                    results[protocol]["throughput"].append(throughput)
                    results[protocol]["latency"].append(latency)
    except FileNotFoundError:
        print(f"Warning: {csv_file} not found")
        return None

    # Calculate averages
    summary = {}
    for protocol in ["HTTP", "gRPC"]:
        if results[protocol]["throughput"]:
            summary[protocol] = {
                "avg_throughput": statistics.mean(results[protocol]["throughput"]),
                "avg_latency": statistics.mean(results[protocol]["latency"]),
                "count": len(results[protocol]["throughput"]),
            }
        else:
            summary[protocol] = None

    return {"model": model_name, "summary": summary}


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

    if arcface_data:
        print(f"\n{arcface_data['model'].upper()} Model:")
        print("-" * 80)
        print(f"{'Protocol':<10} {'Avg Throughput (infer/sec)':<30} {'Avg Latency (ms)':<20} {'Count':<10}")
        print("-" * 80)
        for protocol in ["HTTP", "gRPC"]:
            if arcface_data["summary"][protocol]:
                data = arcface_data["summary"][protocol]
                print(
                    f"{protocol:<10} {data['avg_throughput']:<30.2f} {data['avg_latency']:<20.2f} {data['count']:<10}"
                )
            else:
                print(f"{protocol:<10} {'N/A':<30} {'N/A':<20} {'0':<10}")

    if scrfd_data:
        print(f"\n{scrfd_data['model'].upper()} Model:")
        print("-" * 80)
        print(f"{'Protocol':<10} {'Avg Throughput (infer/sec)':<30} {'Avg Latency (ms)':<20} {'Count':<10}")
        print("-" * 80)
        for protocol in ["HTTP", "gRPC"]:
            if scrfd_data["summary"][protocol]:
                data = scrfd_data["summary"][protocol]
                print(
                    f"{protocol:<10} {data['avg_throughput']:<30.2f} {data['avg_latency']:<20.2f} {data['count']:<10}"
                )
            else:
                print(f"{protocol:<10} {'N/A':<30} {'N/A':<20} {'0':<10}")

    print("\n" + "=" * 80)


def main():
    arcface_data = read_and_summarize(ARCFACE_CSV, "arcface")
    scrfd_data = read_and_summarize(SCRFD_CSV, "scrfd")

    print_summary(arcface_data, scrfd_data)


if __name__ == "__main__":
    main()

