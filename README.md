# Triton Inference Server Benchmark

This project benchmarks Triton Inference Server performance using HTTP and gRPC protocols for face detection (SCRFD) and face recognition (ArcFace) models.

## Environment Profile

### System Information
- **OS**: Windows 11
- **WSL Version**: 2.6.3.0 [Run: `wsl --version`]
- **Ubuntu Version**: Ubuntu 24.04.3 LTS [Run: `lsb_release -a`]
- **Docker Version**: 28.4.0 [Run: `docker --version`]

### Hardware Specifications
- **CPU**: 12th Gen Intel(R) Core(TM) i5-12450H [Run: `lscpu | grep "Model name"`]
- **CPU Cores**: 12 [Run: `nproc`]
- **RAM**: 7.6 GB [Run: `free -h`]
- **GPU**: NVIDIA GeForce RTX 4050 [Run: `nvidia-smi`]
- **GPU Memory**: 6141MiB [Run: `nvidia-smi`]
- **CUDA Version**: 12.7 [Run: `nvidia-smi`]

### Docker Environment
- **Triton Server Image**: `nvcr.io/nvidia/tritonserver:25.06-py3`

### Model Configurations
- **SCRFD Model**:
  - Input: `input.1` (FP32, shape: [batch, 3, 640, 640])
  - Output: `score_8` (FP32)
  - Max Batch Size: 8
  - Instance Count: 2 (GPU)
  - Dynamic Batching: Enabled (max_queue_delay: 100μs)

- **ArcFace Model**:
  - Input: `input.1` (FP32, shape: [batch, 3, 112, 112])
  - Output: `683` (FP32, shape: [batch, 512])
  - Max Batch Size: 8
  - Instance Count: 2 (GPU)
  - Dynamic Batching: Enabled (max_queue_delay: 100μs)

### Benchmark Parameters
- **Batch Sizes**: [1, 4, 8]
- **Iterations**: 1000
- **Warmup Requests**: 10
- **Protocols**: HTTP (port 8000), gRPC (port 8001)

## Benchmark Results

### SCRFD Model

| Batch Size | Protocol | Mean Latency (ms) | P95 Latency (ms) | Concurrent (3 processes) Mean (ms) | Concurrent (3 processes) P95 (ms) |
|------------|----------|-------------------|------------------|-----------------------------------|----------------------------------|
| 1          | HTTP     | **17.96**         | **23.08**        | 40.07                             | **133.87**                       |
| 1          | gRPC     | 20.02             | 26.53            | **38.47**                         | 153.18                           |
| 4          | HTTP     | **59.37**         | **78.57**        | **95.35**                         | **137.40**                       |
| 4          | gRPC     | 82.76             | 116.45           | 104.82                            | 145.25                           |
| 8          | HTTP     | **137.15**        | **186.09**       | **184.51**                        | **257.08**                       |
| 8          | gRPC     | 173.831           | 249.22           | 241.33                            | 350.88                           |

### ArcFace Model

| Batch Size | Protocol | Mean Latency (ms) | P95 Latency (ms) | Concurrent (3 processes) Mean (ms) | Concurrent (3 processes) P95 (ms) |
|------------|----------|-------------------|------------------|-----------------------------------|----------------------------------|
| 1          | HTTP     | 8.80              | 17.13            | **16.60**                         | **57.98**                        |
| 1          | gRPC     | **7.92**          | **12.36**        | 17.50                             | 65.30                            |
| 4          | HTTP     | **13.11**         | **17.10**        | **30.85**                         | **59.63**                        |
| 4          | gRPC     | 27.20             | 45.70            | 34.32                             | 130.80                           |
| 8          | HTTP     | 27.91             | 46.09            | **56.93**                         | **75.78**                        |
| 8          | gRPC     | **24.11**         | **32.85**        | 57.96                             | 77.85                            |

## Running Benchmarks

### Start Triton Server
```bash
docker network create triton
docker-compose up -d triton
```

### Run SCRFD Benchmark
```bash
docker-compose run --rm client python benchmark_scrfd.py
```

### Run SCRFD Concurrent Benchmark (3 processes)
```bash
docker-compose run --rm client python benchmark_scrfd_concurrency.py
```

### Run ArcFace Benchmark
```bash
docker-compose run --rm client python benchmark_arcface.py
```

### Run ArcFace Concurrent Benchmark (3 processes)
```bash
docker-compose run --rm client python benchmark_arcface_concurrency.py
```

## Todo List

- [ ] Add model download URL
- [ ] Test benchmark with remote client
