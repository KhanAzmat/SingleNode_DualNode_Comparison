# 4GPU vs 8GPU Distributed Inference Performance Comparison

This repository contains a comprehensive performance comparison between single-node 4GPU and dual-node 8GPU distributed inference setups using vLLM and DeepSeek R1 model.

## Overview

The project compares the performance characteristics of:
- **Single Node Setup**: 4 GPUs on a single machine
- **Dual Node Setup**: 8 GPUs distributed across 2 machines (4 GPUs each)

Both setups run the DeepSeek R1 model using vLLM for efficient inference and are benchmarked using a custom benchmarking tool.

## Project Structure

```
4GPU_8GPU_comparison/
├── README.md                           # This file
├── benchmarker.py                      # Custom benchmarking tool
├── vllm_distibuted_head.sh            # Head node startup script
├── vllm_distibuted_worker.sh          # Worker node startup script
├── Single_Node_Results/               # 4GPU single node results
│   ├── deepseek_benchmark_20250814_180325.json
│   ├── deepseek_benchmark_20250814_180325.csv
│   └── deepseek_performance_20250814_180325.png
└── Dual_Node_Results/                 # 8GPU distributed node results
    ├── benchmark_20250815_113424_dual.json
    ├── benchmark_20250815_113424_dual.csv
    └── performance_20250815_113424_dual.png
```

## Key Components

### 1. Benchmarking Tool (`benchmarker.py`)

A comprehensive benchmarking tool that tests LLM inference performance across different load scenarios:

- **Light Load**: 50 concurrent requests
- **Medium Load**: 100 concurrent requests  
- **Heavy Load**: 200 concurrent requests
- **Burst Load**: 300 concurrent requests
- **Stress Test**: 500 concurrent requests

The tool measures:
- Latency (avg, median, p95, p99)
- Time to First Token (TTFT)
- Time Per Output Token (TPOT)
- Requests per second
- Tokens per second
- Success rates
- Token generation statistics

### 2. Distributed Setup Scripts

- **`vllm_distibuted_head.sh`**: Configures and starts the head node (10.200.0.1)
- **`vllm_distibuted_worker.sh`**: Configures and starts the worker node (10.200.0.2)

Both scripts include optimized NCCL and Ray configurations for distributed training.

## Performance Results Summary

### Single Node (4 GPU) Results
- **Model**: DeepSeek R1
- **Date**: August 14, 2025
- **Peak Performance**: ~1.59 requests/second, ~1,213 tokens/second
- **Success Rate**: 100% across all load scenarios

### Dual Node (8 GPU) Results  
- **Model**: DeepSeek R1
- **Date**: August 15, 2025
- **Peak Performance**: ~1.32 requests/second, ~1,021 tokens/second
- **Success Rate**: 100% across all load scenarios

## Key Findings

1. **Scalability**: The 8GPU distributed setup shows different performance characteristics compared to the 4GPU single-node setup
2. **Latency**: Distributed setup shows higher latency due to network overhead
3. **Throughput**: Performance varies by load scenario, with different optimal configurations
4. **Reliability**: Both setups achieve 100% success rates across all test scenarios

## Usage

### Prerequisites
- Python 3.8+
- vLLM
- Ray
- DeepSeek R1 model
- NCCL for distributed communication
- Both the environments, head and worker, have to be same

### Running the Benchmark

1. **Single Node Setup**:
   ```bash
   # Start vLLM server on single node with 4 GPUs
   vllm serve \
     --model deepseek-ai/deepseek-coder \
     --tensor-parallel-size 4
   
   # Run benchmark
   python benchmarker.py --url http://localhost:8003
   ```

2. **Distributed Setup**:
   ```bash
   # On head node (10.200.0.1)
   ./vllm_distibuted_head.sh
   
   # On worker node (10.200.0.2)  
   ./vllm_distibuted_worker.sh
   
   # Start vLLM with distributed setup
   vllm serve \
     --model deepseek-ai/deepseek-coder-1.3b-instruct \
     --tensor-parallel-size 4 \
     --pipeline-parallel-size 2
   
   # Run benchmark
   python benchmarker.py --url http://10.200.0.1:8003
   ```

## Results Analysis

The benchmark results are stored in JSON, CSV, and PNG formats for detailed analysis:

- **JSON**: Complete benchmark data with all metrics
- **CSV**: Tabular format for spreadsheet analysis
- **PNG**: Performance visualization charts

## Network Configuration

The distributed setup uses:
- **Head Node IP**: 10.200.0.1
- **Worker Node IP**: 10.200.0.2
- **Ray Port**: 6379
- **Network Interface**: eno1
- **NCCL Configuration**: Optimized for socket-based communication

## Contributing

This project demonstrates the trade-offs between single-node and distributed inference setups. The results can be used to:

- Optimize deployment strategies
- Understand scaling characteristics
- Plan infrastructure requirements
- Compare different hardware configurations
