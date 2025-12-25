# Quantum-Accelerated HPC Engine

This project focuses on developing a high-performance computing (HPC) core that integrates low-level hardware optimizations (SIMD) with quantum-inspired algorithms to accelerate neural network training and inference.

---

## Technical Pillars

### 1. Multi-Architecture SIMD Abstraction Layer
The system provides a unified abstraction for vector instructions, allowing a single codebase to achieve optimal performance across diverse hardware platforms:
* **X86_64 Support:** Deep optimization via SSE, AVX2, and AVX-512 instruction sets for parallel floating-point operations.
* **AArch64 Support:** Leverages NEON Intrinsics to maximize throughput on modern ARM-based silicon (Apple Silicon, AWS Graviton).
* **Fast Approximation Kernels:** Custom implementations of transcendental functions (exp, log) using Union bit-manipulation, reducing latency compared to standard libraries while maintaining required precision.



### 2. Quantum-Inspired Optimization
The computing core integrates algorithms derived from quantum computing to solve convex and non-convex optimization problems:
* **Grover Search Implementation:** Utilizes Amplitude Amplification within the gradient search process, enabling the optimizer to identify optimal weight updates with higher probability than classical stochastic methods.
* **Quantum Fourier Transform (QFT):** Applies QFT to analyze parameter distribution characteristics in vector space, supporting stable model convergence and mitigating vanishing gradients.



### 3. High-Performance Systems Programming
* **Memory Arena Allocator:** Employs a centralized memory arena to manage RAM usage, minimizing OS overhead and preventing memory fragmentation.
* **Cache Alignment:** Strictly enforces 64-byte data alignment (Cache-line alignment) to optimize data loading from RAM into CPU registers.

---

## Installation and Building

### Requirements
* C++17 compliant compiler (GCC 9+, Clang 10+).
* OpenMP Runtime for multi-threaded orchestration.

### Build Instructions
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
