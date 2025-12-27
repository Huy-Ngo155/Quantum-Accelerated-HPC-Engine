# Quantum-Accelerated HPC Engine

A high-performance computing (HPC) core designed to bridge the gap between low-level hardware optimizations (SIMD) and quantum-inspired algorithms. This project serves as the primary implementation for my research on Variational Quantum Algorithms (VQAs).

---

## 📄 Research & Publications

This framework is the foundation for the following research paper:
* **Title:** A Transparent CPU-Based Framework for Studying Optimization Dynamics in Variational Quantum Algorithms
* **Author:** Huy Ngo
* **Read the Full Paper:** [Quantum_Accelerated_HPC_Engine_Paper.pdf](./Quantum_Accelerated_HPC_Engine_Paper.pdf)

---

## Technical Pillars

### 1. Multi-Architecture SIMD Abstraction Layer
The engine provides a unified abstraction for vector instructions, enabling high-throughput execution across diverse CPU architectures:
* **X86_64 Support:** Deep optimization via SSE4.2, AVX2, and AVX-512 for wide-vector floating-point arithmetic.
* **AArch64 Support:** Native NEON Intrinsics to maximize performance on ARM-based silicon.
* **Fast Transcendental Math:** Custom Union-based bit manipulation for exponential and logarithmic approximations, bypassing high-latency standard library calls.

### 2. Quantum-Inspired Optimization Engine
A custom optimizer designed to navigate non-convex landscapes by leveraging quantum mechanics principles:
* **Analytic Gradient Evaluation:** Full implementation of the **Parameter-Shift Rule** for exact gradient calculation, avoiding numerical noise from finite-difference methods.
* **Grover Search Integration:** Utilizes Amplitude Amplification to identify optimal weight updates with higher probability than classical stochastic descent.
* **Quantum Fourier Transform (QFT):** Applied for spectral analysis of parameter distributions to mitigate vanishing gradients (Barren Plateaus).

### 3. Systems-Level Memory Management
* **Memory Arena Allocator:** Implements a centralized arena to minimize heap fragmentation and eliminate allocation overhead during critical execution paths.
* **Cache-Line Alignment:** Strict 64-byte alignment enforcement to maximize data throughput between RAM and CPU registers.

---

## Framework Architecture

* **Quantum Core:** Exact state-vector simulation, parameterized quantum gates, and Born-rule measurements.
* **Execution Engine:** Multi-threaded orchestration via OpenMP with NUMA-aware data locality.
* **Diagnostic Suite:** Real-time monitoring of state evolution, gradient magnitudes, and training convergence.
  
## Implementation Highlights (Technical Documentation)

To ensure maximum performance and clarity, the following architectural choices were made:
- **Manual Vectorization:** Located in `engine-3.cpp`, we use AVX2/NEON intrinsics to perform 8-wide floating-point operations for state-vector updates.
- **Custom Memory Arena:** To avoid OS-level overhead, memory is pre-allocated in a contiguous block, ensuring cache-line alignment (64-byte).
- **Analytic Gradients:** Instead of finite difference, we implement the Parameter-shift rule for $O(1)$ numerical stability.
---

## Installation and Building

### Requirements
* C++17 compliant compiler (GCC 9+, Clang 10+).
* OpenMP Runtime.

### Build Instructions
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
