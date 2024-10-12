## **What is `[[nvcc]]`?**

### **Overview**

- **`[[nvcc]]`** is NVIDIA's CUDA compiler driver.
- **Purpose:** Compiles CUDA code into executable binaries for NVIDIA GPUs.
- **Functionality:** Separates device (GPU) and host (CPU) code, compiling each appropriately.

### **Key Features**

- **CUDA Programming Model:** Supports NVIDIA's proprietary parallel computing platform.
- **Optimization:** Provides optimizations specific to NVIDIA GPU architectures.
- **Integration:** Works seamlessly with NVIDIA's tools and libraries.

---

## **Working with `[[nvcc]]`**

### **Compiling CUDA Code**

- **Basic Command:**

  ```bash
  [[nvcc]] -o my_program my_program.cu
  ```

- **Specifying Target Architecture:**

  ```bash
  [[nvcc]] -arch=sm_75 -o my_program my_program.cu
  ```

### **CUDA Code Example**

```cpp
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Allocate and initialize host and device memory
    // Launch kernel
    // ...
}
```

---

