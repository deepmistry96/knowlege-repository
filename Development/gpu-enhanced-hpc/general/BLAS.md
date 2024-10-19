**[[BLAS]]** stands for **Basic Linear Algebra Subprograms**. It is a specification that defines a set of low-level routines for performing common linear algebra operations such as vector and matrix multiplication. [[BLAS]] is widely used as a building block for higher-level mathematical libraries and is fundamental in the fields of scientific computing, engineering, and machine learning.

### **[[Levels of [[BLAS]]:**

[[BLAS]] is divided into three levels based on the complexity and type of operations:

1. **[[Level 1 [[BLAS]]:**
   - **Operations:** Vector-vector operations.
   - **Examples:** Dot product, vector addition, scaling a vector.
   - **Functions:** `axpy`, `dot`, `scal`, `copy`, `swap`, etc.

2. **[[Level 2 [[BLAS]]:**
   - **Operations:** Matrix-vector operations.
   - **Examples:** Matrix-vector multiplication.
   - **Functions:** `gemv`, `gbmv`, `hemv`, `trmv`, etc.

3. **[[Level 3 [[BLAS]]:**
   - **Operations:** Matrix-matrix operations.
   - **Examples:** Matrix-matrix multiplication.
   - **Functions:** `gemm`, `symm`, `trmm`, `syrk`, etc.

---

## **Other Core Utilities Besides [[BLAS]]**

Beyond [[BLAS]], there are several other core libraries and utilities that are essential for numerical computing and linear algebra tasks:

### **1. LAPACK (Linear Algebra PACKage):**

- **Purpose:** Provides routines for solving systems of linear equations, eigenvalue problems, and singular value decompositions.
- **Features:**
  - Builds upon [[BLAS]] routines to perform higher-level operations.
  - Supports various matrix factorizations: LU, QR, Cholesky, and SVD.
  - Essential for advanced linear algebra applications.

### **2. cu[[BLAS]] and roc[[BLAS]]:**

- **cu[[BLAS]]:**
  - **Vendor:** NVIDIA.
  - **Purpose:** GPU-accelerated version of [[BLAS]] for NVIDIA GPUs.
  - **Features:** Highly optimized for CUDA architecture, supports all three levels of [[BLAS]] operations.

- **roc[[BLAS]]:**
  - **Vendor:** AMD.
  - **Purpose:** GPU-accelerated [[BLAS]] library for AMD GPUs within the ROCm platform.
  - **Features:** Optimized for AMD hardware, provides similar functionality to cu[[BLAS]].

### **3. cuSOLVER and rocSOLVER:**

- **cuSOLVER:**
  - **Vendor:** NVIDIA.
  - **Purpose:** GPU-accelerated library providing LAPACK-like functionalities.
  - **Features:** Solving linear systems, eigenvalue problems, SVDs on NVIDIA GPUs.

- **rocSOLVER:**
  - **Vendor:** AMD.
  - **Purpose:** Provides LAPACK-equivalent functionalities optimized for AMD GPUs.
  - **Features:** Part of the ROCm ecosystem, works seamlessly with roc[[BLAS]].

### **4. MAGMA (Matrix Algebra on GPU and Multicore Architectures):**

- **Purpose:** Designed for heterogeneous computing environments combining GPUs and multicore CPUs.
- **Features:**
  - Provides LAPACK-equivalent routines.
  - Optimizes performance by exploiting both CPU and GPU resources.

### **5. FFTW (Fastest Fourier Transform in the West):**

- **Purpose:** Library for computing discrete Fourier transforms in one or more dimensions.
- **Features:**
  - Highly efficient and adaptable to various hardware architectures.
  - **GPU Variants:**
    - **cuFFT:** NVIDIA's GPU-accelerated FFT library.
    - **rocFFT:** AMD's GPU-accelerated FFT library within ROCm.

### **6. ScaLAPACK (Scalable LAPACK):**

- **Purpose:** Extension of LAPACK for distributed-memory parallel computers.
- **Features:**
  - Uses MPI for communication.
  - Designed for high-performance computing environments.

### **7. Open[[BLAS]]:**

- **Purpose:** An open-source implementation of [[BLAS]] and some LAPACK functionalities.
- **Features:**
  - Optimized for various CPU architectures.
  - Automatically detects and optimizes for the host machine.

### **8. Intel MKL (Math Kernel Library):**

- **Purpose:** Provides highly optimized math routines for Intel CPUs.
- **Features:**
  - Includes [[BLAS]], LAPACK, FFT, and random number generation.
  - Optimized for performance on Intel architectures.

### **9. Eigen:**

- **Purpose:** A C++ template library for linear algebra.
- **Features:**
  - Provides matrix and vector operations, numerical solvers.
  - Header-only library, easy to integrate.

### **10. cuDNN and MIOpen:**

- **cuDNN:**
  - **Vendor:** NVIDIA.
  - **Purpose:** GPU-accelerated library for deep neural networks.
  - **Features:** Provides optimized routines for deep learning primitives.

- **MIOpen:**
  - **Vendor:** AMD.
  - **Purpose:** Deep learning library for AMD GPUs.
  - **Features:** Supports popular deep learning frameworks, provides GPU acceleration.

### **11. BLIS ([[BLAS]]-like Library Instantiation Software):**

- **Purpose:** A framework for rapidly instantiating [[BLAS]]-like dense linear algebra libraries.
- **Features:**
  - Offers a modular and extensible approach.
  - Allows developers to create optimized [[BLAS]] libraries for specific architectures.

### **12. PETSc (Portable, Extensible Toolkit for Scientific Computation):**

- **Purpose:** Suite of data structures and routines for scalable (parallel) solution of scientific applications modeled by partial differential equations.
- **Features:**
  - Supports MPI, GPUs, and hybrid systems.
  - Includes nonlinear solvers, time integrators, and more.

### **13. ATLAS (Automatically Tuned Linear Algebra Software):**

- **Purpose:** Automatically generates optimized versions of [[BLAS]] for specific hardware.
- **Features:**
  - Self-optimizing library that tunes itself for the host hardware during installation.
  - Aims to achieve near-optimal performance.

---

## **Summary**

These core utilities and libraries form the backbone of numerical and scientific computing:

- **[[BLAS]]** provides foundational routines for linear algebra operations.
- **LAPACK** builds upon [[BLAS]] for more complex tasks like solving linear systems and eigenvalue problems.
- **GPU-Accelerated Libraries** (e.g., cu[[BLAS]], roc[[BLAS]]) leverage the parallelism of GPUs to accelerate computations.
- **Specialized Libraries** (e.g., cuDNN, MIOpen) target specific domains like deep learning.
- **High-Performance Computing Libraries** (e.g., ScaLAPACK, MAGMA) are designed for parallel and distributed systems.

**Understanding and utilizing these libraries can significantly enhance the performance and efficiency of applications in fields such as:**

- Scientific research.
- Engineering [[simulation]]s.
- Machine learning and artificial intelligence.
- Financial modeling.
- Data analytics.

These tools are essential for developers and researchers who require reliable and high-performance computational capabilities.