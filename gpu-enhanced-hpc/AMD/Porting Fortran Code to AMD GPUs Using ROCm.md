
**Porting Fortran Code to [[AMD]] GPUs Using [[ROCm]]**

To leverage your [[AMD]] GPU for running the Fire Dynamics Simulator (FDS) written in Fortran, you'll need to port the code to utilize GPU acceleration. Below is a comprehensive guide to help you through the process, including modifying the code, using profiling tools like `rocprof` and `perfetto`, and understanding GPU memory management, [[kernel]] writing, and launching.

---

### **1. Understand the Existing Codebase**

Before making any modifications, familiarize yourself with FDS's code structure:

- **Modular Breakdown**: Identify computationally intensive modules or functions that would benefit most from GPU acceleration.
- **Data Structures**: Understand how data is managed and accessed, as this impacts memory transfer between CPU and GPU.

### **2. Set Up the Development Environment**

To develop and run GPU-accelerated Fortran code on [[AMD]] hardware, you'll need:

- **[[AMD]] [[ROCm]] Platform**: [[ROCm]] is [[AMD]]'s open software platform for GPU computing.
  - **Installation**: Follow the official [[ROCm]] installation guide for your operating system: [[ROCm]] Installation Guide](https://[[ROCm]]docs.[[AMD]].com/en/latest/Installation_Guide/Installation-Guide.html).
- **Fortran Compiler with GPU Support**:
  - **[[AMD]] Optimizing CPU Libraries (AOCL) Fortran Compiler**: Supports [[OpenMP]] offloading to [[AMD]] GPUs.
  - **LLVM Flang**: An open-source Fortran front-end for LLVM, but GPU offloading support may be limited.
  - **[[NVIDIA]]'s NVHPC Compiler**: Supports [[OpenACC]] and [[OpenMP]] offloading, but primarily optimized for [[NVIDIA]] GPUs.
  - **Alternative**: Use the open-source [[gfort]]ran]] compiler with [[OpenMP]] 4.5+ support, though GPU offloading capabilities may vary.

### **3. Choose a Parallelization Model**

Two primary models for GPU acceleration in Fortran are **[[OpenMP]]** and **[[OpenACC]]**.

#### **[[OpenMP]] Offloading**

- **Pros**:
  - Widely supported and integrated into many compilers.
  - Familiar syntax if the code already uses [[OpenMP]] for CPU parallelization.
- **Cons**:
  - GPU offloading features are compiler-dependent.
- **Implementation Steps**:
  - Add `!$omp target` directives to offload code regions to the GPU.
  - Use `!$omp teams` and `!$omp distribute` to manage GPU threads.
- **Example**:
  ```fortran
  !$omp target teams distribute parallel do
  do i = 1, N
    array(i) = compute_value(i)
  end do
  !$omp end target teams distribute parallel do
  ```

#### **[[OpenACC]]**

- **Pros**:
  - Designed specifically for accelerators and GPUs.
  - Simplifies data management between host and device.
- **Cons**:
  - Requires a compiler with [[OpenACC]] support and [[AMD]] GPU backend.
- **Implementation Steps**:
  - Add `!$acc` directives to parallel regions.
  - Manage data movement with `!$acc data` directives.
- **Example**:
  ```fortran
  !$acc [[kernel]]s
  do i = 1, N
    array(i) = compute_value(i)
  end do
  !$acc end [[kernel]]s
  ```

### **4. Modify the Code for GPU Acceleration**

#### **Identify Hotspots**

- Use profiling tools on the CPU-only version to find the most time-consuming functions.
- Focus on loops and computations that can be parallelized.

#### **Add Parallel Directives**

- Insert [[OpenMP]] or [[OpenACC]] directives around computationally intensive loops.
- Ensure that data dependencies are properly managed to avoid race conditions.

#### **Manage Data Movement**

- Minimize data transfer between CPU and GPU by:
  - Allocating data on the GPU memory.
  - Keeping data on the GPU for as long as possible.
- Use `!$omp target data` or `!$acc data` directives to control data scope.

#### **Example Modification**

Original Loop:
```fortran
do i = 1, N
  result(i) = compute_heavy_operation(i)
end do
```

Modified Loop with [[OpenMP]] Offloading:
```fortran
!$omp target teams distribute parallel do
do i = 1, N
  result(i) = compute_heavy_operation(i)
end do
!$omp end target teams distribute parallel do
```

### **5. Compile the Code**

#### **Using AOCC or Compatible Compiler**

- **Compilation Command**:
  ```bash
  flang -f[[OpenMP]] -f[[OpenMP]]-targets=[[AMD]]gcn-[[AMD]]-[[AMD]]hsa -O3 -o fds_gpu fds.f90
  ```
- **Flags Explanation**:
  - `-f[[OpenMP]]`: Enables [[OpenMP]] support.
  - `-f[[OpenMP]]-targets=[[AMD]]gcn-[[AMD]]-[[AMD]]hsa`: Specifies the target GPU architecture.
  - `-O3`: Optimization level 3 for performance.

#### **Potential Issues**

- **Compiler Errors**: If the compiler doesn't support certain [[OpenMP]] features, you may need to update or switch compilers.
- **Linking Libraries**: Ensure that [[ROCm]] libraries are linked correctly.

### **6. Run and Test the Application**

- **Set Environment Variables**:
  - `OMP_TARGET_OFFLOAD=MANDATORY`: Ensures that offloading is required.
  - `ACC_DEVICE_TYPE=gpu`: For [[OpenACC]], specify the device type.
- **Execute the Program**:
  ```bash
  ./fds_gpu
  ```
- **Validate Results**:
  - Compare outputs between CPU-only and GPU-accelerated versions to ensure correctness.

### **7. Profile the Application**

#### **Using `rocprof`**

- **Purpose**: Collect performance data from the GPU.
- **Command**:
  ```bash
  rocprof --stats ./fds_gpu
  ```
- **Output**: Provides [[kernel]] execution times, memory transfer stats, and more.

#### **Using `perfetto`**

- **Purpose**: Visualize trace profiles for in-depth analysis.
- **Steps**:
  - **Collect Trace Data**:
    ```bash
    rocprof --hip-trace --hsa-trace --output-file trace.out ./fds_gpu
    ```
  - **Convert to Perfetto Format**:
    ```bash
    rocprof --hsa-trace --hip-trace --rocprofiler-trace --output-format perfetto ./fds_gpu
    ```
  - **Visualize**:
    - Open the trace file in the Perfetto UI: [Perfetto UI](https://ui.perfetto.dev/).

#### **Interpret Results**

- **Identify Bottlenecks**: Look for [[kernel]]s with high execution time.
- **Optimize**: Modify code to improve performance, such as optimizing memory access patterns or increasing parallelism.

### **8. Advanced Optimization**

#### **[[kernel]] Optimization**

- **Memory Coalescing**: Ensure that memory accesses are sequential to improve bandwidth utilization.
- **Compute-to-Memory Ratio**: Aim for computations that do more work per memory access.

#### **Concurrency**

- **Asynchronous Execution**: Overlap data transfers with computations using asynchronous data directives.
- **Stream Management**: Use multiple streams or queues to manage concurrent [[kernel]] execution.

### **9. Resources and Tutorials**

- **[[AMD]] [[ROCm]] Documentation**:
  - [[ROCm]] Documentation](https://[[ROCm]]docs.[[AMD]].com/)
- **[[OpenMP]] Offloading Guide**:
  - [[OpenMP]] Target Offloading](https://www.[[OpenMP]].org/spec-html/5.0/[[OpenMP]]su50.html#x151-3170002.8)
- **Tutorials**:
  - [Porting Fortran Applications to [[AMD]] GPUs](https://www.olcf.ornl.gov/wp-content/uploads/2020/12/Fortran-GPU-Offloading-Tutorial.pdf)
  - [Reddit Tutorial Post](https://www.reddit.com/r/[[ROCm]]/comments/12dvqtk/tutorial_porting_a_simple_fortran_application_to/)

### **10. Potential Challenges**

- **Compiler Limitations**: Not all compilers fully support [[OpenMP]] offloading to [[AMD]] GPUs.
- **Code Complexity**: Large codebases may have dependencies or programming patterns that complicate GPU acceleration.
- **Performance Tuning**: Achieving optimal performance may require iterative profiling and optimization.

---

### **Conclusion**

By following these steps, you can modify the FDS code to run on your [[AMD]] GPU, potentially achieving significant performance improvements over CPU-only execution. Profiling tools like `rocprof` and `perfetto` will aid in identifying bottlenecks and optimizing your application.

**Next Steps**:

1. **Set Up Your Environment**: Install [[ROCm]] and choose a compatible Fortran compiler.
2. **Modify and Compile**: Begin adding [[OpenMP]] or [[OpenACC]] directives to your code.
3. **Test and Validate**: Ensure that your GPU-accelerated code produces correct results.
4. **Profile and Optimize**: Use profiling tools to identify performance issues and optimize accordingly.
5. **Iterate**: Optimization is an iterative process; continue refining your code for better performance.

---

If you encounter specific issues or have questions during the process, consider reaching out to the [[AMD]] developer community or forums for assistance.