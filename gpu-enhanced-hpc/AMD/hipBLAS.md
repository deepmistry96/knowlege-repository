**## **hip[[BLAS]]: An Additional Portability Layer**

**hip[[BLAS]]** is another library in the ROCm ecosystem that provides a [[BLAS]] API similar to cu[[BLAS]] but is implemented on top of roc[[BLAS]]. It serves as a portability layer, making it easier to port applications from CUDA/cu[[BLAS]] to HIP/hip[[BLAS]].

### **Features of hip[[BLAS]]**

- **API Compatibility:**
    - Offers an API that closely mirrors cu[[BLAS]], simplifying code migration.
- **Backend Flexibility:**
    - Can run on both AMD and NVIDIA GPUs.
    - On AMD GPUs, hip[[BLAS]] calls are forwarded to roc[[BLAS]].
    - On NVIDIA GPUs, hip[[BLAS]] uses cu[[BLAS]] under the hood.

### **Example Usage with hip[[BLAS]]**

Using hip[[BLAS]] for matrix multiplication:

c**