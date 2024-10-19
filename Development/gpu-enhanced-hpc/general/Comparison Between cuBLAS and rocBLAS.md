## **Comparison Between cu[[BLAS]] and roc[[BLAS]]**

| Aspect                        | **cu[[BLAS]]**                                             | **roc[[BLAS]] / hip[[BLAS]]**                           |
| ----------------------------- | ---------------------------------------------------------- | ------------------------------------------------------- |
| **Vendor**                    | NVIDIA                                                     | AMD                                                     |
| **Supported GPUs**            | NVIDIA GPUs                                                | AMD GPUs (hip[[BLAS]] can also run on NVIDIA GPUs)      |
| **Programming Model**         | CUDA                                                       | HIP (Heterogeneous-computing Interface for Portability) |
| **[[BLAS]] Levels Supported** | Levels 1, 2, and 3                                         | Levels 1, 2, and 3                                      |
| **Precision Support**         | FP32, FP64, FP16, Tensor Core support for mixed-precision  | FP32, FP64, FP16, BF16 (on supported hardware)          |
| **Asynchronous Execution**    | Yes (using CUDA streams)                                   | Yes (using HIP streams)                                 |
| **Batched Operations**        | Supported                                                  | Supported                                               |
| **Integration**               | Part of the CUDA toolkit; works with cuDNN, cuSolver, etc. | Part of ROCm; works with MIOpen, rocSOLVER, etc.        |
| **Open Source**               | Closed-source                                              | Open-source (available on GitHub)                       |
| **Deep Learning Support**     | Widely used in AI frameworks                               | Supported in ROCm versions of AI frameworks             |
| **Community and Support**     | Large user base; extensive documentation                   | Growing community; active development                   |

---