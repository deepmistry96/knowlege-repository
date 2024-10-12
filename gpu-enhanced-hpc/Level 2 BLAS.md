## **Level 2 [[BLAS]]: Matrix-Vector Operations**

### **Overview**

- **Operations**: Operations involving a matrix and a vector.
- **Data Structures**: Operate on two-dimensional arrays (matrices) and vectors.
- **Computational Complexity**: O(n²), where n is the dimension of the matrix.

### **Common Functions**

1. **General Matrix-Vector Multiplication**
    
    - **Function**: `gemv`
    - **Operation**: Computes y←αAx+βyy \leftarrow \alpha A x + \beta yy←αAx+βy
    - **Usage**: Applying linear transformations.
2. **Symmetric/Hermitian Matrix-Vector Multiplication**
    
    - **Function**: `symv`, `hemv`
    - **Operation**: Specialized for symmetric (real) or Hermitian (complex) matrices.
    - **Usage**: Efficient computations exploiting matrix properties.
3. **Triangular Matrix-Vector Multiplication**
    
    - **Function**: `trmv`
    - **Operation**: Computes x←Axx \leftarrow A xx←Ax where A is triangular.
    - **Usage**: Solving triangular systems.
4. **Solving Triangular Systems**
    
    - **Function**: `trsv`
    - **Operation**: Solves Ax=bA x = bAx=b where A is triangular.
    - **Usage**: Backward and forward substitution in solving linear systems.
5. **Rank-1 Update**
    
    - **Function**: `ger`
    - **Operation**: Computes A←A+αxyTA \leftarrow A + \alpha x y^TA←A+αxyT
    - **Usage**: Updating matrices with outer products.
6. **Symmetric/Hermitian Rank-1 and Rank-2 Updates**
    
    - **Function**: `syr`, `her`, `syr2`, `her2`
    - **Operation**: Updates symmetric or Hermitian matrices.
    - **Usage**: Modifying matrices while maintaining symmetry.

### **Characteristics**

- **Memory Access Patterns**: Less efficient than Level 1 due to accessing both rows and columns, which may not be contiguous in memory.
- **Parallelism**: More opportunities for parallelism than Level 1, but still limited by memory bandwidth and access patterns.
- **Use Cases**:
    - Solving linear systems.
    - Eigenvalue problems.
    - Preconditioning in iterative methods.

### **Example Usage**

c