## **Level 1 [[BLAS]]: Vector-Vector Operations**

### **Overview**

- **Operations**: Basic operations involving vectors.
- **Data Structures**: Operate on one-dimensional arrays (vectors).
- **Computational Complexity**: O(n), where n is the size of the vector.

### **Common Functions**

1. **Vector Addition/Subtraction**
    
    - **Function**: `axpy` (a * x plus y)
    - **Operation**: Computes y←αx+yy \leftarrow \alpha x + yy←αx+y
    - **Usage**: Scaling and adding vectors.
2. **Dot Product**
    
    - **Function**: `dot`
    - **Operation**: Computes the dot product ∑i=1nxiyi\sum_{i=1}^{n} x_i y_i∑i=1n​xi​yi​
    - **Usage**: Calculating inner products, projections.
3. **Scaling a Vector**
    
    - **Function**: `scal`
    - **Operation**: Computes x←αxx \leftarrow \alpha xx←αx
    - **Usage**: Multiplying a vector by a scalar.
4. **Copying Vectors**
    
    - **Function**: `copy`
    - **Operation**: Copies elements from one vector to another y←xy \leftarrow xy←x
    - **Usage**: Duplicating data.
5. **Swapping Vectors**
    
    - **Function**: `swap`
    - **Operation**: Swaps elements of two vectors x↔yx \leftrightarrow yx↔y
    - **Usage**: Rearranging data.
6. **Norm Computations**
    
    - **Function**: `nrm2`, `asum`
    - **Operation**: Computes vector norms (Euclidean norm, sum of absolute values)
    - **Usage**: Measuring vector magnitudes.
7. **Index of the Element with Maximum Absolute Value**
    
    - **Function**: `iamax`, `iamin`
    - **Operation**: Finds the index of the element with the largest/smallest absolute value.
    - **Usage**: Identifying significant elements.

### **Characteristics**

- **Memory Access Patterns**: Access elements sequentially, leading to efficient use of memory bandwidth.
- **Parallelism**: Limited opportunity for parallelism due to the simplicity of operations.
- **Use Cases**:
    - Fundamental building blocks for more complex algorithms.
    - Preprocessing steps in numerical methods.

### **Example Usage**

c