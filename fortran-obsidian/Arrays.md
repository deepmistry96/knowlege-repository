

Let's delve deeper into **arrays** in Fortran, including their types, declarations, initialization, and related concepts such as allocatable arrays and array manipulation.

### Arrays in Fortran

Arrays are fundamental data structures in Fortran, enabling efficient storage and manipulation of collections of elements of the same data type. They are particularly useful in scientific and numerical computations where operations on multiple data points are common.


### Summary

- **One-Dimensional Arrays**: Simple lists of elements of the same data type.
- **Multi-Dimensional Arrays**: Arrays extending into two or more dimensions, useful for representing matrices and grids.
- **Allocatable Arrays**: Flexible arrays that allow you to allocate memory dynamically, accommodating varying input sizes.



### **[[Character]] Arrays**

Character data can also be stored in arrays, allowing for strings to be handled efficiently.

```fortran
CHARACTER(20), DIMENSION(3) :: names  ! An array of 3 character strings, each up to 20 c
```



### **One-Dimensional Arrays**

**Definition**: A one-dimensional array can be thought of as a simple list of elements. It maintains a single index for accessing its elements.

**Declaration**: You can declare a one-dimensional array by specifying its type and the number of elements:

```fortran
INTEGER :: arr(10)   ! Declares an array of 10 integers
```

**Initialization**: You can initialize it at the time of declaration or assign values later:

```fortran
INTEGER :: arr(10) = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  ! Initialize with values
```

**Accessing Elements**: Elements are accessed using their index (Fortran uses [[1-based indexing]]):

```fortran
PRINT *, arr(1)  ! Accesses the first element (1)
arr(5) = 50      ! Sets the fifth element to 50
```


for a one-dimensional array, you can access elements by their index as follows:

```fortran
integer :: nums(5)      ! Declare an array of size 5
nums(1) = 10            ! Assign value 10 to the first element (indexing starts at 1)
nums(2) = 20            ! Assign value 20 to the second element
```

### **Multi-Dimensional Arrays**

**Definition**: Multi-dimensional arrays (like matrices) allow storage of data in more than one dimension. For instance, a two-dimensional array can represent a grid or matrix.

**Declaration**: To declare a multi-dimensional array, list the dimensions in parentheses:

```fortran
REAL :: matrix(5, 5)  ! Declares a 5x5 matrix of real numbers
```

**Initialization**: Multi-dimensional arrays can also be initialized:

```fortran
REAL :: matrix(2, 2) = RESHAPE([1.0, 2.0, 3.0, 4.0], [2, 2])  ! 2x2 matrix
```

**Accessing Elements**: Access elements using their row and column indices:

```fortran
matrix(1, 1) = 3.5  ! Sets the first element of the matrix to 3.5
PRINT *, matrix(2, 2)  ! Accesses the element in the second row and second column
```




### **Allocatable Arrays**

**Definition**: Allocatable arrays allow you to define arrays whose sizes can be determined during program execution, rather than at compile time. This is especially useful when the size of the data is not known beforehand.

**Declaration**: Use the `ALLOCATABLE` attribute when declaring the array:

```fortran
REAL, ALLOCATABLE :: dynamicArray(:)  ! declare dynamic array without size
```

**Allocation**: The actual size is defined at runtime using the `ALLOCATE` statement:

```fortran
ALLOCATE(dynamicArray(20))  ! Allocate an array of size 20
```

**Deallocation**: It’s important to release the memory once you are done using the array:

```fortran
DEALLOCATE(dynamicArray)  ! Deallocate the array to free up the memory
```

### Example of Using Different Array Types

Here’s an example program that demonstrates these concepts:

```fortran
program array_example
    implicit none
    INTEGER :: arr(10)           ! One-dimensional array
    REAL :: matrix(3, 3)         ! Two-dimensional array
    REAL, ALLOCATABLE :: dynamicArray(:)  ! Dynamic array

    ! Initialize one-dimensional array
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    ! Initialize two-dimensional array
    matrix = RESHAPE([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3])

    ! Allocate and initialize a dynamic array
    ALLOCATE(dynamicArray(5))
    dynamicArray = [10.0, 20.0, 30.0, 40.0, 50.0]

    ! Output values of the one-dimensional array
    PRINT *, "One-dimensional Array:"
    PRINT *, arr

    ! Output values of the two-dimensional array
    PRINT *, "Two-dimensional Matrix:"
    PRINT *, matrix

    ! Output values of the dynamic array
    PRINT *, "Dynamic Array:"
    PRINT *, dynamicArray

    ! Deallocate dynamic array
    DEALLOCATE(dynamicArray)
end program array_example
```

### Output

This program will produce output similar to the following:

```text
One-dimensional Array:
 1  2  3  4  5  6  7  8  9  10
 Two-dimensional Matrix:
  1.00000000  2.00000000  3.00000000
  4.00000000  5.00000000  6.00000000
  7.00000000  8.00000000  9.00000000
 Dynamic Array:
 10.00000000  20.00000000  30.00000000  40.00000000  50.00000000
```


