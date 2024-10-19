

Integrating Fortran with C and C++ can be quite beneficial, especially in scientific computing and high-performance applications, where each language has its strengths. Below are details on how these languages can interoperate:

### Fortran Calling C

#### 1. **C Functions in Fortran**

To call a C function from a Fortran program, some steps need to be followed:

- **Function Declaration**: Use the `INTERFACE` block in Fortran to declare the C function.
- **`ISO_C_BINDING` Module**: This standard module provides interoperability features that make it easier to call C functions.

**Example (Fortran Calling C):**

Here’s a simple example where Fortran calls a C function.

**C Code (my_c_function.c):**

```c
#include <stdio.h>

void hello_from_c() {
    printf("Hello from C!\n");
}

int add(int a, int b) {
    return a + b;
}
```

**Fortran Code (main.f90):**

```fortran
program call_c
    use iso_c_binding

    interface
        subroutine hello_from_c() bind(C)
        end subroutine hello_from_c

        function add(a, b) bind(C) result(res)
            import :: c_int
            integer(c_int), value :: a, b
            integer(c_int) :: res
        end function add
    end interface

    ! Calling the C function
    call hello_from_c()
    
    ! Using the C function to add two integers
    print *, '5 + 7 =', add(5, 7)

end program call_c
```


#### **Compiling**

To compile the above code, you need to compile the C code and then link it with the Fortran program.

```bash
gcc -c my_c_function.c -o my_c_function.o
gfortran main.f90 my_c_function.o -o my_program
```

### C++ Calling Fortran


#### **Fortran Functions in C++**

To call Fortran functions from C/C++, you typically need to ensure that the Fortran functions are made accessible (using `bind(C)`).

**Fortran Code (my_fortran_function.f90):**

```fortran
module myfortran
    implicit none
contains
    subroutine add_numbers(a, b, sum) bind(C, name="add_numbers")
        integer(c_int), value :: a, b
        integer(c_int) :: sum
        sum = a + b
    end subroutine add_numbers
end module myfortran
```

**C++ Code (main.cpp):**

```cpp
extern "C" {
    void add_numbers(int a, int b, int *sum);
}

#include <iostream>

int main() {
    int result;
    add_numbers(5, 7, &result);
    std::cout << "5 + 7 = " << result << std::endl;
    return 0;
}
```


#### **Compiling**

Just like with Fortran calling C, you would compile it like so:

```bash
gfortran -c my_fortran_function.f90 -o my_fortran.o
g++ main.cpp my_fortran.o -o main_program
```

### C and C++ Callable from Fortran

**1. Name Mangling and Interface Blocks** When calling C++ from Fortran, you'd need to take into account name mangling, which is a way compilers modify function names to support function overloading. To facilitate this:

- Use `extern "C"` in the C++ header or implementation to avoid name mangling.
