
Fortran supports several data structures that allow for organized storage and manipulation of data. As a programming language primarily used for scientific computing and numerical problems, Fortran's data structures are designed to effectively handle arrays, records, and modules, among others. Let’s explore the main data structures supported by Fortran:


### Summary

- **[[Scalar Types]]**: [[INTEGER]], [[REAL]], [[COMPLEX]], [[LOGICAL]], [[CHARACTER]].
- **[[Arrays]]**: One-dimensional and multi-dimensional arrays.
- **[[Derived Types]]**: User-defined structures.
- **[[Pointers]]**: Support for dynamic memory management through pointers.
- **[[Modules]]**: Organization of related types and procedures.
- **[[Common Blocks]]**: Sharing variables across program units (less preferred in modern Fortran).

These structures collectively enhance the ability to efficiently manage and operate on data in Fortran, particularly in complex scientific and engineering computations.




### **Scalar Types**

**Scalar Types** include basic data types like [[INTEGER]], [[REAL]], [[COMPLEX]], [[LOGICAL]], and [[CHARACTER]]. Each of these can hold a single value at a time.



### **[[Arrays]]**

Fortran has powerful support for arrays, which are collections of elements of the same type. Arrays can be:


**One-Dimensional Arrays**: A simple list of elements.

```fortran
INTEGER :: arr(10)   ! An array of 10 integers
```


**Multi-Dimensional Arrays**: Arrays with two or more dimensions (e.g., matrices).

```fortran
REAL :: matrix(5, 5)  ! A 5x5 matrix of real numbers
```

**Implicit Arrays**: Fortran can create arrays without explicitly defining their size, using the `ALLOCATABLE` attribute for dynamic allocation.



### **Character Arrays**

Character data can also be stored in arrays, allowing for strings to be handled efficiently.

```fortran
CHARACTER(20), DIMENSION(3) :: names  ! An array of 3 character strings, each up t
```



### **Derived Types (Structures)**

Derived types are user-defined types that can group different data types into a single structure, similar to `struct` in C. They allow for organizing related attributes together.

```fortran
TYPE :: Person
    CHARACTER(30) :: name
    INTEGER :: age
    REAL :: height
END TYPE Person

TYPE(Person) :: individual  ! Declare a variable of derived type Person
```




### **Pointers**

Fortran supports pointers, which can point to variables of any data type, including derived types. Pointers can be used to implement linked data structures.

```fortran
REAL, POINTER :: p_var  ! Declare a pointer to a real number
ALLOCATE(p_var)         ! Dynamically allocate memory
```


### **Associative Arrays**

Fortran 2008 and later standards introduce associative arrays through `coarrays`. These can be used in parallel processing contexts but are more limited compared to true associative arrays found in other languages.



### **Modules**

Modules aren’t data structures per se, but they are a way to group related procedures and data types together, facilitating code organization and reusability. You can define derived types, interfaces, and procedures within a module and then use them in different parts of your program.

```fortran
MODULE MyModule
    TYPE :: Person
        CHARACTER(30) :: name
        INTEGER :: age
    END TYPE Person

    INTERFACE
        ! Define procedures to interact with the Person type
    END INTERFACE
END MODULE MyModule
```


### **Common Blocks**

Although somewhat outdated and generally less preferred compared to modules, common blocks allow for the sharing of variables across multiple program units. This feature enables data sharing but has less explicit structure compared to modules.

```fortran
COMMON /block_name/ variable_list
```