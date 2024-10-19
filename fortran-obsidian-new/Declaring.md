 Declaring a variable means to introduce a new variable to the program. You define its type and its name.

```csharp
int a; //a is declared
```




Examples of how to declare variables of various [[Scalar Types]]:

**[[Integer]]**

The declarations can be: INTEGER, INTEGER*2, INTEGER*4, INTEGER*8.

If you do **not** specify the size, a default size is used. The default size, for a declaration such as INTEGER H, can be altered by compiling with any of the options -dbl, -i2, -r8, or -xtypemap.


**INTEGER**
For a declaration such as INTEGER H, the variable H is usually one INTEGER*4 element in memory, interpreted as a single integer number. Specifying the size is nonstandard. If you do **not** specify the size, a default size is used.

```fortran
integer :: integer4
```


 **INTEGER\*2** 

For a declaration such as INTEGER*2 H, the variable H is always an INTEGER*2 element in memory, interpreted as a single integer number.

```fortran
integer*2 :: integer2
```

 **INTEGER\*4** 
For a declaration such as INTEGER*4 H, the variable H is always an INTEGER*4 element in memory, interpreted as a single integer number.

```fortran
integer*4 :: integer4
```


**INTEGER\*8** 

For a declaration such as INTEGER*8 H, the variable H is always an INTEGER*8 element in memory, interpreted as a single integer number.

```fortran
integer*8 :: integer8
```

Do not use INTEGER*8 variables or 8-byte constants or expressions when indexing arrays, otherwise, only 4 low-order bytes are taken into account. This action can cause unpredictable results in your program if the index value exceeds the range for 4-byte integers.




[[Real]]

**REAL**

For a declaration such as REAL W, the variable W is usually a REAL*4 element in memory, interpreted as a real number.

```fortran
    real :: real_number1
```

**REAL\*4**

For a declaration such as REAL*4 W, the variable W is always a REAL*4 element in memory, interpreted as a single-width real number

```fortran
    real*4 :: real_number4
```

 **REAL\*8** 

For a declaration such as REAL*8 W, the variable W is always a REAL*8 element in memory, interpreted as a double-width real number.

```fortran
    real*8 :: real_number8
```


**REAL\*16** 

**(SPARC only)** For a declaration such as REAL*16 W, the variable W is always an element of type REAL*16 in memory, interpreted as a quadruple-width real.

```fortran
    real*16 :: real_number16
```



[[Complex]]

The declarations can be: COMPLEX, COMPLEX*8, COMPLEX*16, or COMPLEX*32. Specifying the size is nonstandard.

 COMPLEX

For a declaration such as COMPLEX W, the variable W is usually two REAL*4 elements contiguous in memory, interpreted as a complex number.

```fortran
    complex :: complex_num
```

 COMPLEX\*8

For a declaration such as COMPLEX*8 W, the variable W is always two REAL*4 elements contiguous in memory, interpreted as a complex number.

```fortran
    complex*8 :: complex_8
```

 COMPLEX\*16

For a declaration such as COMPLEX*16 W, W is always two REAL*8 elements contiguous in memory, interpreted as a double-width complex number.

```fortran
    complex*16 :: complex_16
```


 COMPLEX\*32

**(SPARC only)** For a declaration such as COMPLEX*32 W, the variable W is always two REAL*16 elements contiguous in memory, interpreted as a quadruple-width complex number.

```fortran
    complex*32 :: complex_32
```




[[Logical Types]]

The declarations can be: LOGICAL, LOGICAL*1, LOGICAL*2, LOGICAL*4, LOGICAL*8

LOGICAL

For a declaration such as LOGICAL H, the variable H is usually one INTEGER*4 element in memory, interpreted as a single logical value. Specifying the size is nonstandard.



 LOGICAL*1 @

For a declaration such as LOGICAL*1 H, the variable H is always an BYTE element in memory, interpreted as a single logical value.

```fortran
    logical :: logic1
```

 LOGICAL*2 @

For a declaration such as LOGICAL*2 H, the variable H is always an INTEGER*2 element in memory, interpreted as a single logical value.

```fortran
    logical*2 :: logic2
```



 LOGICAL*4 @

For a declaration such as LOGICAL*4 H, the variable H is always an INTEGER*4 element in memory, interpreted as a single logical value.


```fortran
    logical*4 :: logic4
```



 LOGICAL*8 @

For a declaration such as LOGICAL*8 H, the variable H is always an INTEGER*8 element in memory, interpreted as a single logical value.

```fortran
    logical*8 :: logic8
```





[[Character]]

Each character occupies 8 bits of storage, aligned on a character boundary. Character arrays and common blocks containing character variables are packed in an array of character variables. The first character of one element follows the last character of the preceding element, without holes.

The length, len must be greater than 0. If len is omitted, it is assumed equal to 1.

For local and common character variables, symbolic constants, dummy arguments, or function names, **len** can be an integer constant, or a parenthesized integer constant expression.

For dummy arguments or function names, **len** can have another form: a parenthesized asterisk, that is, CHARACTER*(*), which denotes that the function name length is defined in referencing the program unit, and the dummy argument has the length of the actual argument.

For symbolic constants, **len** can also be a parenthesized asterisk, which indicates that the name is defined as having the length of the constant.


**Using Implicit Length**: If you omit the length in the declaration, Fortran will consider the default length to be 1.

```fortran
CHARACTER :: initial  ! Declares a character variable 'initial' with a length of 1
```

```fortran
CHARACTER(10) :: name
```


**How character variables work**

```fortran

program character_example
    implicit none
    CHARACTER(15) :: full_name
    CHARACTER(10) :: last_name = "Smith"
    CHARACTER :: initial = 'J'
    
    full_name = "John Smith"  ! Assigns a full name to full_name

    print *, "Initial: ", initial
    print *, "Last Name: ", last_name
    print *, "Full Name: ", full_name
end program character_example

```

When you run the above program, it will output:

```text
Initial:  J
 Last Name:  Smith
 Full Name:  John Smith
```




[[Derived Types]]



[[Pointers]]

