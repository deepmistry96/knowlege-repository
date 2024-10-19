
Understanding the various data types in Fortran is crucial because they directly affect how you manipulate and store data. Pay special attention to the size and range of each type to ensure that your programs run correctly without data overflow or type mismatch errors.



### **[[Integer]]**

- **`INTEGER`**: Stores whole numbers. The default integer size is usually 4 bytes (range: -2,147,483,648 to 2,147,483,647) but can vary based on the system and compiler.
- **`INTEGER*1`**: 1 byte integer (range: -128 to 127).
- **`INTEGER*2`**: 2 bytes integer (range: -32,768 to 32,767).
- **`INTEGER*4`**: 4 bytes integer (range: -2,147,483,648 to 2,147,483,647). This is the default in many compilers.
- **`INTEGER*8`**: 8 bytes integer (range: -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807).





### **[[Real]]**

- **`REAL`**: Holds floating-point numbers (default size is typically 4 bytes).
- **`REAL*4`**: 4 bytes real number (approximate range: 1.18E-38 to 3.40E+38).
- **`REAL*8`**: 8 bytes real number (doubles the precision; approximate range: 2.23E-308 to 1.79E+308).
- **`REAL*16`**: 16 bytes extended precision floating-point number, which is not often used.





### **[[Complex]]**

- **`COMPLEX`**: Represents complex numbers with real and imaginary parts (default is usually 8 bytes).
- **`COMPLEX*8`**: 8 bytes complex number.
- **`COMPLEX*16`**: 16 bytes complex number.




### **[[Logical]]**

- **`LOGICAL`**: Represents true or false values, typically 4 bytes.
- **`LOGICAL*1`**: 1 byte logical value.




### **[[Character]]**

- **`CHARACTER`**: Used to hold character strings. You can define the length:




### **[[Derived Types]]**

- **`TYPE`**: Fortran allows you to define your own data types using `TYPE` to group different data elements, similar to structures in C.

```fortran
TYPE :: Student
```




### **[[Pointers]]**

Fortran also supports pointer types, which can point to objects of any type. This feature allows dynamic memory allocation.