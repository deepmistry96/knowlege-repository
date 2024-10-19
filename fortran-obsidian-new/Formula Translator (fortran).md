

This is one of the first high level languages. It was created for the punchcard era of programming and was largely used by the scientific and mathematical communities at the very beginning.

It turns out that the language does a few things very well, and that is working in matrices. Lots of parallel compute and its not super hard to figure out how to do so with fortran, as long as the problem is well suited for the language. 

The same workloads could be done just as well or even better by modern languages, but the tough part is not rewriting the code, but rewriting the potentially nuanced calculation. Just because computer do math, does not mean that math translates very easily into a language that the computer can understand





### **What is Fortran?**

Fortran, derived from "Formula Translation," is one of the oldest programming languages. It was developed in the 1950s primarily for scientific and engineering applications. The language is known for its efficiency in numerical computation and its ability to handle array operations effectively.

### **[[Key Features]]**

- **[[Array]] Handling**: Fortran provides built-in support for array operations, making it suitable for complex mathematical computations.
- **Rich Set of Libraries**: Fortran has extensive libraries for linear algebra, numerical methods, etc.
- **Portability**: Programs written in Fortran can often be run on different platforms with little modification.
- **Fixed and Free Formats**: Older versions of Fortran (like Fortran 77) used a fixed-format style, whereas newer versions (like Fortran 90 and later) support free-format coding.


### **[[Basic Syntax]]**

Here's a small example to illustrate basic Fortran syntax:

```fortran
program hello_world
    print *, 'Hello, World!'
end program hello_world
```

### **[[Learning Path]]**

To start learning Fortran effectively, consider these steps:

- **Set Up Your Environment**: Install a Fortran compiler (like GNU Fortran or Intel Fortran) on your machine.
- **Get a Good Textbook or Online Course**: Textbooks like "Fortran 95/2003 for Scientists and Engineers" can be helpful. Online platforms like Coursera or edX may also offer courses.
- **Practice Coding**: Write small programs that solve basic problems to get familiar with the syntax.
- **Explore Numerical Libraries**: Get a sense of how libraries like LAPACK or BLAS work with Fortran for complex computations.


### **[[Common Pitfalls]]**

- **Forgetting about [[declaring]] the type: Fortran requires variables to be declared, and type mismatches can lead to errors.
- **Array Bounds**: Be attentive to array bounds; trying to access an array element outside its declared range often leads to runtime errors.



