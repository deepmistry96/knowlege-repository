```fortran
Integer :: int1 = 0
```

In this case, the declaration is done via a DATA STATEMENT. 
The benefit of doing it this way  is that you can continue to instantiate variables after this statement. Take a look at this code and note where the error is being thrown:


```fortran
program hello
    implicit none
    
    Integer :: int1 = 0
    Integer :: int2
    integer :: int3 = 5
    int2 = 10
    integer :: int4 = 4
    print *, int1

end program hello
```

```cmd
.\hello-world.f90:10:23:

   10 |     integer :: int4 = 4
      |                       1
Error: Unexpected data declaration statement at (1)

```

In this code, `int2 = 10` is an **executable statement** that sets the value of `int2`. This line violates the rule that all variable declarations must come before any executable statements. Specifically, `integer :: int4 = 4` is attempting to declare a new variable, `int4`, after an [[executable statement]] has already been encountered
