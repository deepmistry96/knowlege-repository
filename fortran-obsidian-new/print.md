In Fortran, you can print output to the screen (standard output) using the `print` statement or the `write` statement. Here’s a refresher on how to use both methods.

### Using `print`

The simplest way to print text or variable values is by using the `print` statement. Here’s the syntax:

```fortran
print *, 'Hello, World!'
```

- The `*` indicates list-directed output, which allows you to print various types of data without needing to specify format explicitly.

### Example:

```fortran
program print_example
    implicit none
    integer :: a
    a = 42
    print *, 'The value of a is:', a
end program print_example
```


### Using `write`

The `write` statement is more flexible and allows you to specify a format for the output. Here's the basic syntax:

```fortran
write(unit, format) item1, item2, ...
```

- **`unit`**: Specify `*` for standard output or `6` (the default).
- **`format`**: You can use a format specifier or an asterisk (`*`) for list-directed formatting.

### Example with Formatting:

```fortran

program write_example
    implicit none
    integer :: a
    real :: b
    a = 42
    b = 3.14
    write(*, '(A, I3, A, F5.2)') 'The value of a is: ', a, ' and b is: ', b
end program write_example

```


### Format Specifier Details:

- `'A'` is used for character output.
- `'I3'` specifies integer output with a width of 3.
- `'F5.2'` specifies floating-point output with a width of 5 and 2 decimal places.
