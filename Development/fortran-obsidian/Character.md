
- **Length Matters**: Always specify the length of your character strings; otherwise, Fortran defaults to a length of 1.
- **Initialization**: You can initialize character variables at the time of declaration or in separate assignments.
- **Manipulation Functions**: Utilize intrinsic functions for character manipulation, such as concatenation, finding length, and extracting substrings.





**Basic Character Declaration**: To declare a single character variable, you use the `CHARACTER` keyword followed by the desired length of the string in parentheses.

```fortran
CHARACTER(10) :: name  ! Declares a character variable 'name' with a length of 10
```

In this case, `name` can store up to 10 characters. If you attempt to store more than 10 characters in `name`, you will encounter a runtime error.


**Using Implicit Length**: If you omit the length in the declaration, Fortran will consider the default length to be 1.

```fortran
    CHARACTER :: initial  ! Declares a character variable 'initial' with a length of 1
    ```


### Initializing Character Variables

You can initialize character variables right after their declaration using the `=` assignment syntax:

```fortran
CHARACTER(10) :: greeting = "Hello"  ! Initializes the variable 'greeting' with the string "Hello"
```

### Example: Declaring and Initializing Characters

Here’s a small program that illustrates various ways to declare and initialize character variables:

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

### Output

When you run the above program, it will output:

```text
Initial:  J
 Last Name:  Smith
 Full Name:  John Smith
```


### String Manipulation

Fortran provides various intrinsic functions and operators to manipulate character strings. Here are some useful ones:

**Concatenation**: You can concatenate strings using the `//` operator.

```fortran
    CHARACTER(30) :: name1, name2, full_name
    name1 = "John"
    name2 = "Doe"
    full_name = name1 // " " // name2  ! Combines the two names with a space in between
    ```


**Length**: Obtain the length of a character string using the `LEN()` function.

```fortran
print *, "Length of full name: ", LEN(full_name)
```


### Substrings

To work with substrings, you can use the syntax:

```fortran
character_substring = full_name(1:4)  ! Extracts the first 4 characters of full_name
```

### Multi-Dimensional Character Arrays

Fortran allows you to declare arrays of characters as well. For example, you can declare an array of strings:

```fortran
CHARACTER(20), DIMENSION(5) :: names  ! An array that can hold 5 names with a max length o
```




