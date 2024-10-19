In Fortran, **Derived Types** allow you to create custom data types that group together different attributes (data members) into a structured format. This feature is especially useful for managing complex data in situations where you want to encapsulate related pieces of information into a single unit. Derived types are similar to `struct` in C or classes in object-oriented languages.

### **Advantages of Derived Types**

- **Encapsulation**: Related data can be grouped together, making it easier to manage complex data structures.
- **Code Organization**: By defining custom types, your code becomes clearer and easier to read, as related attributes are organized logically.
- **Modularity**: Derived types can be passed between subroutines and functions, enhancing modular programming practices.


#### **Defining a Derived Type**

You define a derived type using the `TYPE` keyword, followed by the definition of its components. Each component can be of any valid Fortran data type (including scalar types, arrays, or even other derived types).

Here's how you can define a simple `Person` derived type, which groups together different attributes that describe a person:

```fortran
TYPE :: Person
    CHARACTER(30) :: name    ! A string to store the person's name
    INTEGER :: age            ! An integer to store the person's age
    REAL :: height            ! A real number to store the person's height
END TYPE Person
```

In this definition:

- `Person` is the name of the derived type.
- `name`, `age`, and `height` are attributes (or members) of the `Person` type, holding different data types.

#### **Declaring Variables of Derived Types**

Once the derived type is defined, you can create variables of that type:

```fortran
TYPE(Person) :: individual  ! A variable 'individual' of type Person
```

Here, `individual` is an instance of the `Person` type. You can now use this instance to store and manipulate data related to a specific person.



#### **Initializing and Using Derived Types**

You can set the values of the attributes of the derived type instance as follows:

```fortran
individual%name = 'Alice'      ! Access fields using '%'
individual%age = 30
individual%height = 5.5
```

In this code:

- The `%` symbol is used to access the components (attributes) of the derived type. You can assign values directly to them.

#### **Accessing Attributes**

You can print out the values of the attributes just like any other variable:

```fortran
PRINT *, 'Name: ', individual%name
PRINT *, 'Age: ', individual%age
PRINT *, 'Height: ', individual%height
```




### Example Program


Here's a complete example demonstrating the use of derived types:

```fortran
program derived_types_example
    implicit none

    ! Define the derived type
    TYPE :: Person
        CHARACTER(30) :: name
        INTEGER :: age
        REAL :: height
    END TYPE Person

    ! Declare a variable of the derived type
    TYPE(Person) :: individual

    ! Initialize the variable
    individual%name = 'Alice'
    individual%age = 30
    individual%height = 5.5

    ! Print the attributes of the Person type
    PRINT *, 'Name: ', individual%name
    PRINT *, 'Age: ', individual%age
    PRINT *, 'Height: ', individual%height

end program derived_types_example
```

### Output

When you run the above program, you will see the following output:

```text
Name:  Alice
 Age: 30
 Height:  5.50000000
```


### **Using Derived Types with [[Arrays]]**

You can also create arrays of derived types. For example, if you want to maintain a list of multiple `Person`s:

```fortran
TYPE(Person), DIMENSION(10) :: people   ! Array of 10 Person type instances
```

This way, you can have an array that holds multiple persons’ attributes, facilitating operations like iteration over their details.