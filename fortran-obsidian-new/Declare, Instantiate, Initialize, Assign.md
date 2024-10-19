**[[Declaring]]** - Declaring a variable means to introduce a new variable to the program. You define its type and its name.

```csharp
int a; //a is declared
```

**[[Instantiating]]** - Instantiating a class means to create a new instance of the class. [Source](https://msdn.microsoft.com/en-us/library/dd460654.aspx#InstantiatingClasses).

```csharp
MyObject x = new MyObject(); //we are making a new instance of the class MyObject
```

**[[Initializing]]** - To initialize a variable means to assign it an initial value.

```csharp
int a; //a is declared
int a = 0; //a is declared AND initialized to 0
MyObject x = new MyObject(); //x is declared and initialized with an instance of MyObject
```

**[[Assigning]]** - Assigning to a variable means to provide the variable with a value.

```csharp
int a = 0; //we are assigning to a; yes, initializing a variable means you assign it a value, so they do overlap!
a = 1; //we are assigning to a
```