**Subsets**

A huge number of coding interview problems involve dealing with Permutations and Combinations of a given set of elements. The pattern Subsets describes an efficient Breadth First Search (BFS) approach to handle all these problems.

The pattern looks like this:
```
Given a set of [1, 5, 3]
 
1. Start with an empty set: [[]]
2. Add the first number (1) to all the existing subsets to create new subsets: [[], [1]];
3. Add the second number (5) to all the existing subsets: [[], [1], [5], [1,5]];
4. Add the third number (3) to all the existing subsets: [[], [1], [5], [1,5], [3], [1,3], [5,3], [1,5,3]].
```

Here is a visual representation of the Subsets pattern:

![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27800%27%20height=%27582.7342888643881%27/%3e)![image](https://hackernoon.imgix.net/images/G9YRlqC9joZNTWsi1ul7tRkO6tv1-hemg3w8d.jpg?w=1200&q=75&auto=format)

How to identify the Subsets pattern:

- Problems where you need to find the combinations or permutations of a given set

Problems featuring Subsets pattern:

- Subsets With Duplicates (easy)
- String Permutations by changing case (medium)