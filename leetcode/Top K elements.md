**Top K elements**

Any problem that asks us to find the top/smallest/frequent ‘K’ elements among a given set falls under this pattern.

The best data structure to keep track of ‘K’ elements is Heap. This pattern will make use of the Heap to solve multiple problems dealing with ‘K’ elements at a time from a set of given elements. The pattern looks like this:

1. Insert ‘K’ elements into the min-heap or max-heap based on the problem.
2. Iterate through the remaining numbers and if you find one that is larger than what you have in the heap, then remove that number and insert the larger one.

![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27800%27%20height=%271202.1649484536083%27/%3e)![image](https://hackernoon.imgix.net/images/G9YRlqC9joZNTWsi1ul7tRkO6tv1-uuow3wb1.jpg?w=1200&q=75&auto=format)

  

There is no need for a sorting algorithm because the heap will keep track of the elements for you.

How to identify the Top ‘K’ Elements pattern:

- If you’re asked to find the top/smallest/frequent ‘K’ elements of a given set
- If you’re asked to sort an array to find an exact element

Problems featuring Top ‘K’ Elements pattern:

- Top ‘K’ Numbers (easy)
- Top ‘K’ Frequent Numbers (medium)