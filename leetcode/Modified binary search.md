**Modified binary search**

Whenever you are given a sorted array, linked list, or matrix, and are asked to find a certain element, the best algorithm you can use is the Binary Search. This pattern describes an efficient way to handle all problems involving Binary Search.

The patterns looks like this for an ascending order set:

1. First, find the middle of start and end. An easy way to find the middle would be: middle = (start + end) / 2. But this has a good chance of producing an integer overflow so it’s recommended that you represent the middle as: middle = start + (end — start) / 2
2. If the key is equal to the number at index middle then return middle
3. If ‘key’ isn’t equal to the index middle:
4. Check if key < arr[middle]. If it is reduce your search to end = middle — 1
5. Check if key > arr[middle]. If it is reduce your search to end = middle + 1

Here is a visual representation of the Modified Binary Search pattern:

![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27800%27%20height=%27600%27/%3e)![image](https://hackernoon.imgix.net/images/G9YRlqC9joZNTWsi1ul7tRkO6tv1-kxno3wrs.jpg?w=1200&q=75&auto=format)

  

Problems featuring the Modified Binary Search pattern:

Order-agnostic Binary Search (easy)Search in a Sorted Infinite Array (medium)