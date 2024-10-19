Question:
```
Write an algorithm to determine if a number `n` is happy.

A **happy number** is a number defined by the following process:

- Starting with any positive integer, replace the number by the sum of the squares of its digits.
- Repeat the process until the number equals 1 (where it will stay), or it **loops endlessly in a cycle** which does not include 1.
- Those numbers for which this process **ends in 1** are happy.

Return `true` _if_ `n` _is a happy number, and_ `false` _if not_.

**Example 1:**

**Input:** n = 19
**Output:** true
**Explanation:**
12 + 92 = 82
82 + 22 = 68
62 + 82 = 100
12 + 02 + 02 = 1

**Example 2:**

**Input:** n = 2
**Output:** false

**Constraints:**

- `1 <= n <= 231 - 1`
```

This is a 


```cpp

class Solution {
public:
bool isHappy(int n) {
int sum = 0;
int lastDigit = 0;
int count = 0;
bool loop = true;
//We want to loop until either we
while ( loop )
{ 
	//Get a sum of 1 for the module once we reach the end of the number
	if ( n < 10 ) //if we get to the last number, set n to the the sum and iterate that we fully evaluated the number
	{
		sum += n * n;
		if ( sum == 1)
		{
			loop = false;
			return true;
		}
		n = sum;
		sum = 0;
		count++;
	}
	else
	{
		lastDigit = n % 10; //will pull the last number
		n = n / 10; //will pop off the lasta number
		sum += ( lastDigit * lastDigit) ; //will take the last number, square it and add it to sum
	}
	//Or we have determined that this will continue forever
	if (count > 1000)
	{
		loop = false;
		return false;
	} 
}
return false;
}
};

```