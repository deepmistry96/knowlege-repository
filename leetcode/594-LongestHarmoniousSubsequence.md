
Question
```
We define a harmonious array as an array where the difference between its maximum value and its minimum value is **exactly** `1`.

Given an integer array `nums`, return the length of its longest harmonious

subsequence

among all its possible subsequences.

**Example 1:**

**Input:** nums = [1,3,2,2,5,2,3,7]

**Output:** 5

**Explanation:**

The longest harmonious subsequence is `[3,2,2,2,3]`.

**Example 2:**

**Input:** nums = [1,2,3,4]

**Output:** 2

**Explanation:**

The longest harmonious subsequences are `[1,2]`, `[2,3]`, and `[3,4]`, all of which have a length of 2.

**Example 3:**

**Input:** nums = [1,1,1,1]

**Output:** 0

**Explanation:**

No harmonic subsequence exists.

**Constraints:**

- `1 <= nums.length <= 2 * 104`
- `-109 <= nums[i] <= 109`
```

This is categorized as a [[sliding window]] problem by leetcode.

So my initial thoughts are that we need to have at least one variable to keep track of the harmony number, but instead lets keep track of the index and use two additional variables to track the high and low.

I think this might be possible to do with only one variable but I think that maybe the easiest thing to do would be to create a second variable that would allow for us to easily calculate if we are going up or down, so we could have a high and a low variable. 
```cpp
int index = 0;
int windowSize = 1;
int endWindow = index+windowSize;
int low = 0;
int high = 0;
int largestHarmony = 0;

```


From there it would be easy to know how large the span of the low and the high are.  Until we go past a span of 1 we can continue to expand the size of the sliding window. 

```cpp


if nums[index] > nums[endWindow] //going down
{
	high = nums[index];
	low = nums[endWindow]
}
if nums[index] < nums[endWindow] //going up 
{
	low = nums[index];
	high = nums[endWindow]
}
```


Once we get a span larger than 1, we capture that as the largest window starting at that index. 
```cpp
if ((high - low) > 1)
{
	//now we need to lock in that window
	if (endWindow - index > largestHarmony)
	{
		largestHarmony = endWindow - index;
	}
}


```





Solution:
```cpp
class Solution {
public:
    int findLHS(vector<int>& nums) {
		int index = 0;
		int windowSize = 1;
		int endWindow = index+windowSize;
		int low = 0;
		int high = 0;
		int largestHarmony = 0;

		while (windowEnd < nums.length())
		{
			if (nums[index] == nums[windowSize])
			{
				index++;
				windowEnd++;
				continue; 
			}
			if nums[index] > nums[endWindow] //going down
			{
				high = nums[index];
				low = nums[endWindow]
			}
			if nums[index] < nums[endWindow] //going up 
			{
				low = nums[index];
				high = nums[endWindow]
			}	
			if (endWindow - index > largestHarmony)
			{
				largestHarmony = endWindow - index;
			}
			index++;
			windowEnd++;

		}
		return largestHarmony;
        
    }
};


```
