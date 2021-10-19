
################################################################################
# ------------------------------ 1 to N Problems ----------------------------- #
################################################################################

def findDuplicateAndMissingNumber(list):
	def swap(i, j):
		(list[i], list[j]) = (list[j], list[i])

	for i in range(len(list)):
		n = list[i]
		while list[n-1] != n:
			swap(i, n-1)
			n = list[i]
	
	return list

#https://leetcode.com/problems/find-the-duplicate-number
def findDuplicate(nums):
    def countLessEqual(k):
        count = 0
        for n in nums:
            count += 1 if n <= k else 0
        return count

    lo, hi = 1, len(nums)-1
    while lo < hi:
        mid = lo + (hi-lo)//2
        if countLessEqual(mid) > mid:
            hi = mid
        else:
            lo = mid + 1
    return lo

#Very interesting, look at other solutions too on leetcode
#https://leetcode.com/problems/find-the-duplicate-number/
def findDuplicateFloyd(nums):
    slow, fast = 0, 0
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break

    slow = 0
    while nums[slow] != nums[fast]:
        slow = nums[slow]
        fast = nums[fast]

    return nums[slow]

#https://practice.geeksforgeeks.org/problems/find-duplicates-in-an-array/1
def duplicates(arr, n):
    for num in arr:
        arr[num%n] += n

    dupes = []
    for i in range(n):
        if arr[i]//n > 1:
            dupes.append(i)

    return dupes

#https://leetcode.com/problems/first-missing-positive/
def firstMissingPositive(self, nums: List[int]) -> int:
    n = 0
    for i in range(len(nums)):
        if 0 < nums[i] <= len(nums)+1:
            nums[i], nums[n] = nums[n], nums[i]
            n += 1

    for i in range(n):
        j = abs(nums[i]) - 1
        if j < n and nums[j] > 0:
            nums[j] *= -1

    for i in range(n):
        if nums[i] > 0:
            return i + 1
    return n+1
