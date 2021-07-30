
#
# 1toNProblems
#

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

    print(slow, fast)
    slow = 0
    while nums[slow] != nums[fast]:
        slow = nums[slow]
        fast = nums[fast]
        print(slow, fast)

    return nums[slow]
