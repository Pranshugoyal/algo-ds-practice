
################################################################################
# ---------------------------- Interval Selection ---------------------------- #
################################################################################

#https://www.geeksforgeeks.org/activity-selection-problem-greedy-algo-1/
def intervalSelection(start, finish) -> int:
	n = len(start)
	if n <= 1:
		return n

	intervals = []
	for i in range(n):
		intervals.append((start[i], finish[i]))

	intervals.sort(key=lambda x: x[1])
	selected = [intervals[0]]
	for i in range(1,n):
		if intervals[i][0] >= selected[-1][1]:
			selected.append(intervals[i])

	return len(selected)

#https://www.geeksforgeeks.org/job-sequencing-problem/
def jobSequencingProblem(jobs) -> int:
	n = len(jobs)
	if n <= 1:
		return sum(jobs)

	def findSlot(slots,job) -> int:
		for slot in range(job[1]-1,-1,-1):
			if slots[slot] == None:
				return slot
		return -1

	jobs.sort(key=lambda x: x[2], reverse=True)
	chosen = [None for x in range(n)]
	for job in jobs:
		#find slot for this job
		slot = findSlot(chosen, job)
		if slot != -1:
			chosen[slot] = job[0]

	final = []
	for job in chosen:
		if job != None:
			final.append(job)
	return final

################################################################################
# ---------------------------- Greedy Must Do GfG ---------------------------- #
################################################################################

from collections import deque

#https://practice.geeksforgeeks.org/problems/activity-selection-1587115620/1
def activitySelection(n,start,end):
	pairs = []
	for i in range(n):
		pairs.append((start[i], end[i]))
	
	pairs.sort(key=lambda x: (x[1], x[0]))
	last = pairs[0]
	count = 1
	for activity in pairs[1:]:
		if activity[0] > last[1]:
			count += 1
			last = activity
	return count

#https://practice.geeksforgeeks.org/problems/choose-and-swap0531/1#
def chooseandswap(A):
	charSet = set()
	firstOrder = []
	for c in A:
		if c not in charSet:
			firstOrder.append(c)
			charSet.add(c)
	
	smallestNext = [None for i in range(len(firstOrder))]
	minYet = firstOrder[-1]
	for i in reversed(range(len(firstOrder[:-1]))):
		if firstOrder[i] <= minYet:
			smallestNext[i] = None
			minYet = firstOrder[i]
		else:
			smallestNext[i] = minYet

	replaceMap = {}
	for i in range(len(firstOrder)):
		if smallestNext[i]:
			replaceMap[firstOrder[i]] = smallestNext[i]
			replaceMap[smallestNext[i]] = firstOrder[i]
			break
	
	res = ""
	for c in A:
		res += replaceMap.get(c, c)
	return res

#https://practice.geeksforgeeks.org/problems/maximize-toys0331/1
def maxToyCount(toyPrices, money):
	toyPrices.sort()
	count = 0
	for price in toyPrices:
		if money >= price:
			count += 1
			money -= price
		else:
			break
	return count

#https://practice.geeksforgeeks.org/problems/page-faults-in-lru5603/1
def pageFaults(N, C, pages):
	if len(pages) <= C:
		return len(set(pages))

	cacheQ = deque()
	cacheSet = set()
	faults = 0

	for request in pages:
		if request in cacheSet:
			if cacheQ[-1] != request:
				# Bring request to front if it's not
				cacheQ.remove(request)
				cacheQ.append(request)
			continue
		
		if len(cacheQ) == C:
			cacheSet.remove(cacheQ.popleft())

		cacheQ.append(request)
		cacheSet.add(request)
		faults += 1
	
	return faults

#https://practice.geeksforgeeks.org/problems/largest-number-possible5028/1
def largestNumberPossible(digitCount, digitSum):
	if digitSum == 0:
		return 0 if digitCount == 1 else -1
	if digitCount*9 < digitSum:
		return -1
	
	nines = digitSum//9
	rem = digitSum%9
	s = ("9"*nines) + (str(rem) if rem > 0 else "")
	if len(s) < digitCount:
		s += "0" * (digitCount-len(s))
	return s

#https://practice.geeksforgeeks.org/problems/shop-in-candy-store1145/1
def candyStore(candies,N,K):
	candies.sort()

	totalCandyTypes = 0
	i = 0
	minMoneySpent = 0
	while totalCandyTypes < N:
		totalCandyTypes += K + 1
		minMoneySpent += candies[i]
		i += 1

	totalCandyTypes = 0
	i = 0
	maxMoneySpent = 0
	while totalCandyTypes < N:
		totalCandyTypes += K + 1
		maxMoneySpent += candies[-i-1]
		i += 1
	
	return [minMoneySpent, maxMoneySpent]

#https://practice.geeksforgeeks.org/problems/geek-collects-the-balls5515/1
def maxBalls(N, M, a, b):
	def getMatchingNumberIndex(a,b):
		print("Comparison Received:\n", a, "\n",b)
		i,j = 0,0
		aSum, bSum = 0,0
		while i < len(a) and j < len(b):
			if a[i] < b[j]:
				aSum += a[i]
				i += 1
			elif b[j] < a[i]:
				bSum += b[j]
				j += 1
			else:
				bSum += b[j]
				aSum += a[i]
				return (i,j,max(aSum, bSum)) 

		return (len(a), len(b), max(sum(a), sum(b))) 
	
	i,j = 0,0
	balls = 0
	while i < len(a) and j < len(b):
		ii,ji,collection = getMatchingNumberIndex(a[i:], b[j:])
		balls += collection
		i += ii + 1
		j += ji + 1
		print("Match found:", i-1, j-1, collection)
	return balls

#https://leetcode.com/problems/patching-array/
def minPatches(self, nums: List[int], n: int) -> int:
    reach, patches, i = 0, 0, 0
    while reach < n:
        if i < len(nums) and nums[i] <= reach+1:
            reach += nums[i]
            i += 1
        else:
            patches += 1
            #Add reach+1 to list
            reach += reach + 1
    return patches
