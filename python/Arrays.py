
################################################################################
# --------------------------- GeeksForGeeks Must Do List --------------------- #
################################################################################
# https://www.geeksforgeeks.org/must-do-coding-questions-for-companies-like-amazon-microsoft-adobe/#arrays

def subArraySum(arr, n, s):
	i,j = 0,0
	current = 0
	while j < n:
		current += arr[j]
		if current > s:
			while current > s and i <= j:
				current -= arr[i]
				i += 1

			if i > j:
				return -1
		if current == s:
			break
		j += 1
	return [i+1, j+1]

def countTriplet(arr, n):
	arr.sort()
	def findSum(s,e,k) -> int:
		i,j = s,e
		count = 0
		while i < j:
			sum = arr[i] + arr[j]
			if sum == k:
				count += 1
				i += 1
				j -= 1
			elif sum < k:
				i += 1
			else:
				j -= 1
		return count

	count = 0
	for i in range(n-1, 1,-1):
		count += findSum(0,i-1,arr[i])

	return count

def maxSubArraySum(a,size):

    def maxSubArrayPrefixSum(a):
        prefixSumArray = [a[0]]
        for i in a[1:]:
            prefixSumArray.append(i+prefixSumArray[-1])

        minSumArray = [min(0, prefixSumArray[0])]
        for s in prefixSumArray[1:]:
            if s < minSumArray[-1]:
                minSumArray.append(s)
            else:
                minSumArray.append(minSumArray[-1])

        import sys
        maxSum = -sys.maxsize
        for i in reversed(range(size)):
            cs = prefixSumArray[i] - minSumArray[i]
            if cs == 0:
                maxSum = max(maxSum, a[i])
            else:
                maxSum = max(maxSum, cs)
        return maxSum

    def maxSubarraySum(arr):
        currentSum = 0
        minSum = 0
        maxSubSum = arr[0]

        for n in arr:
            currentSum += n
            maxSubSum = max(currentSum-minSum, maxSubSum)
            minSum = min(minSum, currentSum)
        return maxSubSum

    def kadanesMaxSubarraySum(nums):
        ls, maxSum = nums[0], nums[0]
        for i in range(1, len(nums)):
            ls = max(ls+nums[i], nums[i])
            maxSum = max(maxSum, ls)
        return maxSum
    
    return maxSubarraySum(arr)

def partition(A,B):
	i,j,k = 0,0,len(A)-1
	while i < k:
		if A[i] < B[j]:
			i += 1
		else:
			(A[k], B[j]) = (B[j], A[k])
			j += 1
			k -= 1
	
	A.sort()
	B.sort()
	print(A,B)

def inversionCountBruteForce(arr) -> int:
	count = 0
	for i in range(n):
		for j in range(i+1,n):
			if arr[j] < arr[i]:
				#print(arr[i], arr[j])
				count += 1
	return count

def inversionCountMerge(arr) -> int:
	count = 0

	def mergeSortAndCount(arr):
		if len(arr) < 2:
			return arr

		nonlocal count
		mid = len(arr)//2
		#print("Divide:", arr, "in", arr[:mid], arr[mid:])
		a = mergeSortAndCount(arr[:mid])
		b = mergeSortAndCount(arr[mid:])

		i,j = 0,0
		merged = []
		#print(a,b, "Merged:", merged)
		while len(merged) < len(arr):
			#print(i,j, "merged:", merged)
			if i < len(a) and j < len(b):
				if a[i] <= b[j]:
					merged.append(a[i])
					i += 1
				else:
					merged.append(b[j])
					j += 1
					count += len(a) - i
			else:
				while i < len(a):
					merged.append(a[i])
					i += 1
				while j < len(b):
					merged.append(b[j])
					j += 1
		return merged

	mergeSortAndCount(arr)
	return count

def minimumPlatform(n,arr,dep):
	events = []
	for i in range(n):
		events.append((arr[i], -1))
		events.append((dep[i], 1))

	# lambda not required as event is sorted acc to k1 then k2 in increasing order
	# Both keys are set to behave correctly with this behavoir
	# else this could also be used: sort(key = lambda e: (e[0], -e[]))
	events.sort()
	print(events)
	prefix = 0
	maxP = 0
	for event in events:
		prefix += -event[1]
		maxP = max(maxP, prefix)
	
	return maxP

def trappingRainWater(A) -> int:
	prefixMax = [A[0]]
	for h in A[1:]:
		prefixMax.append(max(prefixMax[-1], h))

	suffixMax = [A[-1]]
	for h in reversed(A[:-1]):
		suffixMax.append(max(suffixMax[-1], h))
	suffixMax.reverse()

	water = 0
	for i in range(1,len(A)-1):
		h = min(suffixMax[i],prefixMax[i])
		water += max(h-A[i],0)
	
	return water

def pythagoreanTriplets(arr):
	# 1 <= N <= 107
	# 1 <= arr[i] <= 1000

	MAX = 1000 + 1
	countMap = {}
	for i in arr:
		countMap[i] = countMap.get(i,0) + 1
	
	for i in range(MAX):
		if i not in countMap:
			continue

		for j in range(MAX):
			if j not in countMap:
				continue

			if i == j and countMap[i] == 1:
				continue

			squareSum = (i*i) + (j*j)
			c = int(squareSum**0.5)
			if c*c != squareSum:
				# its not a perfect square
				continue

			if c in countMap:
				return True
	return False

def chocolateDistributionProblem(A,N,M):
	print(A)
	A.sort()
	print(A)

	i, j = 0, M-1
	diff = A[j] - A[i]
	while j < N:
		nd = A[j] - A[i]
		diff = min(diff,nd)
		i += 1
		j += 1
	return diff

def zigZag(arr, n):
	def swap(i):
		j = i -1
		arr[i], arr[j] = arr[j], arr[i]

	for i in range(1,n):
		if i%2 == 0 and arr[i] > arr[i-1]:
			swap(i)
		elif i%2 != 0 and arr[i] < arr[i-1]:
			swap(i)
	print(arr)

def printMatrixInSpiral(matrix,r,c):
	visited = []
	for ri in range(r):
		row = [False for ci in range(c)]
		visited.append(row)

	def nextDirection(cell, currentDirection, visited):
		nextDirection = {
						(0,1): (1,0),
						(1,0): (0,-1),
						(0,-1): (-1,0),
						(-1,0): (0,1)
						}

		nextCell = (cell[0]+currentDirection[0],cell[1]+currentDirection[1])
		if nextCell[0] < r and nextCell[0] >= 0 and nextCell[1] >= 0 and nextCell[1] < c:
			isVisited = visited[nextCell[0]][nextCell[1]]
			if isVisited:
				return nextDirection[currentDirection]
			else:
				return currentDirection
		else:
			return nextDirection[currentDirection]


	stepsRemaining = r*c
	currentCell = (0,0)
	direction = (0,1)
	traverseList = []
	while stepsRemaining > 0:
		#print("Visiting:", currentCell,matrix[currentCell[0]][currentCell[1]])
		traverseList.append(matrix[currentCell[0]][currentCell[1]])
		visited[currentCell[0]][currentCell[1]] = True
		direction = nextDirection(currentCell, direction, visited)
		currentCell = (currentCell[0]+direction[0], currentCell[1]+direction[1])
		stepsRemaining -= 1
	
	return traverseList

def createLargestNumber(arr):
	import functools
	def greater(a, b):
		a1 = int(a+b)
		b1 = int(b+a)
		if a1 == b1:
			return 0
		else:
			return 1 if a1 > b1 else -1

	arr.sort(key=functools.cmp_to_key(greater), reverse=True)
	return "".join(arr)

# https://practice.geeksforgeeks.org/problems/equivalent-sub-arrays3731/1/
def equivalentSubArrays(arr):
	map = {}
	for n in arr:
		map[n] = map.get(n, 0) + 1
	
	TOTAL = len(map)
	map = {}
	count = 0
	i, j = 0,0
	while j < len(arr):
		map[arr[j]] = map.get(arr[j], 0) + 1

		if len(map) < TOTAL:
			j += 1
		else:
			while len(map) >= TOTAL and i <= j:
				if len(map) == TOTAL:
					count += len(arr) - j

				map[arr[i]] -= 1
				if map[arr[i]] == 0:
					map.pop(arr[i])
				i += 1
			j +=1
	return count

#https://practice.geeksforgeeks.org/problems/minimum-swaps/1
def minSwaps(nums):
    nSorted = sorted(nums)
    h = {}

    for i in range(len(nums)):
        h[nums[i]] = i

    def correct(i):
        c = h[nSorted[i]]
        nums[i], nums[c] = nums[c], nums[i]
        h[nums[i]] = i
        h[nums[c]] = c

    swaps = 0
    for i in range(len(nums)):
        if nums[i] == nSorted[i]:
            continue

        swaps += 1
        correct(i)
    return swaps

#https://leetcode.com/problems/reduce-array-size-to-the-half/
def reduceArrayToHalf(arr):
    from collections import Counter

    counts = sorted(Counter(arr).values(), reverse=True)
    setSize, target = 0, len(arr)//2
    for n in counts:
        setSize += 1
        target -= n
        if target <= 0:
            break
    return setSize

#https://leetcode.com/problems/reshape-the-matrix/
def matrixReshape(mat, r, c):
    n, m = len(mat), len(mat[0])
    if n*m != r*c:
        return mat
    
    res = []
    for i in range(r):
        res.append([])
    for k in range(n*m):
        res[k//c].append(mat[k//m][k%m])
    return res
