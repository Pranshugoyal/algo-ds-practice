
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

#https://practice.geeksforgeeks.org/problems/inversion-of-array-1587115620/1
def inversionCountInPlace(arr, l=0, r=None):
    if r is None:
        r = len(arr)
    if l >= r-1:
        return 0

    n = r - l
    mid = l + (n//2)

    count = inversionCountInPlace(arr, l, mid)
    count += inversionCountInPlace(arr, mid, r)
    res = []

    i, j = l, mid
    while i < mid and j < r:
        if arr[i] <= arr[j]:
            res.append(arr[i])
            i += 1
        else:
            res.append(arr[j])
            count += mid-i
            j += 1
    res += arr[i:mid]
    res += arr[j:r]
    arr[l:r] = res

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

################################################################################
# --------------------------------- Love's Sheet ----------------------------- #
################################################################################

#https://www.geeksforgeeks.org/minimize-the-maximum-difference-between-the-heights/amp/
def getMinDiff(arr, n, k):
    arr.sort()
    minDiff = arr[-1] - arr[0]
    highest, lowest = 0, 0
    for i in range(0, n-1):
        highest = max(arr[i]+k, arr[-1]-k)
        lowest = min(arr[0]+k, arr[i+1]-k)
        if highest >= 0 and lowest >= 0:
            minDiff = min(minDiff, highest-lowest)
    return minDiff

#https://practice.geeksforgeeks.org/problems/minimum-number-of-jumps-1587115620/1
def minJumps(arr):
    n = len(arr)
    lastMin, lastMax = 0, 0
    for jumps in range(1, n):
        newMax = lastMax + arr[lastMax]
        for cell in range(lastMin, lastMax):
            newMax = max(newMax, cell + arr[cell])

        #print("Jump", jumps, "Range:", lastMax+1, newMax)
        if newMax <= lastMax:
            return -1
        lastMin, lastMax = lastMax+1, newMax
        if lastMax >= n-1:
            return jumps

#https://practice.geeksforgeeks.org/problems/merge-two-sorted-arrays5135/1
def mergeWithoutExtraSpace(a, b):
    n, m = len(a), len(b)
    lo, hi = 0, n-1
    while lo < hi:
        mid = lo + (hi-lo)//2
        exchange = n - mid
        print(lo, hi, "Mid:", mid, "exchange:", exchange)
        if m >= exchange and a[mid] > b[exchange-1]:
            hi = mid
        else:
            lo = mid+1

    print("Search end, lo:", lo)
    if lo == n-1 and a[-1] < b[0]:
        return

    print("Exchange", n-lo, " numbers")
    a[lo:], b[:n-lo] = b[:n-lo], a[lo:]
    a.sort()
    b.sort()

#https://leetcode.com/problems/next-permutation/solution/
def nextLexicographicalPermutation(nums):
    n = len(nums)
    i = n-1
    while i >= 0:
        if nums[i] > nums[i-1]:
            break
        i -= 1

    i -= 1
    if i >= 0:
        for j in reversed(range(i+1, n)):
            if nums[j] > nums[i]:
                break

        nums[i], nums[j] = nums[j], nums[i]
    i, j = i+1, n-1
    while i < j:
        nums[i], nums[j] = nums[j], nums[i]
        i += 1
        j -= 1

#https://practice.geeksforgeeks.org/problems/count-pairs-with-given-sum5022/1
def countTwoSumPairs(arr, n, k):
    d = {}
    count = 0
    for e in arr:
        count += d.get(k-e, 0)
        d[e] = d.get(e, 0) + 1
    return count

#https://practice.geeksforgeeks.org/problems/common-elements1132/1
def sortedArrayIntersection(A, B, C):
    from heapq import heappush, heappop, heapify

    space = [A, B, C]
    h = [(space[i][0], 0, i) for i in range(3)]
    n = sum([len(a) for a in space])
    heapify(h)

    common, last, sets = [], None, set()
    for _ in range(n):
        next, i, li = heappop(h)
        l = space[li]
        if i < len(l)-1:
            heappush(h, (l[i+1], i+1, li))

        if last != next:
            last = next
            sets = set([li])
        elif len(sets) < 3:
            sets.add(li)
            if len(sets) == 3:
                common.append(next)
    return common

#https://www.geeksforgeeks.org/rearrange-array-alternating-positive-negative-items-o1-extra-space/amp/
def rearrangeArrayAlternatively(arr):
    def partition(arr):
        i = 0
        for j in range(len(arr)):
            if arr[j] < 0:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        return i

    i = partition(arr)
    i += i%2
    j = 1
    while i < len(arr) and arr[j] < 0:
        arr[i], arr[j] = arr[j], arr[i]
        i += 2
        j += 2
        print(arr, i, j)
    return arr
