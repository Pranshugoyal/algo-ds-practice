
################################################################################
# ----------------------------- Hashing Must Do GfG -------------------------- #
################################################################################

from functools import cmp_to_key
from collections import Counter

#https://practice.geeksforgeeks.org/problems/relative-sorting/0
def relativeSort(A1, N, A2, M):
	mapA2 = {}
	for i in range(len(A2)):
		if A2[i] not in mapA2:
			mapA2[A2[i]] = i
	
	def comparator(l,r):
		nonlocal mapA2
		if l in mapA2 and r in mapA2:
			li = mapA2[l]
			ri = mapA2[r]
			if li == ri:
				return 0
			else:
				return 1 if li > ri else -1
		elif l in mapA2:
			return -1
		elif r in mapA2:
			return 1
		else:
			if l == r:
				return 0
			else:
				return 1 if l > r else -1
	
	A1.sort(key=cmp_to_key(comparator))
	return A1

#https://leetcode.com/problems/sort-array-by-increasing-frequency/submissions/
def sortByFrequency(A):
	mapA = Counter(A)
	def comparator(l,r):
		nonlocal mapA
		if mapA[l] != mapA[r]:
			return -1 if mapA[l] < mapA[r] else 1
		else:
			return -1 if l > r else 1
	A.sort(key=cmp_to_key(comparator))
	return A

#https://practice.geeksforgeeks.org/problems/largest-subarray-with-0-sum/1
def maxSubArrayWith0Sum(arr):
	sumMap = {0:-1}
	maxLen = -1
	lastSum = 0

	for i in range(len(arr)):
		lastSum += arr[i]
		if lastSum in sumMap:
			j = sumMap[lastSum]
			maxLen = max(maxLen, i-j)
		else:
			sumMap[lastSum] = i
	
	return maxLen

#https://practice.geeksforgeeks.org/problems/common-elements5420/1
#https://leetcode.com/problems/intersection-of-two-arrays/submissions/
def common_element(v1,v2):
	mapV1 = Counter(V1)
	res = []
	for n in V2:
		if n in mapV1:
			res.append(n)
			mapV1[n] -= 1
			if mapV1[n] == 0:
				mapV1.pop(n)
	
	res.sort()
	return res

def fourSum(arr, s):
	def twoSum(arr, k):
		counter = Counter(arr)
		res = set()
		for n in arr:
			r = k-n
			if r in counter:
				if r == n:
					if counter[n] > 1:
						res.add(n)
				else:
					res.add(n)
		pairs = []
		for n in res:
			pairs.append([n, k-n])
		return pairs

	arr.sort()
	n = len(arr)
	counter = Counter(arr)
	selectionMap = {}
	res = []
	for i in range(n-2):
		for j in range(i+1, n-1):
			k = arr[i] + arr[j]
			pairs = twoSum(arr[j+1:], s-arr[i]-arr[j])
			for pair in pairs:
				selected = sorted(pair + [arr[i], arr[j]])
				if selected not in res:
					res.append(selected)
	res.sort()
	return res

#https://practice.geeksforgeeks.org/problems/swapping-pairs-make-sum-equal4142/1
def findSwapValues(a, n, b, m):
	sumA = sum(a)
	sumB = sum(b)

	if sumA < sumB:
		sumA, sumB = sumB, sumA
		a,b = b,a

	diff = sumA - sumB
	if diff%2 == 1:
		return -1

	cb = Counter(b)
	for i in a:
		j = i - diff//2
		if j in cb:
			return 1
	return -1

def findSwapValues2(a,b):
	sumA = sum(a)
	sumB = sum(b)

	if sumA < sumB:
		sumA, sumB = sumB, sumA
		a,b = b,a

	diff = sumA - sumB
	if diff%2 == 1:
		return -1
	
	a.sort()
	b.sort()
	
	target = diff//2
	i,j = 0,0
	while i < len(a) and j < len(b):
		d = a[i] - b[j]
		if d == target:
			return 1
		elif d > target:
			if j < len(b):
				j += 1
			else:
				break
		else:
			if i < len(a):
				i += 1
			else:
				break
	return -1

#https://practice.geeksforgeeks.org/problems/count-distinct-elements-in-every-window/1
def countDistinctElementsInEveryWindow(arr, k):
	m = {}
	res = []
	i,j = 0,0
	while j < len(arr):
		m[arr[j]] = m.get(arr[j], 0) + 1

		if j-i+1 < k:
			j += 1
			continue

		res.append(len(m))
		m[arr[i]] -= 1
		if m[arr[i]] == 0:
			m.pop(arr[i])
		i += 1
		j += 1
	return res

#https://practice.geeksforgeeks.org/problems/array-pair-sum-divisibility-problem3257/1
def canPair(nums, k):
	if len(nums)%2 != 0:
		return False

	r = {}
	for i in range(k):
		r[i] = 0
	
	for n in nums:
		r[n%k] += 1
	
	if r[0]%2 != 0:
		return False
	
	if k%2 == 0 and r[k//2]%2 == 1:
		return False

	for i in range(1,k//2 + 1):
		if r[i] != r[k-i]:
			return False
	
	return True

#https://practice.geeksforgeeks.org/problems/longest-consecutive-subsequence2449/1
def findLongestConseqSubseq(arr, N):
	unique = set(arr)
	start = min(unique)
	end = max(unique)

	maxLen = 1
	lastStart = start
	lastFound = start
	for i in range(start+1, end+1):
		if i in unique:
			if i == lastFound + 1:
				maxLen = max(maxLen, i-lastStart + 1)
			else:
				lastStart = i
			lastFound = i
	return maxLen

#https://practice.geeksforgeeks.org/problems/array-subset-of-another-array2317/1
def isSubset( a1, a2):
	superSet = set(a1)
	for i in a2:
		if i not in superSet:
			return False
	return True

def allPairs(A, B, N, M, X):
	setA = set(A)

	res = []
	for i in B:
		if X-i in setA:
			res.append((X-i,i))
	res.sort()
	return res

#https://practice.geeksforgeeks.org/problems/zero-sum-subarrays1825/1
def findZeroSumSubArrays(arr,n):
	sumMap = {0:1}
	lastSum = 0
	zeroCount = 0
	for i in range(len(arr)):
		lastSum += arr[i]
		if lastSum in sumMap:
			zeroCount += sumMap[lastSum]
			sumMap[lastSum] += 1
		else:
			sumMap[lastSum] = 1
	return zeroCount

def smallestWindow(s, p):
	res = (0,len(s))
	def updateResult(new):
		nonlocal res
		if new[1] - new[0] + 1 < res[1] - res[0] + 1:
			res = new
	
	target = Counter(p)
	tlen = len(target)
	i,j = 0,0
	while j < len(s):
		if s[j] in target:
			target[s[j]] -= 1
			if target[s[j]] == 0:
				tlen -= 1

		if tlen > 0:
			j += 1
			continue

		while tlen <= 0:
			if s[i] in target:
				if target[s[i]] < 0:
					target[s[i]] += 1
					i += 1
				else:
					break
			elif s[i] not in target:
				i += 1
		updateResult((i,j))
		j += 1
	
	cl = res[1] - res[0] + 1
	return res if cl <= len(s) else -1
