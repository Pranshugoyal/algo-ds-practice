
# SlidingWindowProblems

################################################################################
# --------------------------- Fixed Sized Window ----------------------------- #
################################################################################

from collections import deque

def maxSubarraySum(list, k):
	maxSum = -1
	lastSum = 0

	i = 0
	j = 0
	while j < len(list):
		lastSum += list[j]
		if j - i + 1 == k:
			maxSum = max(maxSum, lastSum)
			lastSum -= list[i]
			i += 1
		j += 1

	return maxSum

def firstNegativeNumberInSubArrays(list, k):
	q = deque()
	result = []
	i = 0
	j = 0
	while j < len(list):
		if list[j] < 0:
			q.append(list[j])

		if j - i + 1 == k:
			if len(q) > 0:
				result.append(q[0])
			else:
				result.append(0)

			if len(q) > 0 and list[i] == q[0]:
				q.popleft()
			i += 1
			j += 1
		else:
			j += 1
	return result

def countAnagramsInString(S, s):
	def createCounter(t):
		counter = {}
		for c in t:
			counter[c] = counter.get(c, 0) + 1
		return counter

	sCounter = createCounter(s)
	currentCounter = {}
	anagramCount = 0
	i = 0
	j = 0
	k = len(s)
	while j < len(S):
		currentCounter[S[j]] = currentCounter.get(S[j], 0) + 1
		if j - i + 1 == k:
			if currentCounter == sCounter:
				anagramCount += 1

			currentCounter[S[i]] -= 1
			if currentCounter[S[i]] == 0:
				currentCounter.pop(S[i], None)

			j += 1
			i += 1
		else:
			j += 1
	return anagramCount

################################################################################
# ------------------------- Variable Sized Window ---------------------------- #
################################################################################

# This is O(n) with O(n) space, brute force is O(n^2) with O(1) space
def largestSubarrayWithSumK(list, k):
	sumCache = {}

	sum = 0
	maxLen = 0
	for i in range(len(list)):
		sum += list[i]
		if sum == k:
			maxLen = i + 1
		
		if sum not in sumCache:
			# Choose smaller subarray with this sum
			# as we want to find longest subarray later
			sumCache[sum] = i

		if (sum-k) in sumCache:
			ri = sumCache[sum-k]
			subArraySize = i - ri
			maxLen = max(maxLen, subArraySize)

	return maxLen

def longestSubstringWithKUniqueCharacters(S, k):
	map = {}

	result = (0,0,0)
	i = 0
	j = 0
	while j < len(S):
		map[S[j]] = map.get(S[j], 0) + 1
		if len(map) == k:
			if j-i+1 > result[2]:
				result = (i,j,j-i+1)
			j += 1
		elif len(map) > k:
			map[S[i]] -= 1
			map[S[j]] -= 1
			if map[S[i]] == 0:
				map.pop(S[i])
			i += 1
		else:
			j += 1
	return S[result[0]:result[1]+1]

def longestSubstringWithAllUniqueCharacters(S):
	map = {}

	result = (0,0,0)
	i = 0
	j = 0
	while j < len(S):
		map[S[j]] = map.get(S[j], 0) + 1
		k = j - i + 1
		if len(map) == k:
			if k > result[2]:
				result = (i,j,k)
			j += 1
		elif len(map) < k:
			map[S[i]] -= 1
			if map[S[i]] == 0:
				map.pop(S[i])
			map[S[j]] -= 1
			i += 1
	return S[result[0]:result[1]+1]

def pickToys(S, k):
	map = {}
	result = (0,0,0)

	i = 0
	j = 0
	while j < len(S):
		map[S[j]] = map.get(S[j], 0) + 1
		if len(map) < k:	
			j += 1
		elif len(map) == k:
			if j - i + 1 > result[2]:
				result = (i,j,j-i+1)
			j += 1
		else:
			while len(map) > k:
				map[S[i]] -= 1
				if map[S[i]] == 0:
					map.pop(S[i])
				i += 1
	
	return S[result[0]:result[1]+1]

def minWindowSubstring(S, t):
	targetMap = {}
	map = {}
	mapCount = 0
	result = (0,0,len(S))
	for c in t:
		map[c] = 0
		targetMap[c] = targetMap.get(c,0) + 1

	i = 0
	j = 0
	def updateResult(r):
		k = j - i + 1
		if k < r[2]:
			r = (i,j,k)
		return r

	while j < len(S):
		if S[j] in map:
			map[S[j]] += 1
			mapCount += 1
		else:
			j += 1
			continue

		if mapCount >= len(t):
			if map[S[j]] == targetMap[S[j]]:
				result = updateResult(result)

			while mapCount >= len(t):
				result = updateResult(result)
				if S[i] in map:
					map[S[i]] -= 1
					mapCount -= 1
				i += 1
			if mapCount < len(t):
				i -= 1
				map[S[i]] += 1
				mapCount += 1

		j += 1
	
	return S[result[0]:result[1]+1]

def minWindowSubstring2(S, t):
	map = {}
	uCharCount = 0
	result = (0,0,len(S)+1)
	for c in t:
		map[c] = map.get(c,0) + 1
	uCharCount = len(map)

	i, j = 0, 0
	def updateResult(r):
		k = j - i + 1
		if k < r[2]:
			r = (i,j,k)
		return r

	while j < len(S):
		if S[j] not in map:
			j += 1
			continue

		map[S[j]] -= 1
		if map[S[j]] == 0:
			uCharCount -= 1
		
		while uCharCount == 0:
			result = updateResult(result)	
			if S[i] in map:
				map[S[i]] += 1
				if map[S[i]] > 0:
					uCharCount += 1
			i += 1

		j += 1
	return S[result[0]:result[1]+1]

#https://leetcode.com/problems/find-k-closest-elements/
def elementsClosestToK(arr, k, x):
    def searchOrCeil(arr, n):
        lo, hi = 0, len(arr)-1
        while lo < hi:
            mid = lo + (hi-lo)//2
            if arr[mid] >= n:
                hi = mid
            else:
                lo = mid+1
        return lo

    ceil = searchOrCeil(arr, x)
    if ceil == 0:
        return arr[:k]
    elif ceil == len(arr):
        return arr[n-k:]

    if arr[ceil] != x:
        if abs(arr[ceil-1]-x) <= abs(arr[ceil]-x):
            ceil -= 1

    i, j = ceil, ceil
    while j-i+1 < k:
        if i == 0:
            j += 1
            continue
        elif j == len(arr)-1:
            i -= 1
            continue

        if abs(arr[i-1]-x) <= abs(arr[j+1]-x):
            i -= 1
        else:
            j += 1

    return arr[i:j+1]

#https://leetcode.com/problems/longest-substring-of-all-vowels-in-order/
def longestBeautifulSubstring(word):
    def isPartBeautiful(d):
        order = "aeiou"
        for i in range(len(d)):
            c = order[i]
            if c not in d:
                return False 
            if i > 0 and d[c][0] < d[order[i-1]][-1]:
                return False
        return True

    n = len(word)
    i, j = 0,0
    d, lb = {}, 0
    while j < n:
        if word[j] not in d:
            d[word[j]] = deque([j])
        else:
            d[word[j]].append(j)

        while len(d) > 0 and not isPartBeautiful(d):
            c = word[i]
            d[c].popleft()
            if len(d[c]) == 0:
                d.pop(c)
            i += 1

        if len(d) < 5:
            j += 1
            continue

        lb = max(lb, j-i+1)
        j += 1

    return lb

def longestBeautifulSubstring2(word):
    seen = set()
    lo, longest = -1, 0
    for hi, c in enumerate(word):
        if hi > 0 and c < word[hi - 1]:
            seen = set()
            lo = hi - 1    
        seen.add(c)    
        if len(seen) == 5:
            longest = max(longest, hi - lo)
    return longest

#https://leetcode.com/problems/maximum-length-of-subarray-with-positive-product/
def getMaxLen(nums):
    i, j = 0, 0
    product = 1
    maxlen = 0
    while j < len(nums):
        if nums[j] < 0:
            product *= -1
            if product > 0:
                maxlen = max(maxlen, j-i+1)
        elif nums[j] == 0:
            while i < j:
                if nums[i] < 0:
                    product *= -1
                i += 1
                if product > 0:
                    maxlen = max(maxlen, j-i)
            i += 1
            product = 1
        else:
            if product > 0:
                maxlen = max(maxlen, j-i+1)
        j += 1

    while i < j:
        if nums[i] < 0:
            product *= -1
        i += 1
        if product > 0:
            maxlen = max(maxlen, j-i)

    return maxlen
