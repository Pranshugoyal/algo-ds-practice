
################################################################################
# --------------------------- 0/1 Knapsack Problems -------------------------- #
################################################################################

def knapsackProblem(weights, values, W):
	n = len(weights)
	t = []
	for i in range(n+1):
		t.append([-1 for j in range(W+1)])

	for j in range(W+1):
		t[0][j] = 0
	for i in range(n+1):
		t[i][0] = 0

	def show(k):
		for row in k:
			print(row)
		print()

	for i in range(1,n+1):
		#show(t)
		for j in range(1,W+1):
			if weights[i-1] > j:
				t[i][j] = t[i-1][j]
			else:
				t[i][j] = max(t[i-1][j], t[i-1][j-weights[i-1]]+values[i-1])

	#show(t)
	return t[n][W]

# This is a recursive version
def subsetSumProblem(list, n):
	#print("Testing for", n, list)
	if len(list) == 1:
		return list[0] == n
	
	head = list[:-1]
	tail = list[-1]
	if tail == n:
		return True

	if tail < n:
		return subsetSumProblem(head, n-tail) or subsetSumProblem(head, n)
	else:
		return subsetSumProblem(head, n)

def subsetSumProblemIterative(list, s):
	n = len(list)
	t = []
	for i in range(n+1):
		t.append([None for j in range(s+1)])
	
	for i in range(s+1):
		t[0][i] = False
	
	for i in range(n+1):
		t[i][0] = True

	for i in range(1, n+1):
		for j in range(1, s+1):
			if list[i-1] == j:
				t[i][j] = True
			elif list[i-1] > j:
				t[i][j] = t[i-1][j]
			else:
				t[i][j] = t[i-1][j] or t[i-1][j-list[i-1]]
	
	return t

def countSubsetsWithSum(list, s):
	n = len(list)
	t = []
	for i in range(n+1):
		t.append([None for j in range(s+1)])
	
	for i in range(s+1):
		t[0][i] = 0
	
	for i in range(n+1):
		t[i][0] = 1

	for i in range(1, n+1):
		for j in range(1, s+1):
			if list[i-1] > j:
				t[i][j] = t[i-1][j]
			else:
				t[i][j] = t[i-1][j] + t[i-1][j-list[i-1]]
	
	return t

def sumPartition(list, diff = 0):
	print("Testing for", diff, "in", list)
	if len(list) == 1:
		return abs(list[0]) == abs(diff)

	if diff != 0:
		return sumPartition(list[:-1], abs(diff-list[-1])) or sumPartition(list[:-1], abs(diff+list[-1]))
	else:
		return sumPartition(list[:-1], abs(list[-1]))

def minSubsetDifference(list):
	if len(list) == 1:
		return list[0]

	sum = 0
	for i in list:
		sum += i
	half = sum//2

	for s1 in range(half, -1, -1):
		if subsetSumProblem(list, s1):
			return sum - (2*s1) 

def targetSum(list, t):
	if len(list) == 1:
		return 1 if list[0] == abs(t) else 0
	
	head = list[:-1]
	tail = list[-1]

	positive = targetSum(head, t+tail)
	negative = targetSum(head, t-tail)
	return positive + negative

################################################################################
# ------------------------ Unbounded Knapsack Problems ----------------------- #
################################################################################

def unboundedKnapsackProblem(weights, values, W):
	print(weights, values, W)
	if len(weights) == 0 or W == 0:
		return 0

	notSelected = unboundedKnapsackProblem(weights[:-1], values[:-1], W)
	if weights[-1] <= W:
		selected = unboundedKnapsackProblem(weights, values, W-weights[-1]) + values[-1]
		return max(notSelected, selected)
	else:
		return notSelected
	
def coinChangeI(coins, target):
	if target == 0:
		return 1
	elif len(coins) == 0:
		return 0

	notSelected = coinChangeI(coins[:-1], target)
	if coins[-1] <= target:
		selected = coinChangeI(coins, target-coins[-1])
		return selected + notSelected
	else:
		return notSelected

def coinChangeII(coins, target):
	import sys
	ans = 0
	if target == 0:
		ans = 0
	elif len(coins) == 0:
		ans = sys.maxsize - 1
	else:	
		notSelected = coinChangeII(coins[:-1], target)
		if coins[-1] <= target:
			selected = coinChangeII(coins, target-coins[-1]) + 1
			ans = min(selected, notSelected)
		else:
			ans = notSelected
	
	return ans

################################################################################
# -------------------------- Longest Common Subsequence ---------------------- #
################################################################################

def lcsCount(x,y):
	if len(x) == 0 or len(y) == 0:
		return 0

	if x[-1] == y[-1]:
		return lcsCount(x[:-1], y[:-1]) + 1
	else:
		return max(lcsCount(x[:-1], y), lcsCount(x, y[:-1]))

def lcsString(x,y):
	if len(x) == 0 or len(y) == 0:
		return ""

	if x[-1] == y[-1]:
		return lcsString(x[:-1], y[:-1]) + x[-1]
	else:
		a = lcsString(x[:-1], y)
		b = lcsString(x, y[:-1])
		return a if len(a) > len(b) else b

def lcsCountTopDown(x,y):
	t = []
	for i in range(len(x)+1):
		t.append([-1 for j in range(len(y)+1)])
	
	for j in range(len(y)+1):
		t[0][j] = 0
	for i in range(len(x)+1):
		t[i][0] = 0

	for i in range(1, len(x)+1):
		for j in range(1, len(y)+1):
			if x[i-1] == y[j-1]:
				t[i][j] = t[i-1][j-1] + 1
			else:
				t[i][j] = max(t[i-1][j], t[i][j-1])
	
	return t

def lcsStringTopDown(x,y):
	t = lcsCountTopDown(x,y)

	i = len(x)
	j = len(y)
	s = ""
	while i > 0 and j > 0:
		if x[i-1] == y[j-1]:
			s += x[i-1]
			j -= 1
			i -= 1
		else:
			if t[i][j] == t[i-1][j]:
				i -= 1
			else:
				j -= 1

	return s[::-1]

def longestPalindromicSubsequenceCount(s):
	return lcsCount(s, s[::-1])

def longestPalindromicSubsequence(s):
	#Using LCS as base code
	return lcsString(s, s[::-1])

	#Recursive solution
	if len(s) < 2:
		return s
	
	if s[0] == s[-1]:
		return s[0] + longestPalindromicSubsequence(s[1:-1]) + s[-1]
	else:
		X = longestPalindromicSubsequence(s[1:])
		Y = longestPalindromicSubsequence(s[:-1])
		return X if len(X) > len(Y) else Y

def minInsertsAndDeletesForAtoB(a,b):
	lcs = lcsCount(a,b)
	inserts = len(b) - lcs
	deletes = len(a) - lcs
	return (inserts, deletes)

def sequencePatternMatching(x,p):
	lcsLen = lcsCountTopDown(x,p)[len(x)][len(p)]
	return len(p) == lcsLen

################################################################################
# --------------------------- Longest Common Substring ----------------------- #
################################################################################

def lcsubstringCountTopDown(x,y):
	t = []
	for i in range(len(x)+1):
		t.append([-1 for j in range(len(y)+1)])
	
	for j in range(len(y)+1):
		t[0][j] = 0
	for i in range(len(x)+1):
		t[i][0] = 0

	maxLength = 0
	for i in range(1, len(x)+1):
		for j in range(1, len(y)+1):
			if x[i-1] == y[j-1]:
				t[i][j] = t[i-1][j-1] + 1
			else:
				t[i][j] = 0
			
			maxLength = max(maxLength, t[i][j])
	
	return maxLength

################################################################################
# ------------------------ Shortest Common Supersequence --------------------- #
################################################################################

def shortestCommonSupersequenceCount(x,y):
	return len(x) + len(y) - lcsCount(x,y)

def shortestCommonSupersequence(x,y):
	if len(x) == 0 or len(y) == 0:
		return x + y

	if x[-1] == y[-1]:
		return shortestCommonSupersequence(x[:-1], y[:-1]) + x[-1]
	else:
		X = shortestCommonSupersequence(x[:-1], y) + x[-1]
		Y = shortestCommonSupersequence(x, y[:-1]) + y[-1]
		return X if len(X) < len(Y) else Y

def shortestCommonSupersequenceDP(x,y):
	t = lcsCountTopDown(x,y)

	i = len(x)
	j = len(y)
	s = ""
	while i > 0 and j > 0:
		if x[i-1] == y[j-1]:
			s += x[i-1]
			i -= 1
			j -= 1
		else:
			if t[i][j] == t[i-1][j]:
				s += x[i-1]
				i -= 1
			else:
				s += y[j-1]
				j -= 1
	
	while i > 0:
		s += x[i-1]
		i -= 1

	while j > 0:
		s += y[j-1]
		j -= 1

	return s[::-1]

################################################################################
# ------------------------- Matrix Chain Multiplication ---------------------- #
################################################################################

def minMCMCost(arr):
	import sys
	#print("Enter:", arr)
	if len(arr) < 2:
		raise Exception("Array too small: ", arr)
	elif len(arr) == 2:
		#print("Early exit:", arr, "Cost:", 0)
		return 0
	elif len(arr) == 3:
		#print("Early exit:", arr, "Cost:", arr[0]*arr[1]*arr[2])
		return arr[0]*arr[1]*arr[2]
	
	i = 0
	j = len(arr)-2
	minCost = sys.maxsize
	for k in range(j):
		#print(arr, i,j,k,"Groups:", arr[i:k+2], arr[k+1:j+2])
		cost = minMCMCost(arr[i:k+2]) + minMCMCost(arr[k+1:j+2]) + (arr[i] * arr[k+1] * arr[j+1])
		minCost = min(minCost, cost)

	#print("Exit:", arr, "Cost:", minCost)
	return minCost

def palindromePartitioning(s):
	import sys
	def isPalindrome(s):
		if len(s) < 2:
			return True
		i = 0
		j = len(s)-1
		while i <= j:
			if s[i] != s[j]:
				return False
			i += 1
			j -= 1
		return True

	if isPalindrome(s):
		return 0	
	
	i = 0
	j = len(s)-1
	minPartitions = sys.maxsize
	for k in range(j):
		cost = palindromePartitioning(s[i:k+1]) + palindromePartitioning(s[k+1:j+1]) + 1
		minPartitions = min(minPartitions, cost)
	
	return minPartitions

def scrambledStringMatching(s1, s2):
	print("Called for:", s1, s2)
	if len(s1) != len(s2):
		return False
	elif s1 == s2:
		return True
	
	for k in range(1,len(s1)):
		print("Pairs:", s1[:k], s2[-k:], "and:", s1[k:], s2[:-k])
		swapped = scrambledStringMatching(s1[:k], s2[-k:]) and scrambledStringMatching(s1[k:], s2[:-k])

		print("Pairs:", s1[:k], s2[:k], "and:", s1[k:], s2[k:])
		unswapped = scrambledStringMatching(s1[:k], s2[:k]) and scrambledStringMatching(s1[k:], s2[k:]) 
		if swapped or unswapped:
			return True

	return False

def eggDropMinCount(e,f) -> int:
	import sys
	import math
	map = {}

	def eggDroppingMinTrials(e,f) -> int:
		if e == 0:
			return sys.maxsize
		elif e == 1 or f < 2:
			return f

		if (e,f) in map:
			return map[(e,f)]

		if e == 2:
			trials = math.ceil((math.sqrt(1+8*f) - 1)/2)
			map[(e,f)] = trials
			return trials
		
		minTrials = sys.maxsize
		minData = 0
		for k in range(1,f+1):
			trials = 1 + max(eggDroppingMinTrials(e-1, k-1), eggDroppingMinTrials(e, f-k))
			if trials < minTrials:
				minData = k
			minTrials = min(trials, minTrials)
		
		print("Floors:", f, "eggs:", e, "trials:", minTrials, "K:", minData, sep="\t")
		map[(e,f)] = minTrials
		return minTrials
	
	return eggDroppingMinTrials(e,f)

################################################################################
# --------------------------- DP GfG Must Do List ---------------------------- #
################################################################################

from functools import lru_cache
import bisect
import sys

#https://practice.geeksforgeeks.org/problems/find-optimum-operation4504/1
def minOperationToReachN(n):
	if n <= 1:
		return n
	
	if n%2 == 0:
		return minOperationToReachN(n//2) + 1
	else:
		return minOperationToReachN(n-1) + 1

#https://practice.geeksforgeeks.org/problems/max-length-chain/1
def maxChainLenGreedy(Parr, n):
	pairs = []
	for i in range(n//2):
		j = 2*i
		pairs.append((Parr[j], Parr[j+1]))
	
	pairs.sort(key=lambda pair: pair[1])
	
	count = 1
	last = pairs[0]
	for pair in pairs[1:]:
		if last[1] < pair[0]:
			count += 1
			last = pair
	return count

def maxChainLenDP(Parr, n):
	Parr.sort(key=lambda x:x.a)
	dp = [None]*n
	dp[0] = 1

	gMax = 1
	for i in range(1, n):
		maxI = 1
		for j in range(i):
			if Parr[j].b < Parr[i].a:
				maxI = max(maxI, dp[j]+1)
		dp[i] = maxI
		gMax = max(maxI, gMax)
	return gMax

#https://practice.geeksforgeeks.org/problems/-minimum-number-of-coins4426/1#
def minimumNumberOfCoins(n):
	denom = [1,2,5,10,20,50,100,200,500,2000]
	currency = []
	for note in reversed(denom):
		currency += [note]*(n//note)
		n = n%note
		if n == 0:
			break
	return currency

#https://practice.geeksforgeeks.org/problems/longest-common-substring1452/1
def longestCommonSubsequence(s1,s2):
	dp = []
	for i in range(len(s1)+1):
		dp.append([0 for j in range(len(s2)+1)])
	
	for i in range(1, len(s1)+1):
		for j in range(1, len(s2)+1):
			if s1[i-1] == s2[j-1]:
				dp[i][j] = dp[i-1][j-1] + 1
			else:
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
	return dp[-1][-1]

#https://practice.geeksforgeeks.org/problems/longest-common-substring/0
def longestCommonSubstring(s1, s2):
	dp, n, m = [], len(s1)+1, len(s2)+1
	for i in range(n):
		dp.append([0 for j in range(m)])
	
	maxl = 0
	for i in range(1, n):
		for j in range(1,m):
			if s1[i-1] == s2[j-1]:
				dp[i][j] = dp[i-1][j-1] + 1
			else:
				dp[i][j] = 0
			maxl = max(maxl, dp[i][j])
	return maxl

#https://practice.geeksforgeeks.org/problems/longest-increasing-subsequence-1587115620/1#
def longestStrictlyIncreasingSubsequenceDP(a):
	dp = [None]*len(a)
	dp[0] = 1

	gMax = 1
	for i in range(1,len(a)):
		maxI = 1
		for j in range(i):
			if a[j] < a[i]:
				maxI = max(maxI, dp[j]+1)
		dp[i] = maxI
		gMax = max(gMax, maxI)
	return gMax

def longestStrictlyIncreasingSubsequenceNLogN(a):
	#https://stackoverflow.com/a/2631810/5952189
	s = [a[0]]
	for i in a[1:]:
		if i > s[-1]:
			s.append(i)
		elif i < s[-1]:
			x = bisect.bisect(s, i)
			s[x] = i
	return len(s)

#https://practice.geeksforgeeks.org/problems/0-1-knapsack-problem0945/1
def knapSack(W, wt, val, n):
	dp = []
	for i in range(n+1):
		dp.append([0 for j in range(W+1)])
	
	for i in range(1,n+1):
		for w in range(1,W+1):
			v = dp[i-1][w]
			if wt[i-1] <= w:
				v = max(v, val[i-1] + dp[i-1][w-wt[i-1]])
			dp[i][w] = v
	return dp[-1][-1]

#https://practice.geeksforgeeks.org/problems/maximum-sum-increasing-subsequence4749/1
def maxSumOfIncreasingSubsequence(arr):
	dp = [None]*len(arr)
	dp[0] = arr[0]

	gMax = arr[0]
	for i in range(1,len(arr)):
		maxI = arr[i]
		for j in range(i):
			if arr[i] > arr[j]:
				maxI = max(maxI,dp[j]+arr[i])
		dp[i] = maxI
		gMax = max(maxI, gMax)
	return gMax

#https://practice.geeksforgeeks.org/problems/minimum-number-of-jumps-1587115620/1
def minimumJumpsRequired(arr, n):
	if n <= 1:
		return 0
	if arr[0] == 0:
		return -1

	jumps = 1
	maxRange = arr[0]
	steps = arr[0]
	for i in range(1,n):
		if i == n-1:
			return jumps

		maxRange = max(maxRange, i+arr[i])
		steps -= 1
		if steps == 0:
			jumps += 1
			if i >= maxRange:
				return -1
			steps = maxRange - i

#https://practice.geeksforgeeks.org/problems/edit-distance3702/1#
def editDistanceDP(s1,s2):
	dp = []
	n,m = len(s1), len(s2)
	for i in range(n+1):
		dp.append([j for j in range(m+1)])
	
	dp[0] = [j for j in range(m+1)]
	for i in range(n+1):
		dp[i][0] = i
	
	for i in range(1, n+1):
		for j in range(1, m+1):
			if s1[i-1] == s2[j-1]:
				dp[i][j] = dp[i-1][j-1]
			else:
				add = dp[i][j-1]
				delete = dp[i-1][j]
				replace = dp[i-1][j-1]
				dp[i][j] = min(add, delete, replace) + 1
	return dp[-1][-1]

#https://practice.geeksforgeeks.org/problems/coin-change/0
def coinChangeII(coins, m, n):
	last = [0]*(n+1)
	last[0] = 1
	current = []
	for i in range(m):
		current = [0]*(n+1)
		current[0] = 1
		for j in range(1, n+1):
			count = last[j]
			if coins[i-1] <= j:
				count += current[j-coins[i-1]]
			current[j] = count
		last = current
	
	return last[-1]

#https://practice.geeksforgeeks.org/problems/path-in-matrix3805/1
def maximumPath(self, N, Matrix):
	dp = []
	for i in range(N-1):
		dp.append([0]*N)
	dp.append(Matrix[-1])

	def maxFor(M,i,j):
		M = dp
		m = M[i+1][j]
		if j-1 >= 0:
			m = max(m, M[i+1][j-1])
		if j+1 < N:
			m = max(m, M[i+1][j+1])
		return m

	for i in reversed(range(N-1)):
		for j in range(N):
			dp[i][j] = Matrix[i][j] + maxFor(i,j)
	return max(dp[0])

#https://practice.geeksforgeeks.org/problems/subset-sum-problem2014/1
def equalPartition(N, arr):
	target = sum(arr)

	if target%2 == 1:
		return 0
	
	def partitionArray(arr, t):
		n = len(arr)
		dp = []
		for i in range(n+1):
			dp.append([False]*(t+1))
			dp[-1][0] = True

		for i in range(1,n+1):
			for j in range(1,t+1):
				r = dp[i-1][j]
				if arr[i-1] <= t:
					r = r or (dp[i-1][j-arr[i-1]])
				dp[i][j] = r
		return 1 if dp[-1][-1] else 0

	return partitionArray(arr, target//2)

#https://practice.geeksforgeeks.org/problems/cutted-segments/0
def maximizeTheCuts(n,x,y,z):
	cuts = [x,y,z]
	dp = [-1]*(n+1)
	for i in range(1, n+1):
		for c in cuts:
			if i == c:
				dp[i] = max(1, dp[i])
			if i > c and dp[i-c] != -1:
				dp[i] = max(dp[i], dp[i-c]+1)
	return 0 if dp[n] == -1 else dp[i]

#https://practice.geeksforgeeks.org/problems/minimum-sum-partition3317/1#
def minimumPartitionSumDifference(arr):
	total, n = sum(arr), len(arr)

	@lru_cache(maxsize=None)
	def closestToTarget(n,t):
		if n == 0 or t == 0:
			return 0

		s = closestToTarget(n-1, t)
		if arr[n-1] <= t:
			s2 = closestToTarget(n-1, t-arr[n-1]) + arr[n-1]
			if s2 <= t:
				s = max(s, s2)
		return s

	s = closestToTarget(n,(total+1)//2)
	#print(s)
	return abs(total - (2*s))

def minimumPartitionSumDifferenceDP(arr):
	total, n = sum(arr), len(arr)
	target = (total+1)//2
	dp = []
	for i in range(n+1):
		dp.append([0 for j in range(target+1)])
	
	for i in range(1, n+1):
		for j in range(1, target+1):
			dp[i][j] = dp[i-1][j]
			if arr[i-1] <= j:
				s2 = dp[i-1][j-arr[i-1]] + arr[i-1]
				if s2 <= target:
					dp[i][j] = max(dp[i][j], s2)
	s = dp[-1][-1] * 2
	return abs(total - s)

def minimumPartitionSumDifferenceDP2(arr):
	total, n = sum(arr), len(arr)
	target = (total+1)//2

	dp = []
	for i in range(n+1):
		dp.append([False for j in range(target+1)])
		dp[-1][0] = True

	for i in range(1,n+1):
		for j in range(1, target+1):
			dp[i][j] = dp[i-1][j]
			if j >= arr[i-1]:
				dp[i][j] = dp[i][j] or (dp[i-1][j-arr[i-1]])

	maxS = 0
	for i in reversed(range(target+1)):
		if dp[-1][i]:
			maxS = i*2
			break
	return abs(total-s)

#https://practice.geeksforgeeks.org/problems/count-number-of-hops-1587115620/1#
def countWaysToReachN(n):
	dp = [1,2,4]
	if n <= 2:
		dp[n-1]

	modulo = 1000000007
	for i in range(4,n+1):
		dp.append(sum(dp))
		dp.pop(0)
		print(i, dp[-1])

	return dp[-1]%modulo

#https://practice.geeksforgeeks.org/problems/egg-dropping-puzzle-1587115620/1
@lru_cache(maxsize=None)
def eggDroppingProblem(n, k):
	if n == 0:
		return sys.maxsize
	if k <= 1 or n == 1:
		return k

	minNeeded = sys.maxsize
	for i in range(k):
		broken = eggDroppingProblem(n-1,i)
		nonBroken = eggDroppingProblem(n,k-i-1)
		trials = max(broken, nonBroken) + 1
		minNeeded = min(trials, minNeeded)
	return minNeeded

#https://practice.geeksforgeeks.org/problems/optimal-strategy-for-a-game-1587115620/1
def optimalStrategyOfGame(arr, n):
	@lru_cache(maxsize=None)
	def calculateMax(l,h):
		#print("Max for:", l, h)
		if l == h:
			#print("Max for:", l, h, "is", arr[l])
			return arr[l]

		pickFirst = calculateMax(l+1,h) 
		pickLast = calculateMax(l,h-1) 
		s = sum(arr[l:h+1]) - min(pickFirst, pickLast)
		#print("Max for:", l, h, "is", s)
		return s
	
	return calculateMax(0,n-1)

def optimalStrategyOfGameDP(arr, n):
	prefixSum = [arr[0]]
	for i in arr[1:]:
		prefixSum.append(prefixSum[-1]+i)

	def arrSum(l,h):
		if l == 0:
			return prefixSum[h]
		else:
			return prefixSum[h] - prefixSum[l-1]

	dp = []
	for i in range(n):
		dp.append([None for j in range(n)])
		dp[-1][i] = arr[i]
	
	for l in reversed(range(n)):
		for h in range(l+1,n):
			pickFirst = dp[l+1][h]
			pickLast = dp[l][h-1]
			dp[l][h] = arrSum(l,h) - min(pickFirst, pickLast)
	return dp[0][n-1]

def shortestCommonSupersequence(X,Y):
	lcsCount = longestCommonSubsequence(X,Y)
	return len(X) + len(Y) - lcsCount

#https://practice.geeksforgeeks.org/problems/flip-bits0240/1#
def maxOnes(a, n):
    ones, currSum, minSum, maxZeroes = 0, 0, 0, 0
    for b in a:
        ones += 1 if b == 1 else 0
        currSum += 1 if b == 0 else -1
        maxZeroes = max(maxZeroes, currSum-minSum)
        minSum = min(minSum, currSum)
    return ones+maxZeroes

#https://leetcode.com/problems/house-robber/
def rob(nums) -> int:
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def robCached(n):
        if n == 0:
            return 0
        elif n == 1:
            return nums[0]
        elif n == 2:
            return max(nums[:2])

        tl = nums[n-1] + robCached(n-2)
        ll = robCached(n-1)
        return max(ll, tl)

    return robCached(len(nums))

#https://leetcode.com/problems/house-robber-ii/
def robCircular(nums):
    tl = nums[-1] + rob(nums[1:-2])
    ll = rob(nums[:-1])
    return max(tl, ll)

#https://leetcode.com/problems/count-vowels-permutation/
def countVowelPermutation(n):
    ruleMap = { 'a': ['e'],
                'e': ['a', 'i'],
                'i': ['a', 'e', 'o', 'u'],
                'o': ['u', 'i'],
                'u': ['a']
            }

from functools import lru_cache

    @lru_cache(maxsize=None)
    def countUtil(n, last):
        if n == 0:
            return 1
        elif n == 1:
            return len(ruleMap[last])
        else:
            total = 0
            for k in ruleMap[last]:
                total += countUtil(n-1, k)
            return total

    total, l = 0, (10**9 + 7)
    for k in ruleMap.keys():
        total += countUtil(n-1, k)%l
    return total%l

#https://leetcode.com/problems/maximum-length-of-repeated-subarray/
def longestCommonSubarray(nums1, nums2):
    n, m = len(nums1), len(nums2)
    dp = [[0]*(m+1)]
    for _ in range(n):
        dp.append([None]*(m+1))

    for r in range(1, n):
        dp[r][0] = 0

    lcs = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            if nums1[i-1] == nums2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = 0
            lcs = max(lcs, dp[i][j])
    return lcs

#https://www.geeksforgeeks.org/longest-monotonically-increasing-subsequence-size-n-log-n/amp/
def longestIncreasingSubsequence(nums):
    def findCeilBS(arr, k):
        lo, hi = 0, len(arr)-1
        while lo < hi:
            mid = lo + (hi-lo)//2
            if arr[mid] >= k:
                hi = mid
            else:
                lo = mid+1
        return lo

    lis = [nums[0]]
    for n in nums[1:]:
        if n > lis[-1]:
            lis.append(n)
        else:
            ceil = findCeilBS(lis, n)
            lis[ceil] = n

    return len(lis)

#https://leetcode.com/problems/russian-doll-envelopes/
def maxEnvelopes(envelopes):
    envelopes.sort(key=lambda x: (x[0], -x[1]))
    heights = list(map(lambda x: x[1], envelopes))
    return longestIncreasingSubsequence(heights)

#https://leetcode.com/problems/minimum-number-of-removals-to-make-mountain-array/
#https://leetcode.com/problems/minimum-number-of-removals-to-make-mountain-array/discuss/952053/Python-3-solutions%3A-LIS-dp-O(n-log-n)-explained
def makeMountainArray(nums):
    def lisSizeArray(nums):
        def ceil(arr, k):
            lo, hi = 0, len(arr)-1
            while lo < hi:
                mid = lo + (hi-lo)//2
                if arr[mid] >= k:
                    hi = mid
                else:
                    lo = mid+1
            return lo

        lis, lisL = nums[:1], [1]
        for n in nums[1:]:
            if n > lis[-1]:
                lis.append(n)
                lisL.append(len(lis))
            else:
                i = ceil(lis, n)
                lis[i] = n
                lisL.append(i+1)
        return lisL
    
    lis, lds = lisSizeArray(nums), lisSizeArray(nums[::-1])[::-1]
    mountainSize = 0
    for i in range(len(nums)):
        if lis[i] >= 2 and lds[i] >= 2:
            mountainSize = max(mountainSize, lis[i] + lds[i] - 1)
    return len(nums) - mountainSize

#https://practice.geeksforgeeks.org/problems/unique-bsts-1587115621/1
def uniqueBst(n):
    dp = [None]*(n+1)
    dp[0:2] = [1, 1]

    M = 1000000007
    for i in range(2, n+1):
        count = 0
        for k in range(i):
            count += dp[k] * dp[i-k-1]
        dp[i] = count%M
    return dp[n]

#https://leetcode.com/problems/01-matrix/
def closestZeroes(mat):
    r, c = len(mat), len(mat[0])
    dp = [[r+c]*c for _ in range(r)]

    for i in range(r):
        for j in range(c):
            if mat[i][j] == 0:
                dp[i][j] = 0
                continue

            if j > 0:
                dp[i][j] = min(dp[i][j], dp[i][j-1]+1)
            if i > 0:
                dp[i][j] = min(dp[i][j], dp[i-1][j]+1)

    for i in reversed(range(r)):
        for j in reversed(range(c)):
            if mat[i][j] == 0:
                dp[i][j] = 0
                continue

            if j < c-1:
                dp[i][j] = min(dp[i][j], dp[i][j+1]+1)
            if i < r-1:
                dp[i][j] = min(dp[i][j], dp[i+1][j]+1)

    return dp
