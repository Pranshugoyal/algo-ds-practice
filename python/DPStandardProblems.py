
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
		return s1 == s2
	
	for k in range(1,len(s1)):
		print("Pairs:", s1[:k], s2[-k:], "and:", s1[k:], s2[:-k])
		swapped = scrambledStringMatching(s1[:k], s2[-k:]) and scrambledStringMatching(s1[k:], s2[:-k])

		print("Pairs:", s1[:k], s2[:k], "and:", s1[k:], s2[k:])
		unswapped = scrambledStringMatching(s1[:k], s2[:k]) and scrambledStringMatching(s1[k:], s2[k:]) 
		if swapped or unswapped:
			return True

	return False

