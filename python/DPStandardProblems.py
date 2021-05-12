
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
# ---------------------- Unbounded Knapsack Problems ------------------------- #
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

